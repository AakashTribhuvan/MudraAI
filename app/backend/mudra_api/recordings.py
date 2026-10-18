"""Signs recorded in the app: stored for retraining and used as few-shot custom words.

Only landmarks are stored (``.npz`` in the training format), never video. Every recording
carries the signer's name (``signer_id``) for a signer-independent test, and a session group
(signer + day) so takes from one sitting never cross splits. Rows go to
``data/metadata/manifest_app.csv`` in the standard manifest format, ready for ``make_splits.py``.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import numpy as np
from mudra_ml import schema
from mudra_ml.data.manifest import (
    ManifestRow,
    normalize_label,
    read_manifest,
    write_manifest,
)
from mudra_ml.fewshot import CustomWordBank, FewShotConfig
from mudra_ml.preprocessing.sequence import (
    LandmarkSequence,
    assign_hands,
    load_sequence,
)
from mudra_ml.streaming import Frame, ModelBundle

SOURCE = "app"
MIN_DURATION_S = 0.5
MIN_HAND_RATE = 0.3


def parse_frame(msg: dict) -> Frame:
    """One client frame {t, hands: [{landmarks, handedness, score}], pose} -> Frame (signer's left/right)."""
    pose = np.asarray(msg["pose"], np.float32) if msg.get("pose") else None
    if pose is not None and pose.shape[1] == 3:
        pose = np.concatenate([pose, np.ones((len(pose), 1), np.float32)], axis=1)
    raw = msg.get("hands", [])[:2]
    hands, present, _ = assign_hands(
        [np.asarray(h["landmarks"], np.float32) for h in raw],
        [h.get("handedness") or "" for h in raw],
        [h.get("score", 1.0) for h in raw],
        pose,
        False,
    )
    return Frame(
        float(msg["t"]), hands, present,
        pose if pose is not None else np.zeros((schema.NUM_POSE_LANDMARKS, 4), np.float32),
        pose is not None,
    )


def frames_to_sequence(frames: list[Frame], width: int, height: int, meta: dict) -> LandmarkSequence:
    t0 = frames[0].t
    duration = max(frames[-1].t - t0, 1e-6)
    return LandmarkSequence(
        hands=np.stack([f.hands for f in frames]).astype(np.float32),
        hand_present=np.stack([f.present for f in frames]),
        hand_score=np.stack([f.present for f in frames]).astype(np.float32),
        pose=np.stack([f.pose for f in frames]).astype(np.float32),
        pose_present=np.array([f.pose_present for f in frames]),
        timestamps_ms=np.array([round((f.t - t0) * 1000) for f in frames]),
        meta={**meta, "width": width, "height": height, "fps": round((len(frames) - 1) / duration, 3)},
    )


def signer_key(name: str) -> str:
    return normalize_label(name).lower() or "anonymous"


# A word the model already knows gets the signer's own examples once it has this many good recordings
PERSONAL_MIN_EXAMPLES = 3
# A take "looks like" another known word when the trained model picks that word at this confidence or more
LOOKS_LIKE_CONFIDENCE = 0.7


class RecordingStore:
    """Thread-safe store of app recordings plus the custom-word bank built from them."""

    def __init__(self, data_root: Path, bundle: ModelBundle | None, fewshot_cfg: FewShotConfig):
        self.root = Path(data_root)
        self.landmarks_dir = self.root / "processed" / "landmarks" / SOURCE
        self.manifest_path = self.root / "metadata" / f"manifest_{SOURCE}.csv"
        self.bundle = bundle
        self.known = set(bundle.classes) if bundle else set()
        self.bank = CustomWordBank(fewshot_cfg)
        self._lock = threading.Lock()
        self.rows: list[ManifestRow] = read_manifest(self.manifest_path) if self.manifest_path.exists() else []
        self._embeddings: dict[str, np.ndarray] = {}
        self._probs: dict[str, np.ndarray] = {}
        self.rebuild_bank()

    # -- custom words -------------------------------------------------------------------------
    def _embedding(self, row: ManifestRow) -> np.ndarray:
        if row.sample_id not in self._embeddings:
            seq = load_sequence(Path(row.landmarks_path))
            p, e = self.bundle.probs_and_embeddings([seq], int(seq.meta["width"]), int(seq.meta["height"]))
            self._embeddings[row.sample_id], self._probs[row.sample_id] = e[0], p[0]
        return self._embeddings[row.sample_id]

    def quality(self) -> dict[str, dict]:
        """Per recorded word: the bank's barrier report (consistency, takes left out, nearest other custom
        word, too_close) plus which known word the trained model reads its takes as (``looks_like``)."""
        if self.bundle is None:
            return {}
        report = self.bank.report()
        out: dict[str, dict] = {}
        by_label: dict[str, list[ManifestRow]] = {}
        for r in self.rows:
            if not r.quality_flag and Path(r.landmarks_path).exists():
                by_label.setdefault(r.label, []).append(r)
        for label, rows in by_label.items():
            entry = dict(report.get(label, {"takes": len(rows), "used": 0, "left_out": [], "too_close": False}))
            entry["left_out"] = [rows[i].sample_id for i in entry.get("left_out", []) if i < len(rows)]
            reads = {}
            for r in rows:
                self._embedding(r)
                d = self.bundle.decide(self._probs[r.sample_id], threshold=LOOKS_LIKE_CONFIDENCE)
                if d["status"] == "ok" and d["sign"] != label:
                    reads[d["sign"]] = reads.get(d["sign"], 0) + 1
            other = max(reads, key=reads.get) if reads else None
            entry["looks_like"] = {"sign": other, "takes": reads[other]} if other and reads[other] * 2 >= len(rows) else None
            out[label] = entry
        return out

    def rebuild_bank(self) -> None:
        """Custom-word bank: new words from their first good recording; words the model already knows
        once they have PERSONAL_MIN_EXAMPLES good recordings (the signer's own variant, e.g. a COMPUTER
        the model does not recognise: audit of 2026-10-15)."""
        if self.bundle is None:
            return
        examples: dict[str, list[np.ndarray]] = {}
        for r in self.rows:
            if not r.quality_flag and Path(r.landmarks_path).exists():
                examples.setdefault(r.label, []).append(self._embedding(r))
        active = {k: np.stack(v) for k, v in examples.items() if k not in self.known or len(v) >= PERSONAL_MIN_EXAMPLES}
        self.bank.set_examples(active)

    # -- recordings ---------------------------------------------------------------------------
    def add(self, label: str, signer: str, frames: list[Frame], width: int, height: int) -> dict:
        label = normalize_label(label)
        if not label:
            raise ValueError("label is empty")
        if len(frames) < 8:
            raise ValueError("recording has fewer than 8 frames")
        frames = sorted(frames, key=lambda f: f.t)
        sid = uuid.uuid4().hex[:12]
        signer_id = signer_key(signer)
        seq = frames_to_sequence(frames, width, height, {"label": label, "source": SOURCE, "sample_id": sid,
                                                         "signer": signer_id, "recorded_at": int(time.time())})
        rates = seq.detection_rates()
        duration = (frames[-1].t - frames[0].t)
        flag = ""
        if duration < MIN_DURATION_S:
            flag = "too_short"
        elif rates["any_hand"] < MIN_HAND_RATE:
            flag = "low_hand_detection"
        elif rates["pose"] == 0:
            flag = "no_pose"
        path = self.landmarks_dir / label / f"{sid}.npz"
        seq.save(path)
        row = ManifestRow(
            sample_id=sid, label=label, source=SOURCE, kind="landmarks", signer_id=f"{SOURCE}:{signer_id}",
            group_id=f"{SOURCE}:session:{signer_id}:{time.strftime('%Y%m%d')}", raw_path=str(path), landmarks_path=str(path),
            num_frames=seq.num_frames, fps=seq.meta["fps"], duration_s=round(duration, 3),
            any_hand_rate=round(rates["any_hand"], 4), pose_rate=round(rates["pose"], 4), quality_flag=flag,
        )
        with self._lock:
            self.rows.append(row)
            write_manifest(self.rows, self.manifest_path)
            if not flag:
                self.rebuild_bank()
        return {**self._public(row), "label_count": self.count(label)}

    def delete(self, sample_id: str) -> bool:
        with self._lock:
            row = next((r for r in self.rows if r.sample_id == sample_id), None)
            if row is None:
                return False
            self.rows = [r for r in self.rows if r.sample_id != sample_id]
            Path(row.landmarks_path).unlink(missing_ok=True)
            self._embeddings.pop(sample_id, None)
            self._probs.pop(sample_id, None)
            write_manifest(self.rows, self.manifest_path)
            self.rebuild_bank()
            return True

    def count(self, label: str) -> int:
        return sum(1 for r in self.rows if r.label == label and not r.quality_flag)

    def summary(self) -> list[dict]:
        quality = self.quality()
        by_label: dict[str, list[ManifestRow]] = {}
        for r in self.rows:
            by_label.setdefault(r.label, []).append(r)
        out = []
        for label, rows in sorted(by_label.items()):
            good = [r for r in rows if not r.quality_flag]
            out.append({
                "sign": label, "recordings": len(good), "rejected": len(rows) - len(good),
                "signers": sorted({r.signer_id.split(":", 1)[-1] for r in good}),
                "in_model": label in self.known, "custom_active": label in self.bank.labels,
                "items": [self._public(r) for r in rows],
                "quality": quality.get(label),
            })
        return out

    @staticmethod
    def _public(row: ManifestRow) -> dict:
        return {k: getattr(row, k) for k in ("sample_id", "label", "duration_s", "any_hand_rate", "quality_flag")} | {
            "signer": row.signer_id.split(":", 1)[-1]
        }
