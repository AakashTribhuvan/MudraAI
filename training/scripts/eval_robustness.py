"""How the live recognizer copes with real-life signing: low camera frame rates and imperfect signs.

Each held-out clip is replayed through the live path (StreamingRecognizer: segmenter, model, taught-word bank)
as a camera stream with the hands out of view before and after it, under these conditions:

- ``fps``: the camera delivers only this many frames per second (random phase), e.g. 12 / 8 / 6;
- ``drop``: in dim light the tracker loses a hand in this share of frames;
- ``fast`` / ``slow``: the sign is performed 1.35x faster / 0.75x slower;
- ``late`` / ``early``: the first 20% of the sign is missing (started late) / the last 15% (stopped early);
- ``small``: the hands move 25% less (a lazy or tired sign).

A take counts as **right** when the app adds the label and nothing else, **wrong** when it adds another word,
**missed** when it adds nothing (the app adds "ok" results and "unsure" results whose best guess is >= 0.5 with
hands in >= 50% of the segment, as CommunicatePage does). Clips:

- ``owner``: the signer's recorded takes held out from the split (test + val) plus takes recorded after it;
  the taught-word bank is built from the split's train takes only, as the server would build it;
- ``other``: every other test clip of the split (signers the model never saw), for "does it break other words".

    python training/scripts/eval_robustness.py --conditions base fps8 fps6 fps8+drop
    python training/scripts/eval_robustness.py --set '{"tta": 3}' --sets owner
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from mudra_ml.config import project_path
from mudra_ml.fewshot import CustomWordBank, FewShotConfig
from mudra_ml.preprocessing.sequence import LandmarkSequence, load_sequence
from mudra_ml.streaming import Frame, ModelBundle, SegmenterConfig, StreamingRecognizer

AUTO_ADD_MIN = 0.5  # CommunicatePage: an unsure sign still adds its best guess at this confidence


def clip_times(seq: LandmarkSequence) -> np.ndarray:
    t = np.asarray(seq.timestamps_ms, np.float64) / 1000
    if len(t) < 2 or np.any(np.diff(t) <= 0) or t[-1] - t[0] < 0.1:
        fps = float(seq.meta.get("fps") or 0) or 25.0
        t = np.arange(seq.num_frames) / fps
    return t - t[0]


def active_span(seq: LandmarkSequence) -> tuple[int, int]:
    """Frames where a hand is visible (the sign without the leading / trailing rest)."""
    idx = np.flatnonzero(seq.hand_present.any(axis=1))
    return (int(idx[0]), int(idx[-1])) if len(idx) else (0, seq.num_frames - 1)


def perturb(seq: LandmarkSequence, cond: str, rng: np.random.Generator) -> tuple[LandmarkSequence, np.ndarray]:
    """(sequence, frame times in s) under one condition like 'fps8+drop+late'."""
    t = clip_times(seq)
    hands, present = seq.hands.copy(), seq.hand_present.copy()
    pose, pose_present = seq.pose.copy(), seq.pose_present.copy()
    keep = np.arange(seq.num_frames)
    for part in cond.split("+"):
        if part == "base":
            continue
        if part in ("late", "early"):
            a, b = active_span(seq)
            n = b - a + 1
            if part == "late":  # the first 20% of the sign never reaches the camera (hands enter later)
                present[a: a + int(0.2 * n)] = False
            else:
                keep = keep[keep <= b - int(0.15 * n)]
        elif part in ("fast", "slow"):
            t = t / (1.35 if part == "fast" else 0.75)
        elif part == "small":
            for h in range(2):
                m = present[:, h]
                if m.sum() > 1:
                    centre = hands[m, h, 0, :2].mean(axis=0)  # mean wrist position
                    shift = (hands[:, h, 0, :2] - centre) * -0.25
                    hands[:, h, :, :2] += shift[:, None, :] * m[:, None, None]
                    for wi, hi in ((15, 0), (16, 1)):  # the pose wrists follow the hands
                        if hi == h:
                            pose[:, wi, :2] += shift * m[:, None]
        elif part == "drop":
            present &= rng.random(present.shape) >= 0.2
        elif part.startswith("fps"):
            fps = float(part[3:])
            grid = np.arange(rng.uniform(0, 1 / fps), t[keep[-1]], 1 / fps)
            sel = np.unique(np.searchsorted(t[keep], grid).clip(0, len(keep) - 1))
            if len(sel) >= 4:  # a clip this short is left as it is
                keep = keep[sel]
        else:
            raise ValueError(f"unknown condition {part!r}")
    out = LandmarkSequence(hands[keep], present[keep], present[keep].astype(np.float32), pose[keep], pose_present[keep],
                           (t[keep] * 1000).astype(np.int64), dict(seq.meta))
    return out, t[keep]


def replay(rec: StreamingRecognizer, seq: LandmarkSequence, t: np.ndarray) -> list[dict]:
    """Stream the clip with 0.6 s of hands out of view before and 2 s after; returns the sign_ended results."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.04
    no_hands = np.zeros_like(seq.hands[0])
    frames = [Frame(-0.6 + k * dt, no_hands, np.zeros(2, bool), seq.pose[0], bool(seq.pose_present[0]))
              for k in range(int(0.6 / dt))]
    frames += [Frame(float(t[i]), seq.hands[i], seq.hand_present[i], seq.pose[i], bool(seq.pose_present[i]))
               for i in range(seq.num_frames)]
    frames += [Frame(float(t[-1]) + (k + 1) * dt, no_hands, np.zeros(2, bool), seq.pose[-1], bool(seq.pose_present[-1]))
               for k in range(int(2.0 / dt))]
    out = []
    for f in frames:
        ev = rec.push(f)
        if ev and ev["event"] == "sign_ended":
            out.append(ev)
    return out


def added_words(events: list[dict]) -> list[tuple[str, bool]]:
    """(word, sure) for each word the app would add."""
    words = []
    for ev in events:
        r = ev["result"]
        if r["status"] == "ok":
            words.append((r["sign"], True))
        elif r["status"] in ("uncertain", "no_sign") and r.get("top_k") and ev["stats"]["hand_rate"] >= 0.5 \
                and r["top_k"][0]["confidence"] >= AUTO_ADD_MIN:
            words.append((r["top_k"][0]["sign"], False))
    return words


def load_clips(split: str) -> tuple[list[dict], list[dict], list[dict]]:
    """(owner held-out takes, other test clips, owner train takes)."""
    rows = {p: list(csv.DictReader(open(project_path(f"data/metadata/splits/{split}/{p}.csv"), encoding="utf-8")))
            for p in ("train", "val", "test")}
    seen = {r["sample_id"].split("#")[0] for p in rows.values() for r in p}
    train = {r["sample_id"].split("#")[0]: r for r in rows["train"] if r["source"] == "app"}
    owner = [r for p in ("test", "val") for r in rows[p] if r["source"] == "app"]
    manifest = project_path("data/metadata/manifest_app.csv")
    for r in csv.DictReader(open(manifest, encoding="utf-8")):  # takes recorded after the split was made
        if r["sample_id"] not in seen and not r["quality_flag"] and Path(project_path(r["landmarks_path"])).exists():
            owner.append({**r, "split": "new"})
    other = [r for r in rows["test"] if r["source"] != "app"]
    return owner, other, list(train.values())


_EMB: dict[str, np.ndarray] = {}


def take_embedding(bundle: ModelBundle, r: dict) -> np.ndarray:
    if r["sample_id"] not in _EMB:
        seq = load_sequence(project_path(r["landmarks_path"]))
        _EMB[r["sample_id"]] = bundle.embed([seq], int(seq.meta["width"]), int(seq.meta["height"]))[0]
    return _EMB[r["sample_id"]]


def build_bank(bundle: ModelBundle, takes: list[dict], classes: set[str], cfg: FewShotConfig,
               leave_out: str | None = None) -> CustomWordBank:
    """The server's rule (recordings.py rebuild_bank): new words always, known words from 3 good takes."""
    ex: dict[str, list] = defaultdict(list)
    for r in takes:
        if r["sample_id"].split("#")[0] != leave_out:
            ex[r["label"]].append(take_embedding(bundle, r))
    bank = CustomWordBank(cfg)
    bank.set_examples({k: np.stack(x) for k, x in ex.items() if k not in classes or len(x) >= 3})
    return bank


def all_takes() -> list[dict]:
    """Every good take in the app manifest (what the live server's bank is built from)."""
    rows = csv.DictReader(open(project_path("data/metadata/manifest_app.csv"), encoding="utf-8"))
    return [r for r in rows if not r["quality_flag"] and Path(project_path(r["landmarks_path"])).exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="app/backend/models/isl_mudra_combined_v15b_bilstm")
    ap.add_argument("--split", default="combined_words_v9_app_r6")
    ap.add_argument("--config", default="training/configs/streaming.json")
    ap.add_argument("--set", default=None, help="JSON settings on top of --config")
    ap.add_argument("--sets", nargs="+", default=["owner", "other"])
    ap.add_argument("--conditions", nargs="+", default=["base", "fps12", "fps8", "fps6", "fps8+drop", "fps8+fast",
                                                         "fps8+slow", "fps8+late", "fps8+early", "fps8+small"])
    ap.add_argument("--repeats", type=int, default=1, help="random draws per clip and condition (owner clips)")
    ap.add_argument("--no-bank", action="store_true")
    ap.add_argument("--bank-mode", choices=["strict", "live"], default="strict",
                    help="strict: bank from the split's train takes; live: from every other recorded take (as the server)")
    ap.add_argument("--fewshot", default=None, help="JSON settings on top of training/configs/fewshot.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write per-clip outcomes as JSON")
    args = ap.parse_args()

    bundle = ModelBundle(project_path(args.model), device=args.device)
    cfg = SegmenterConfig.load(project_path(args.config))
    if args.set:
        cfg = SegmenterConfig(**{**cfg.to_dict(), **json.loads(args.set)})
    cfg = SegmenterConfig(**{**cfg.to_dict(), "provisional_s": 1e9})
    bundle.use_cleaning(cfg)  # as the server does, before the taught-word bank is built
    owner, other, train = load_clips(args.split)
    fs = FewShotConfig.load(project_path("training/configs/fewshot.json"))
    if args.fewshot:
        fs = FewShotConfig(**{**fs.to_dict(), **json.loads(args.fewshot)})
    classes = set(bundle.classes)
    bank = None if args.no_bank else build_bank(bundle, train, classes, fs)
    live_takes = all_takes() if args.bank_mode == "live" else None
    clips = {"owner": owner, "other": other}
    detail = []
    print(f"model {bundle.name}; settings {json.dumps(args.set) if args.set else 'default'}; bank "
          f"{'off' if bank is None else len(bank)} words ({args.bank_mode}, {fs.to_dict()})")
    header = f"{'set':6s} {'condition':16s} {'clips':>5s} {'right':>6s} {'sure':>6s} {'wrong':>6s} {'missed':>6s}"
    print(header)
    for name in args.sets:
        for cond in args.conditions:
            res = Counter()
            reps = args.repeats if name == "owner" else 1
            for ci, r in enumerate(clips[name]):
                seq0 = load_sequence(project_path(r["landmarks_path"]))
                clip_bank = bank
                if live_takes is not None and bank is not None and name == "owner":
                    clip_bank = build_bank(bundle, live_takes, classes, fs, leave_out=r["sample_id"].split("#")[0])
                w, h = int(seq0.meta.get("width", 0) or 640), int(seq0.meta.get("height", 0) or 480)
                for k in range(reps):
                    rng = np.random.default_rng(zlib.crc32(f"{ci}:{k}:{cond}".encode()))
                    seq, t = perturb(seq0, cond, rng)
                    rec = StreamingRecognizer(bundle, cfg, w, h, clip_bank)
                    words = added_words(replay(rec, seq, t))
                    label = r["label"]
                    wrong = [x for x, _ in words if x != label]
                    res["clips"] += 1
                    if wrong:
                        res["wrong"] += 1
                        outcome = "wrong"
                    elif words:
                        res["right"] += 1
                        res["sure"] += any(s for _, s in words)
                        outcome = "right"
                    else:
                        res["missed"] += 1
                        outcome = "missed"
                    if args.out:
                        detail.append({"set": name, "cond": cond, "id": r["sample_id"], "label": label,
                                       "split": r.get("split"), "outcome": outcome, "words": words})
            n = max(res["clips"], 1)
            print(f"{name:6s} {cond:16s} {res['clips']:5d} {res['right'] / n:6.1%} {res['sure'] / n:6.1%}"
                  f" {res['wrong'] / n:6.1%} {res['missed'] / n:6.1%}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(detail), encoding="utf-8")


if __name__ == "__main__":
    main()
