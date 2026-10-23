"""Real-time sign segmentation: resting -> signing -> ended, then classify the whole sign once.

Shared by the live API (WebSocket /stream) and the continuous-stream benchmark, so the
app and its evaluation run the same logic. Time-based (seconds), so browser frame-rate
changes do not alter behaviour.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from mudra_ml import schema
from mudra_ml.datasets.isl_dataset import BACKGROUND_LABEL, FeatureConfig, sequence_features
from mudra_ml.models.sequence_model import ModelConfig, build_model
from mudra_ml.preprocessing.resample import clean
from mudra_ml.preprocessing.sequence import LandmarkSequence


@dataclass
class SegmenterConfig:
    raise_y: float = 1.5  # wrist higher than this many shoulder-widths below the shoulder line = raised
    speed: float = 1.0  # wrist speed (shoulder-widths per second) that counts as movement
    on_s: float = 0.15  # continuous activity needed to start a sign
    off_s: float = 0.35  # continuous rest needed to end a sign
    pad_pre_s: float = 0.3  # context kept before the detected start (training clips start at rest)
    pad_post_s: float = 0.2
    min_sign_s: float = 0.3  # shorter bursts are ignored (fidgets, detection glitches)
    max_sign_s: float = 6.0  # a sign running this long is classified anyway
    buffer_s: float = 12.0
    commit_threshold: float | None = None  # whole-sign confidence needed to add a word (None = model card value)
    margin: float = 0.0  # top-1 must beat top-2 by this much (look-alike signs become "uncertain")
    # "I don't know this sign": a confident word whose embedding is outside that word's training boundary
    # (KnownWordBoundaries, model_dir/known_boundaries.npz) becomes "uncertain". None = off.
    known_boundary_k: float | None = None
    tta: int = 1  # smoothing: classify this many slightly different cuts of the sign and average them
    cooldown_s: float = 1.0  # the same word is not added again within this time (double triggers)
    provisional_s: float = 0.5  # while signing, send a live guess this often (UI only, never committed)
    # Parked hands: at a desk in front of a close webcam the hands rest *in view*, where they always
    # count as raised. Hands that stay still this long count as resting wherever they are; the rest
    # starts when the stillness began, and only movement starts the next sign. 0 = off.
    still_s: float = 0.0
    still_speed: float = 0.5  # mean speed of the hand's 21 points (shoulder widths / s) below which it is still
    still_window_s: float = 0.25  # speed for stillness is measured over this span, so tracking jitter does not grow with fps
    still_below_y: float = -10.0  # parking only applies to hands lower than this (shoulder widths below the shoulder line)
    still_keep_s: float = 0.0  # a sign ended by stillness keeps this much of the still part (a held final handshape)
    # Fragment joining: a piece shorter than this that ends by stillness is held back (reported as
    # too_short); if movement starts again within join_gap_s, the next sign starts from the piece's
    # start, so a sign with a pause is classified whole. 0 = off.
    join_short_s: float = 0.0
    join_gap_s: float = 1.0
    # A held-back piece that is not continued within join_gap_s is classified on its own (a short sign that ends
    # in a held handshape, e.g. FOOD at the mouth, was otherwise dropped) with this much of its held part kept.
    flush_fragments: bool = False
    fragment_keep_s: float = 0.0
    # A sign that ends in a still pause with an unsure result is held this long (0 = off): when signing resumes,
    # the two parts are classified as one sign (a phrase with a pause); the parts' own results are used when the
    # whole is not more confident. Not resumed in time: the held part is reported on its own.
    hold_unsure_s: float = 0.0
    # Second look: an unsure sign is classified again with more context before / after it (the hands entered the
    # camera late, or the cut ended early); the most confident of these that passes the threshold is used.
    rescue_cuts: bool = False
    rescue_threshold: float | None = None  # confidence a second-look result needs (None = commit_threshold)
    # Input clean-up before the model (preprocessing/resample.py): a sign recorded below clean_below_fps is put on an
    # even clean_fps time grid, and a hand lost for at most fill_gap_s between two sightings is filled in. 0 = off.
    clean_below_fps: float = 0.0
    clean_fps: float = 25.0
    fill_gap_s: float = 0.0

    @classmethod
    def load(cls, path: Path | None) -> SegmenterConfig:
        if path and Path(path).exists():
            return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Frame:
    t: float
    hands: np.ndarray  # [2, 21, 3] signer-left / signer-right
    present: np.ndarray  # [2] bool
    pose: np.ndarray  # [33, 4]
    pose_present: bool


MOVE_SPAN_S = 0.1  # wrist speed for "moving" is measured over at least this span


class SignSegmenter:
    """Hysteresis state machine over a per-frame activity signal (hands raised or moving)."""

    def __init__(self, cfg: SegmenterConfig, width: int, height: int):
        self.cfg = cfg
        self.aspect = width / height if width > 0 and height > 0 else 1.0
        self.frames: deque[Frame] = deque()
        self.state = "idle"
        self.active_since: float | None = None
        self.rest_since: float | None = None
        self.sign_start: float | None = None
        self.body: tuple[np.ndarray, float] | None = None  # shoulder centre, shoulder width
        self.point_hist: list[deque] = [deque(), deque()]  # (t, points) per hand over the last still_window_s
        self.still_since: float | None = None  # all visible hands still (and low enough) since then
        self.parked = False
        self.pending: tuple[float, float] | None = None  # (start, end) of a held-back short piece
        self.pending_held = False  # the held-back piece is an unsure sign (hold), not a short fragment
        self.joined: tuple[float, float, bool] | None = None  # current sign continues (piece start, piece end, held)
        self.own_start: float | None = None  # where the current sign's own movement started

    def _activity(self, f: Frame) -> bool:
        if f.pose_present:
            ls, rs = f.pose[schema.POSE_LEFT_SHOULDER], f.pose[schema.POSE_RIGHT_SHOULDER]
            if ls[3] >= 0.5 and rs[3] >= 0.5:
                c = (ls[:2] + rs[:2]) / 2 * [self.aspect, 1]
                w = float(np.linalg.norm((ls[:2] - rs[:2]) * [self.aspect, 1]))
                if w > 1e-4:
                    self.body = (c, w)
        raised = moving = False
        still = any(f.present)  # every visible hand still (unknown speed does not break it)
        for h in range(2):
            if not f.present[h]:
                self.point_hist[h].clear()
                continue
            points = f.hands[h, :, :2] * [self.aspect, 1]
            wrist = points[schema.WRIST]
            if self.body is None:
                raised = True  # no body reference: a visible hand is the only evidence
            else:
                c, w = self.body
                height = (wrist[1] - c[1]) / w
                raised = raised or height < self.cfg.raise_y
                hist = self.point_hist[h]
                # wrist speed over >= 0.1 s (the previous frame at low fps): at 30 fps frame-to-frame
                # tracking jitter alone would read as movement
                ref = next(((t0, p0) for t0, p0 in reversed(hist) if f.t - t0 >= MOVE_SPAN_S), hist[0] if hist else None)
                if ref is not None and f.t > ref[0]:
                    moving = moving or np.linalg.norm(wrist - ref[1][schema.WRIST]) / w / (f.t - ref[0]) > self.cfg.speed
                while len(hist) > 1 and f.t - hist[1][0] >= self.cfg.still_window_s:
                    hist.popleft()
                if height < self.cfg.still_below_y:
                    still = False
                elif hist and f.t - hist[0][0] >= 0.6 * self.cfg.still_window_s:
                    t0, p0 = hist[0]
                    speed = float(np.linalg.norm(points - p0, axis=1).mean()) / w / (f.t - t0)
                    still = still and speed < self.cfg.still_speed
                hist.append((f.t, points))
        if self.cfg.still_s > 0 and self.body is not None:
            if not still or moving:
                self.still_since, self.parked = None, False
            else:
                self.still_since = self.still_since if self.still_since is not None else f.t
                self.parked = f.t - self.still_since >= self.cfg.still_s
        return moving or (raised and not self.parked)

    def current(self) -> LandmarkSequence | None:
        """The sign in progress so far (for live provisional guesses)."""
        if self.state != "signing":
            return None
        return self._segment(self.sign_start - self.cfg.pad_pre_s, self.frames[-1].t)

    def variants(self, t0: float, t1: float, k: int) -> list[LandmarkSequence]:
        """k cuts of one sign with slightly different start/end padding (test-time smoothing)."""
        shifts = [(0.0, 0.0), (-0.15, 0.0), (0.15, 0.0), (0.0, 0.15), (-0.15, 0.15)][: max(1, k)]
        segs = [self._segment(t0 + a, t1 + b) for a, b in shifts]
        return [s for s in segs if s is not None]

    def push(self, f: Frame) -> tuple[str | None, LandmarkSequence | None]:
        """Add one frame. Returns (event, segment): event is 'sign_started', 'sign_ended' or None."""
        cfg = self.cfg
        self.frames.append(f)
        while self.frames and f.t - self.frames[0].t > cfg.buffer_s:
            self.frames.popleft()
        active = self._activity(f)
        if active:
            self.rest_since = None
            self.active_since = self.active_since if self.active_since is not None else f.t
        else:
            self.active_since = None
            if self.rest_since is None:  # parked hands have been resting since they became still
                self.rest_since = self.still_since if self.parked else f.t

        if self.state == "idle":
            gap = cfg.hold_unsure_s if self.pending_held else cfg.join_gap_s
            if self.pending is not None and (cfg.flush_fragments or self.pending_held) and f.t - self.pending[1] > gap:
                start, end = self.pending
                held, self.pending, self.pending_held = self.pending_held, None, False
                if end - start >= cfg.min_sign_s:
                    self.last_end = {"reason": "held" if held else "fragment", "duration_s": round(end - start, 3)}
                    self.last_bounds = (start - cfg.pad_pre_s, end + cfg.fragment_keep_s + cfg.pad_post_s)
                    seg = self._segment(*self.last_bounds)
                    if seg is not None:
                        return "sign_ended", seg
            if self.active_since is not None and f.t - self.active_since >= cfg.on_s:
                start = self.own_start = self.active_since
                self.joined = None
                if self.pending is not None and start - self.pending[1] <= gap:
                    self.joined = (*self.pending, self.pending_held)
                    start = self.pending[0]  # continue the held-back piece: one sign with a pause
                self.pending, self.pending_held = None, False
                self.state, self.sign_start = "signing", start
                return "sign_started", None
            return None, None

        ended = self.rest_since is not None and f.t - self.rest_since >= cfg.off_s
        too_long = f.t - self.sign_start >= cfg.max_sign_s
        if not (ended or too_long):
            return None, None
        end = self.rest_since if ended else f.t
        if ended and self.parked and cfg.still_keep_s > 0:
            end = min(f.t, end + cfg.still_keep_s)
        start = self.sign_start
        self.state, self.sign_start, self.active_since = "idle", None, None
        reason = ("still" if self.parked else "rest") if ended else "max_length"
        self.last_end = {"reason": reason, "duration_s": round(end - start, 3)}
        if reason == "still" and cfg.join_short_s > 0 and end - start < cfg.join_short_s:
            self.pending = (start, end)
            self.pending_held = self.joined is not None and self.joined[2]  # a held part stays held when continued
            self.last_end = {"reason": "fragment", "duration_s": round(end - start, 3)}
            return "too_short", None
        if end - start < cfg.min_sign_s:
            return "too_short", None
        self.last_bounds = (start - cfg.pad_pre_s, end + cfg.pad_post_s)
        self.last_span = (start, end)
        self.last_own_bounds = (self.own_start - cfg.pad_pre_s, end + cfg.pad_post_s)
        return "sign_ended", self._segment(*self.last_bounds)

    def hold(self) -> None:
        """Hold the sign that just ended (unsure, ended in a still pause): signing again soon continues it."""
        self.pending, self.pending_held = self.last_span, True

    def _segment(self, t0: float, t1: float) -> LandmarkSequence | None:
        fs = [f for f in self.frames if t0 <= f.t <= t1]
        if len(fs) < 4:
            return None
        return LandmarkSequence(
            hands=np.stack([f.hands for f in fs]).astype(np.float32),
            hand_present=np.stack([f.present for f in fs]),
            hand_score=np.stack([f.present for f in fs]).astype(np.float32),
            pose=np.stack([f.pose for f in fs]).astype(np.float32),
            pose_present=np.array([f.pose_present for f in fs]),
            timestamps_ms=np.array([int(f.t * 1000) for f in fs]),
            meta={"width": 0, "height": 0},
        )


class ModelBundle:
    """Exported model directory (model.pt + model_card.json) -> calibrated class probabilities."""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        card = json.loads((Path(model_dir) / "model_card.json").read_text(encoding="utf-8"))
        self.card, self.name, self.classes = card, card["name"], card["classes"]
        self.cfg = FeatureConfig.from_dict(card["feature_config"])
        self.mean = np.asarray(card["feature_mean"], np.float32)
        self.std = np.asarray(card["feature_std"], np.float32)
        self.temperature = float(card["temperature"])
        self.threshold = float(card["uncertain_threshold"])
        self.device = torch.device(device)
        self.model = build_model(ModelConfig.from_dict(card["model_config"]))
        self.model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
        self.cleaning: tuple[float, float, float] | None = None  # (below_fps, fps, max_gap_s), see use_cleaning
        boundaries = Path(model_dir) / "known_boundaries.npz"
        self.known_boundaries_path = boundaries if boundaries.exists() else None
        self._known: dict[float, object] = {}

    def use_cleaning(self, cfg: SegmenterConfig) -> None:
        """Clean every sign before the model (live signs and taught-word takes alike): see preprocessing/resample.py."""
        on = cfg.clean_below_fps > 0 or cfg.fill_gap_s > 0
        self.cleaning = (cfg.clean_below_fps, cfg.clean_fps, cfg.fill_gap_s) if on else None

    def known_boundaries(self, k: float):
        """KnownWordBoundaries at strictness k (cached), or None when the model has no known_boundaries.npz."""
        if self.known_boundaries_path is None:
            return None
        if k not in self._known:
            from mudra_ml.fewshot import KnownWordBoundaries

            self._known[k] = KnownWordBoundaries.load(self.known_boundaries_path, k)
        return self._known[k]

    def _features(self, seqs: list[LandmarkSequence], width: int, height: int) -> torch.Tensor:
        feats = []
        for s in seqs:
            s.meta = {**s.meta, "width": width, "height": height}
            if self.cleaning is not None:
                s = clean(s, *self.cleaning)
            x = np.clip(sequence_features(s, self.cfg), -self.cfg.clip_value, self.cfg.clip_value)
            feats.append((x - self.mean) / self.std)
        return torch.from_numpy(np.stack(feats)).to(self.device)

    @torch.no_grad()
    def probs(self, seqs: list[LandmarkSequence], width: int, height: int) -> np.ndarray:
        return self.probs_and_embeddings(seqs, width, height)[0]

    @torch.no_grad()
    def embed(self, seqs: list[LandmarkSequence], width: int, height: int) -> np.ndarray:
        """Pooled sequence representation [N, D] (input of the classifier head), for few-shot matching."""
        return self.probs_and_embeddings(seqs, width, height)[1]

    @torch.no_grad()
    def probs_and_embeddings(self, seqs: list[LandmarkSequence], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        z = self.model.encode(self._features(seqs, width, height))
        logits = self.model.head(z).float().cpu().numpy() / self.temperature
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True), self.model.embedding_from(z).float().cpu().numpy()

    def decide(self, p: np.ndarray, top_k: int = 3, threshold: float | None = None, margin: float = 0.0) -> dict:
        """Whole-sign decision: background -> no_sign; below threshold -> uncertain; else ok."""
        best = int(p.argmax())
        order = [i for i in np.argsort(-p) if self.classes[i] != BACKGROUND_LABEL][:top_k]
        top = [{"sign": self.classes[i], "confidence": float(p[i])} for i in order]
        if self.classes[best] == BACKGROUND_LABEL:
            status = "no_sign"
        elif p[best] < (self.threshold if threshold is None else threshold) or (
            len(top) > 1 and top[0]["sign"] == self.classes[best] and p[best] - top[1]["confidence"] < margin
        ):
            status = "uncertain"
        else:
            status = "ok"
        bg = self.classes.index(BACKGROUND_LABEL) if BACKGROUND_LABEL in self.classes else None
        return {"status": status, "sign": self.classes[best] if status == "ok" else None,
                "confidence": float(p[best]), "top_k": top, "background": float(p[bg]) if bg is not None else 0.0}


class StreamingRecognizer:
    """Segmenter + whole-sign classification. Feed frames, receive events."""

    def __init__(self, bundle: ModelBundle, cfg: SegmenterConfig, width: int, height: int, custom=None, fallback=None):
        self.bundle, self.width, self.height = bundle, width, height
        self.custom = custom  # optional mudra_ml.fewshot.CustomWordBank
        # optional second model for words this one does not know: fallback(segment, width, height, result) ->
        # result or None, asked only when this model is "uncertain" (never on "no sign", so fidgets stay quiet)
        self.fallback = fallback
        self.segmenter = SignSegmenter(cfg, width, height)
        self.cfg = cfg
        self.threshold = cfg.commit_threshold if cfg.commit_threshold is not None else bundle.threshold
        # the "I don't know this sign" barrier, only when configured and built for this model
        self.known = bundle.known_boundaries(cfg.known_boundary_k) if cfg.known_boundary_k is not None else None
        self.last_word: tuple[str, float] | None = None
        self.last_provisional = -1e9
        self.outbox: list[dict] = []  # a second result from one sign end, sent with the next frame

    def push(self, frame: Frame) -> dict | None:
        event, segment = self.segmenter.push(frame)
        if event is None and self.outbox:
            return self.outbox.pop(0)
        if event is None and self.segmenter.state == "signing" and frame.t - self.last_provisional >= self.cfg.provisional_s:
            self.last_provisional = frame.t
            cur = self.segmenter.current()
            if cur is not None and cur.num_frames >= 8:
                guess = self.bundle.decide(self.bundle.probs([cur], self.width, self.height)[0], threshold=0.0)
                if guess["top_k"]:
                    return {"event": "provisional", "state": "signing", "guess": guess["top_k"][0]}
            return None
        if event != "sign_ended":
            if event == "too_short":
                return {"event": event, "state": self.segmenter.state, "stats": self.segmenter.last_end}
            return {"event": event, "state": self.segmenter.state} if event else None
        if segment is None:
            return {"event": "sign_ended", "state": "idle", "result": {"status": "no_sign", "sign": None, "confidence": 0.0, "top_k": []}}
        seg = self.segmenter
        result = self._best(seg.last_bounds, segment)
        joined = seg.joined if seg.last_end.get("reason") not in ("held", "fragment") else None
        if joined is not None and joined[2]:
            # this sign continued an unsure held part: keep the whole only when it beats both parts
            first_bounds = (joined[0] - self.cfg.pad_pre_s, joined[1] + self.cfg.pad_post_s)
            first_seg, own_seg = seg._segment(*first_bounds), seg._segment(*seg.last_own_bounds)
            if first_seg is not None and own_seg is not None:
                first, own = self._best(first_bounds, first_seg), self._best(seg.last_own_bounds, own_seg)
                best_part = max(first["top_k"][0]["confidence"] if first.get("top_k") else 0.0,
                                own["top_k"][0]["confidence"] if own.get("top_k") else 0.0)
                whole = result["top_k"][0]["confidence"] if result.get("top_k") else 0.0
                if not (result["status"] == "ok" and whole >= best_part):
                    first_event = self._event(first, first_seg, frame,
                                              {"reason": "held", "duration_s": round(joined[1] - joined[0], 3)})
                    self.outbox.append(self._event(own, own_seg, frame, {**seg.last_end, "reason": "held"}))
                    return first_event
        if (self.cfg.hold_unsure_s > 0 and seg.last_end.get("reason") == "still" and result["status"] != "ok"
                and not result.get("custom")):
            seg.hold()
            return {"event": "sign_held", "state": "idle"}
        return self._event(result, segment, frame, seg.last_end)

    RESCUE_SHIFTS = ((-0.5, 0.0), (-1.0, 0.0), (0.0, 0.3), (-0.5, 0.3), (0.2, 0.0))  # (start, end) shifts in s

    def _best(self, bounds: tuple[float, float], segment: LandmarkSequence) -> dict:
        """The sign's result; when unsure (and rescue_cuts is on), the most confident passing result of wider cuts."""
        result = self._classify(bounds, segment)
        if not self.cfg.rescue_cuts or result["status"] == "ok":
            return result
        best = None
        for a, b in self.RESCUE_SHIFTS:
            cut = self.segmenter._segment(bounds[0] + a, bounds[1] + b)
            if cut is None or cut.num_frames == segment.num_frames:
                continue
            r = self._classify((bounds[0] + a, bounds[1] + b), cut)
            need = self.cfg.rescue_threshold if self.cfg.rescue_threshold is not None else 0.0
            if r["status"] == "ok" and r["confidence"] >= need and (best is None or r["confidence"] > best["confidence"]):
                best = r
        return {**best, "rescued": True} if best is not None else result

    def _classify(self, bounds: tuple[float, float], segment: LandmarkSequence) -> dict:
        cuts = self.segmenter.variants(*bounds, self.cfg.tta) or [segment]
        probs, emb = self.bundle.probs_and_embeddings(cuts, self.width, self.height)
        probs = probs.mean(axis=0)  # smoothing over cuts
        result = self.bundle.decide(probs, threshold=self.threshold, margin=self.cfg.margin)
        if result["status"] == "ok" and self.known is not None and not self.known.inside(result["sign"], emb.mean(axis=0)):
            result = {**result, "status": "uncertain", "sign": None, "reason": "outside_word_boundary"}
        if self.custom is not None and len(self.custom):
            result = self.custom.decide(emb.mean(axis=0), result) or result
        if result["status"] == "uncertain" and self.fallback is not None:
            result = self.fallback(segment, self.width, self.height, result) or result
        return result

    def _event(self, result: dict, segment: LandmarkSequence, frame: Frame, end_stats: dict) -> dict:
        if result["status"] == "ok":
            if self.last_word and self.last_word[0] == result["sign"] and frame.t - self.last_word[1] < self.cfg.cooldown_s:
                result = {**result, "status": "duplicate", "sign": None}
            else:
                self.last_word = (result["sign"], frame.t)
        span = (float(segment.timestamps_ms[-1]) - float(segment.timestamps_ms[0])) / 1000
        stats = {**end_stats, "frames": segment.num_frames,
                 "fps": round((segment.num_frames - 1) / span, 1) if span > 0 else 0.0,
                 "hand_rate": round(float(segment.hand_present.any(axis=1).mean()), 3),
                 "both_hands_rate": round(float(segment.hand_present.all(axis=1).mean()), 3),
                 "pose_rate": round(float(segment.pose_present.mean()), 3),
                 "body_reference": self.segmenter.body is not None}
        return {"event": "sign_ended", "state": "idle", "result": result, "stats": stats,
                "t0": float(segment.timestamps_ms[0]) / 1000, "t1": float(segment.timestamps_ms[-1]) / 1000}
