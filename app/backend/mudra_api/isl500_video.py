"""Live adapter for the teammate's whole-clip video-word SVMs (ISL500 14-word pilot, SignLine 26-word).

Ported from AakashTribhuvan/MitIndia, branch work/static-landmark-tracking, commit 8105c5e (2026-10-19):
feature and model logic unchanged; joblib.load replaced by mudra_api.safe_joblib.load (allow-list unpickler).
Not ported: the held-out test-video viewers (they serve the teammate's local videos).
The model files derive from ISL500/ISL-DATA: research and academic use only.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
from mudra_ml.preprocessing.sequence import LandmarkSequence
from mudra_ml.streaming import Frame, SegmenterConfig, SignSegmenter

from mudra_api import safe_joblib
from mudra_api.static_top10 import FEATURE_SCHEMA, feature_from_frame


def _palm_directions(hands: np.ndarray, present: np.ndarray) -> np.ndarray:
    output = np.zeros((2, 3), dtype=np.float32)
    for slot in range(2):
        if not present[slot]:
            continue
        hand = hands[slot]
        normal = np.cross(hand[5] - hand[0], hand[17] - hand[0])
        length = float(np.linalg.norm(normal))
        if np.isfinite(length) and length > 1e-8:
            output[slot] = normal / length * (-1.0 if slot == 1 else 1.0)
    return output


def temporal_features(sequence: LandmarkSequence, width: int, height: int, descriptor: str) -> tuple[np.ndarray, float]:
    """Match training: feature every available frame, crop to hand activity, then sample the sequence."""
    if sequence.num_frames == 0:
        raise ValueError("The sign segment is empty.")
    features: list[np.ndarray | None] = []
    for i in range(sequence.num_frames):
        frame = Frame(
            t=float(sequence.timestamps_ms[i]) / 1000,
            hands=sequence.hands[i],
            present=sequence.hand_present[i],
            pose=sequence.pose[i],
            pose_present=bool(sequence.pose_present[i]),
        )
        features.append(feature_from_frame(frame, _palm_directions(frame.hands, frame.present), width, height))
    valid = np.asarray([feature is not None for feature in features], dtype=bool)
    if not valid.any():
        raise ValueError("No usable hand, torso and face-anchor frames were tracked.")

    active = np.flatnonzero(sequence.hand_present.any(axis=1))
    if not len(active):
        active = np.flatnonzero(valid)
    start, end = int(active[0]), int(active[-1])
    ids = np.rint(np.linspace(start, end, 12)).astype(int)
    valid_ids = np.flatnonzero(valid)
    sampled = []
    for index in ids:
        if valid[index]:
            sampled.append(features[index])
        else:
            nearest = valid_ids[np.argmin(np.abs(valid_ids - index))]
            sampled.append(features[nearest])
    matrix = np.stack(sampled).astype(np.float32)
    if descriptor.startswith("flat"):
        expected = int(descriptor.removeprefix("flat"))
        if expected != len(matrix):
            source_ids = np.rint(np.linspace(0, len(matrix) - 1, expected)).astype(int)
            matrix = matrix[source_ids]
        return matrix.reshape(-1), float(valid[start : end + 1].mean())

    bins = int(descriptor.removeprefix("bins"))
    edges = np.linspace(0, len(matrix), bins + 1).round().astype(int)
    parts = []
    for begin, finish in pairwise(edges):
        block = matrix[begin : max(begin + 1, finish)]
        parts.extend((block.mean(axis=0), block.std(axis=0)))
    return np.concatenate(parts).astype(np.float32), float(valid[start : end + 1].mean())


class ISL500WordModel:
    def __init__(self, path: Path):
        payload = safe_joblib.load(path)
        if payload.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("Unsupported ISL500 model feature schema.")
        self.estimator = payload["estimator"]
        self.classes = [str(label) for label in payload["classes"]]
        self.descriptor = payload["descriptor"]
        self.name = "ISL500 video pilot"

    @property
    def live_labels(self) -> set[str]:
        return set(self.classes)

    def predict(self, features: np.ndarray) -> dict:
        if hasattr(self.estimator, "predict_proba"):
            probabilities = self.estimator.predict_proba(features[None, :])[0]
        else:
            scores = np.asarray(self.estimator.decision_function(features[None, :])).reshape(-1)
            scores = scores - scores.max()
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum()
        order = np.argsort(probabilities)[::-1][:3]
        top = [{"sign": self.classes[i], "confidence": float(probabilities[i])} for i in order]
        confidence = top[0]["confidence"]
        return {
            "status": "ok" if confidence >= 0.5 else "uncertain",
            "sign": top[0]["sign"] if confidence >= 0.5 else None,
            "confidence": confidence,
            "top_k": top,
            "score_type": "svm_probability",
        }


class ISL500VideoStreamingRecognizer:
    """Segment dynamic sign motion and classify the full trajectory once it ends."""

    def __init__(self, model: ISL500WordModel, cfg: SegmenterConfig, width: int, height: int,
                 model_id: str = "isl500_video"):
        self.model, self.width, self.height = model, width, height
        self.model_id = model_id
        self.segmenter = SignSegmenter(cfg, width, height)
        self.last_word: tuple[str, float] | None = None

    def push(self, frame: Frame) -> dict | None:
        event, segment = self.segmenter.push(frame)
        if event == "sign_started":
            return {"event": event, "state": "signing"}
        if event == "too_short":
            return {"event": event, "state": self.segmenter.state, "stats": self.segmenter.last_end}
        if event != "sign_ended":
            return None
        if segment is None:
            result = {"status": "no_sign", "sign": None, "confidence": 0.0, "top_k": []}
            return {"event": "sign_ended", "state": "idle", "result": result, "stats": self.segmenter.last_end}

        try:
            vector, coverage = temporal_features(segment, self.width, self.height, self.model.descriptor)
            result = self.model.predict(vector)
        except ValueError:
            result = {"status": "no_hands", "sign": None, "confidence": 0.0, "top_k": []}
            coverage = 0.0
        sign = result.get("sign")
        if sign and self.last_word and self.last_word[0] == sign and frame.t - self.last_word[1] < self.segmenter.cfg.cooldown_s:
            result = {**result, "status": "duplicate", "sign": None}
        elif sign:
            self.last_word = (sign, frame.t)
        result["model"] = self.model_id
        result["frames_received"] = segment.num_frames
        span = (float(segment.timestamps_ms[-1]) - float(segment.timestamps_ms[0])) / 1000
        stats = {
            **self.segmenter.last_end,
            "frames": segment.num_frames,
            "fps": round((segment.num_frames - 1) / span, 1) if span > 0 else 0.0,
            "hand_rate": round(coverage, 3),
            "pose_rate": round(float(segment.pose_present.mean()), 3),
            "body_reference": self.segmenter.body is not None,
        }
        return {
            "event": "sign_ended", "state": "idle", "result": result, "stats": stats,
            "t0": float(segment.timestamps_ms[0]) / 1000,
            "t1": float(segment.timestamps_ms[-1]) / 1000,
        }
