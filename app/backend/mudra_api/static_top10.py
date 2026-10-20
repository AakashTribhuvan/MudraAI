"""Live adapter for the locally trained top-ten landmark SVM experiment.

Ported from AakashTribhuvan/MitIndia, branch work/static-landmark-tracking, commit 8105c5e (2026-10-19):
feature and model logic unchanged; joblib.load replaced by mudra_api.safe_joblib.load (allow-list unpickler).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from mudra_ml.streaming import Frame

from mudra_api import safe_joblib


FEATURE_SCHEMA = "mudra_static_top10_body_face_hands_v1"
MIN_MARGIN = 0.15  # conservative reject band for SVC's top-vs-runner-up decision scores


def feature_from_frame(frame: Frame, palm_directions: np.ndarray, width: int, height: int) -> np.ndarray | None:
    if not frame.pose_present or width <= 0 or height <= 0 or not frame.present.any():
        return None
    pose = frame.pose.astype(np.float32, copy=True)
    pose[:, :3] *= np.asarray((width, height, width), dtype=np.float32)
    if min(pose[11, 3], pose[12, 3]) < 0.35:
        return None
    shoulder_mid = (pose[11, :3] + pose[12, :3]) / 2
    shoulder_width = float(np.linalg.norm(pose[12, :2] - pose[11, :2]))
    if shoulder_width < 1e-5:
        return None
    hip_mid = (pose[23, :3] + pose[24, :3]) / 2 if min(pose[23, 3], pose[24, 3]) >= 0.2 else shoulder_mid + (0, 2 * shoulder_width, 0)

    def unit(vector: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(vector))
        return vector / length if np.isfinite(length) and length > 1e-7 else np.zeros_like(vector)

    lateral = unit(pose[12, :3] - pose[11, :3])
    upward = unit(shoulder_mid - hip_mid)
    depth = unit(np.cross(lateral, upward))
    basis = np.stack((lateral, upward, depth))
    body_points = (0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24)
    values: list[float] = []
    for index in body_points:
        values.extend((basis @ (pose[index, :3] - shoulder_mid) / shoulder_width).tolist())
        values.append(float(pose[index, 3]))

    anchors = {"eyes": (2, 5), "ears": (7, 8), "mouth": (9, 10)}
    descriptions: list[tuple[float, list[float]]] = []
    for slot in np.where(frame.present)[0]:
        hand = frame.hands[slot].astype(np.float32, copy=True)
        hand *= np.asarray((width, height, width), dtype=np.float32)
        wrist = hand[0].copy()
        palm_width = float(np.linalg.norm(hand[5, :2] - hand[17, :2]))
        if palm_width < 1e-5:
            continue
        wrist3 = wrist - shoulder_mid
        lateral_position = float(np.dot(wrist3, lateral))
        hand_local = (hand - wrist) / palm_width
        block = (basis @ wrist3 / shoulder_width).tolist()
        block.extend(hand_local.reshape(-1).tolist())
        block.extend(np.asarray(palm_directions[slot], dtype=np.float32).tolist())
        extensions = []
        for tip, mcp in zip((4, 8, 12, 16, 20), (1, 5, 9, 13, 17)):
            tip_len = float(np.linalg.norm(hand[tip] - wrist)) / palm_width
            base_len = float(np.linalg.norm(hand[mcp] - wrist)) / palm_width
            extensions.append(tip_len / max(base_len, 1e-4))
        block.extend(extensions)
        block.append(float(sum(value > 1.35 for value in extensions)))
        for indices in anchors.values():
            visible = [pose[index, :2] for index in indices if pose[index, 3] >= 0.2]
            if not visible:
                block.extend((2.0, 0.0, 0.0))
                continue
            distances = np.linalg.norm(hand[:, None, :2] - np.asarray(visible)[None, :, :], axis=2)
            minimum = float(distances.min()) / shoulder_width
            block.extend((minimum, float(minimum < 0.15), float(minimum < 0.25)))
        descriptions.append((lateral_position, block))
    descriptions.sort(key=lambda item: item[0])
    for _, block in descriptions[:2]:
        values.extend(block)
    while len(descriptions) < 2:
        values.extend([0.0] * 84)
        descriptions.append((0.0, []))
    values.append(float(frame.present.sum()))
    feature = np.asarray(values, dtype=np.float32)
    return feature if feature.shape == (229,) and np.isfinite(feature).all() else None


class StaticTop10Model:
    def __init__(self, path: Path):
        payload = safe_joblib.load(path)
        if payload.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("Unsupported top-ten model feature schema")
        self.estimator = payload["estimator"]
        self.classes = self.estimator.classes_.astype(str).tolist()

    @property
    def live_labels(self) -> set[str]:
        return set(self.classes)

    def predict(self, feature: np.ndarray) -> dict:
        scores = np.asarray(self.estimator.decision_function(feature[None, :])).reshape(-1)
        order = np.argsort(scores)[::-1]
        best, second = int(order[0]), float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else math.inf
        confidence = float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, second)))))
        accepted = second >= MIN_MARGIN
        return {"prediction": self.classes[best] if accepted else None,
                "candidate": self.classes[best], "margin": second, "confidence": confidence}


class StaticTop10StreamingRecognizer:
    """Classify a held static sign; dynamic motion remains for the future sequence model."""

    def __init__(self, model: StaticTop10Model, width: int, height: int):
        self.model, self.width, self.height = model, width, height
        self.votes: list[tuple[float, dict]] = []
        self.candidate_since: float | None = None
        self.candidate: str | None = None
        self.emitted = False

    def push(self, frame: Frame, palm_directions: np.ndarray) -> dict | None:
        feature = feature_from_frame(frame, palm_directions, self.width, self.height)
        if feature is None:
            self.votes.clear()
            self.candidate = None
            self.candidate_since = None
            self.emitted = False
            return None
        result = self.model.predict(feature)
        label = result["prediction"]
        if label is None:
            self.candidate = None
            self.candidate_since = None
            self.emitted = False
            self.votes.clear()
            return None
        if label != self.candidate:
            self.candidate, self.candidate_since = label, frame.t
            self.emitted = False
            self.votes.clear()
        self.votes.append((frame.t, result))
        self.votes = [(t, value) for t, value in self.votes if frame.t - t <= 0.8]
        if self.candidate_since is None or frame.t - self.candidate_since < 0.45 or self.emitted:
            return None
        if sum(value["candidate"] == label for _, value in self.votes) < 3:
            return None
        self.emitted = True
        confidence = float(np.median([value["confidence"] for _, value in self.votes]))
        return {"event": "sign_ended", "state": "signing", "result": {
            "status": "ok", "sign": label, "label": label.replace("_", " ").title(),
            "confidence": confidence,
            "top_k": [{"sign": label, "label": label.replace("_", " ").title(), "confidence": confidence}],
            "frames_received": len(self.votes), "model": "top10_v2", "score_type": "svm_margin",
            "margin": float(np.median([value["margin"] for _, value in self.votes])),
        }, "stats": {"frames": len(self.votes), "hand_rate": 1.0, "body_reference": frame.pose_present},
            "t0": self.candidate_since, "t1": frame.t}


def frame_palm_directions(msg: dict, frame: Frame) -> np.ndarray:
    """Align the client's canonical palm vectors with parse_frame's anatomical hand slots."""
    output = np.zeros((2, 3), dtype=np.float32)
    used: set[int] = set()
    for hand in msg.get("hands", [])[:2]:
        direction = hand.get("palm_direction")
        points = hand.get("landmarks")
        if direction is None or points is None:
            continue
        wrist = np.asarray(points[0][:2], dtype=np.float32)
        slots = [slot for slot in range(2) if frame.present[slot] and slot not in used]
        if not slots:
            continue
        slot = min(slots, key=lambda index: float(np.linalg.norm(frame.hands[index, 0, :2] - wrist)))
        output[slot] = np.asarray(direction, dtype=np.float32)
        used.add(slot)
    return output
