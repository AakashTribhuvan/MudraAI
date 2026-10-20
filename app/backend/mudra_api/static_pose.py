"""Experimental still-pose model adapted from MudraPhoneWebcam's v10 medoids.

Ported unchanged (logic) from AakashTribhuvan/MitIndia, branch work/static-landmark-tracking, commit
e040672 (2026-10-18), file app/backend/mudra_api/static_pose.py. Measured on our held-out test data in
training/scripts/compare_static_model.py: see the model card mudra_intro_static_pose_medoids_v1.json.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
from mudra_ml.streaming import Frame, LandmarkSequence

FEATURE_SCHEMA = "mudra_static_pose_v10"
HAND_FEATURE_DIM = 72
FEATURE_DIM = 32 + 2 * HAND_FEATURE_DIM
POSE_INDICES = (11, 12, 13, 14, 15, 16, 23, 24)
FACE_ANCHOR_FLAG_INDICES = (32 + HAND_FEATURE_DIM - 1, 32 + 2 * HAND_FEATURE_DIM - 1)
MIN_ANCHOR_VISIBILITY = 0.45
MIN_HAND_DIAGONAL_PX = 30.0
MIN_HAND_IMAGE_DIAGONAL_RATIO = 0.015
MIN_FRAME_AGREEMENT = 0.5


def _pose_face_anchor(
    pose_xyz: np.ndarray,
    pose_visibility: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    if pose_visibility[0] < 0.35:
        return None, 0.0
    for left_index, right_index in ((7, 8), (2, 5)):
        if min(pose_visibility[left_index], pose_visibility[right_index]) < 0.35:
            continue
        face_width = float(np.linalg.norm((pose_xyz[left_index] - pose_xyz[right_index])[:2]))
        if math.isfinite(face_width) and face_width > 1e-6:
            return pose_xyz[0], face_width
    return None, 0.0


def _torso_anchor(
    pose_xyz: np.ndarray,
    pose_visibility: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    left_shoulder, right_shoulder = pose_xyz[11], pose_xyz[12]
    shoulder_visibility = float(min(pose_visibility[11], pose_visibility[12]))
    shoulder_width = float(np.linalg.norm((left_shoulder - right_shoulder)[:2]))
    if not math.isfinite(shoulder_width) or shoulder_width <= 1e-6:
        raise ValueError("Shoulder width is missing or degenerate.")

    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    if min(pose_visibility[23], pose_visibility[24]) >= 0.45:
        hip_midpoint = (pose_xyz[23] + pose_xyz[24]) / 2
        anchor_visibility = min(shoulder_visibility, float(pose_visibility[23]), float(pose_visibility[24]))
    else:
        hip_midpoint = shoulder_midpoint + np.asarray((0.0, shoulder_width * 2.0, 0.0), dtype=np.float32)
        anchor_visibility = shoulder_visibility
    return (shoulder_midpoint + hip_midpoint) / 2, shoulder_width, anchor_visibility


def _feature_from_frame(
    frame: Frame,
    width: int,
    height: int,
) -> tuple[np.ndarray, int] | None:
    if not frame.pose_present or width <= 0 or height <= 0:
        return None
    pose_xyz = frame.pose[:, :3].astype(np.float32, copy=True)
    pose_xyz[:, 0] *= width
    pose_xyz[:, 1] *= height
    pose_xyz[:, 2] *= width
    visibility = frame.pose[:, 3].astype(np.float32, copy=False)

    present_hands = [
        frame.hands[index].astype(np.float32, copy=True)
        for index in range(min(2, len(frame.present)))
        if frame.present[index]
    ]
    if not present_hands:
        return None
    hand_diagonals = []
    scaled_hands = []
    for hand in present_hands:
        hand[:, 0] *= width
        hand[:, 1] *= height
        hand[:, 2] *= width
        hand_diagonals.append(float(np.linalg.norm(np.ptp(hand[:, :2], axis=0))))
        scaled_hands.append(hand)
    minimum_hand_diagonal = max(MIN_HAND_DIAGONAL_PX, math.hypot(width, height) * MIN_HAND_IMAGE_DIAGONAL_RATIO)
    if max(hand_diagonals, default=0.0) < minimum_hand_diagonal:
        return None

    try:
        torso_center, shoulder_width, anchor_visibility = _torso_anchor(pose_xyz, visibility)
    except ValueError:
        return None
    if anchor_visibility < MIN_ANCHOR_VISIBILITY:
        return None

    face_center, face_width = _pose_face_anchor(pose_xyz, visibility)
    body_features: list[float] = []
    body_confidence = []
    hips_visible = min(visibility[23], visibility[24]) >= 0.45
    left_shoulder, right_shoulder = pose_xyz[11], pose_xyz[12]
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    if hips_visible:
        left_hip, right_hip = pose_xyz[23], pose_xyz[24]
    else:
        hip_midpoint = shoulder_midpoint + np.asarray((0.0, shoulder_width * 2.0, 0.0), dtype=np.float32)
        shoulder_axis = (right_shoulder - left_shoulder) / 2
        left_hip, right_hip = hip_midpoint - shoulder_axis, hip_midpoint + shoulder_axis

    for index in POSE_INDICES:
        point_visibility = float(visibility[index])
        if index in (23, 24) and not hips_visible:
            point_visibility = 0.0
        body_confidence.append(point_visibility)
        if index in (11, 12, 23, 24):
            if index in (23, 24) and not hips_visible:
                hip = left_hip if index == 23 else right_hip
                body_features.extend(((hip - torso_center) / shoulder_width).tolist())
            elif point_visibility > 0:
                body_features.extend(((pose_xyz[index] - torso_center) / shoulder_width).tolist())
            else:
                body_features.extend((0.0, 0.0, 0.0))

    for start, end in ((11, 13), (13, 15), (12, 14), (14, 16)):
        vector = pose_xyz[end] - pose_xyz[start]
        vector_length = float(np.linalg.norm(vector))
        if min(visibility[start], visibility[end]) < 0.1 or vector_length <= 1e-6:
            body_features.extend((0.0, 0.0, 0.0))
        else:
            body_features.extend((vector / vector_length).tolist())
    body_features.extend(body_confidence)

    body_right_axis = right_shoulder - left_shoulder
    body_right_axis[2] = 0.0
    body_right_axis /= max(float(np.linalg.norm(body_right_axis[:2])), 1e-6)
    sorted_hands = sorted(scaled_hands, key=lambda hand: float(np.dot(hand[0] - torso_center, body_right_axis)))
    hand_blocks = []
    for hand in sorted_hands[:2]:
        wrist = hand[0]
        palm_width = float(np.linalg.norm((hand[5] - hand[17])[:2]))
        if not math.isfinite(palm_width) or palm_width <= 1e-6:
            hand_blocks.append([0.0] * HAND_FEATURE_DIM)
            continue
        block = ((wrist - torso_center) / shoulder_width).tolist()
        block.extend(((hand - wrist) / palm_width).reshape(-1).tolist())
        block.append(1.0)
        if face_center is None:
            block.extend((0.0,) * 5)
        else:
            block.extend(((wrist - face_center) / face_width).tolist())
            index_direction = hand[8] - hand[5]
            face_direction = face_center - wrist
            index_length = float(np.linalg.norm(index_direction))
            face_distance = float(np.linalg.norm(face_direction))
            alignment = (
                float(np.dot(index_direction / index_length, face_direction / face_distance))
                if index_length > 1e-6 and face_distance > 1e-6
                else 0.0
            )
            block.extend((alignment, 1.0))
        hand_blocks.append(block)
    while len(hand_blocks) < 2:
        hand_blocks.append([0.0] * HAND_FEATURE_DIM)

    feature = np.asarray(body_features + [value for block in hand_blocks for value in block], dtype=np.float32)
    if feature.shape != (FEATURE_DIM,) or not np.isfinite(feature).all():
        return None
    return feature, len(present_hands)


class StaticPoseModel:
    """Nearest-medoid matcher; only labels with finite source-group thresholds can be returned."""

    def __init__(self, model_path: Path):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Intro static model is missing: {model_path}")
        with np.load(model_path, allow_pickle=False) as archive:
            self.classes = archive["classes"].astype(str).tolist()
            self.prototypes = archive["prototypes"].astype(np.float32)
            self.hand_counts = archive["required_hand_counts"].astype(np.int32)
            self.face_required = archive["required_face_anchors"].astype(bool)
            self.thresholds = archive["rejection_thresholds"].astype(np.float32)
            self.feature_schema = archive["feature_schema"].astype(str).tolist()
        if self.feature_schema != [FEATURE_SCHEMA]:
            raise ValueError(f"Unsupported static model feature schema: {self.feature_schema}")
        expected = (len(self.classes), FEATURE_DIM)
        if (
            self.prototypes.shape != expected
            or self.hand_counts.shape != (len(self.classes),)
            or self.face_required.shape != (len(self.classes),)
            or self.thresholds.shape != (len(self.classes),)
        ):
            raise ValueError("Intro static model arrays have inconsistent shapes.")

    @property
    def live_labels(self) -> set[str]:
        return {
            label for label, threshold in zip(self.classes, self.thresholds)
            if math.isfinite(float(threshold))
        }

    def predict_feature(self, feature: np.ndarray, hand_count: int) -> dict:
        distances = np.linalg.norm(self.prototypes - feature, axis=1)
        eligible = self.hand_counts == hand_count
        if not eligible.any():
            return {"prediction": None}
        distances[~eligible] = np.inf
        best = int(np.argmin(distances))
        threshold = float(self.thresholds[best])
        if not math.isfinite(threshold) or distances[best] > threshold:
            return {"prediction": None}
        if self.face_required[best] and not any(feature[index] >= 0.5 for index in FACE_ANCHOR_FLAG_INDICES):
            return {"prediction": None}
        return {
            "prediction": self.classes[best],
            "distance": float(distances[best]),
            "threshold": threshold,
            "hand_count": hand_count,
        }

    def predict_sequence(self, sequence: LandmarkSequence, width: int, height: int) -> dict:
        frame_predictions = []
        valid_frames = 0
        for index in range(sequence.num_frames):
            frame = Frame(
                t=float(sequence.timestamps_ms[index]) / 1000,
                hands=sequence.hands[index],
                present=sequence.hand_present[index],
                pose=sequence.pose[index],
                pose_present=bool(sequence.pose_present[index]),
            )
            extracted = _feature_from_frame(frame, width, height)
            if extracted is None:
                continue
            valid_frames += 1
            feature, hand_count = extracted
            result = self.predict_feature(feature, hand_count)
            if result["prediction"]:
                frame_predictions.append(result)

        counts = Counter(item["prediction"] for item in frame_predictions)
        total = valid_frames
        ordered = sorted(counts, key=lambda label: (-counts[label], label))
        top = [
            {
                "sign": label,
                "confidence": counts[label] / max(total, 1),
                "label": label.replace("_", " ").title(),
            }
            for label in ordered[:3]
        ]
        if not top:
            return {
                "status": "no_sign",
                "sign": None,
                "label": None,
                "confidence": 0.0,
                "top_k": [],
                "frames_received": sequence.num_frames,
                "model": "intro_static",
                "score_type": "frame_agreement",
            }

        best = top[0]
        agreement = float(best["confidence"])
        accepted = counts[best["sign"]] >= 3 and agreement >= MIN_FRAME_AGREEMENT
        distances = [item["distance"] for item in frame_predictions if item["prediction"] == best["sign"]]
        thresholds = [item["threshold"] for item in frame_predictions if item["prediction"] == best["sign"]]
        return {
            "status": "ok" if accepted else "uncertain",
            "sign": best["sign"] if accepted else None,
            "label": best["label"] if accepted else None,
            "confidence": agreement,
            "top_k": top,
            "frames_received": sequence.num_frames,
            "model": "intro_static",
            "score_type": "frame_agreement",
            "distance": float(np.median(distances)),
            "threshold": float(np.median(thresholds)),
        }


class StaticPoseStreamingRecognizer:
    """Port the phone prototype's held-pose recognition into the browser landmark stream."""

    def __init__(
        self,
        model: StaticPoseModel,
        width: int,
        height: int,
    ):
        self.model = model
        self.width = width
        self.height = height
        self.aspect = width / height if width > 0 and height > 0 else 1.0
        self.start_hold_s = 0.15
        self.stop_hold_s = 0.7
        self.prediction_hold_s = 0.45
        self.movement_threshold = 1.0
        self.raised_below_shoulder = 1.0
        self.state = "idle"
        self.mode = "idle"
        self.candidate_since: float | None = None
        self.rest_since: float | None = None
        self.mode_candidate: str | None = None
        self.mode_candidate_since: float | None = None
        self.previous_wrists: dict[int, np.ndarray] = {}
        self.previous_t: float | None = None
        self.stable_candidate: str | None = None
        self.stable_since: float | None = None
        self.last_provisional = -1e9
        self.emitted = False

    def _activity(self, frame: Frame) -> tuple[bool, bool, bool]:
        visible = bool(np.any(frame.present))
        shoulder_center_y = None
        shoulder_width = self.aspect * 0.25
        if frame.pose_present and min(float(frame.pose[11, 3]), float(frame.pose[12, 3])) >= 0.5:
            left, right = frame.pose[11, :2].copy(), frame.pose[12, :2].copy()
            left[0] *= self.aspect
            right[0] *= self.aspect
            shoulder_center_y = float((left[1] + right[1]) / 2)
            shoulder_width = max(float(np.linalg.norm(left - right)), 1e-4)

        current: dict[int, np.ndarray] = {}
        raised = moving = False
        dt = max(1e-3, frame.t - self.previous_t) if self.previous_t is not None else 1e-3
        for index in range(min(2, len(frame.present))):
            if not frame.present[index]:
                continue
            wrist = frame.hands[index, 0, :2].astype(np.float32, copy=True)
            wrist[0] *= self.aspect
            current[index] = wrist
            if shoulder_center_y is not None:
                raised |= float(wrist[1]) <= shoulder_center_y + self.raised_below_shoulder * shoulder_width
            else:
                raised |= float(wrist[1]) < 0.7
            previous = self.previous_wrists.get(index)
            if previous is not None:
                speed = float(np.linalg.norm(wrist - previous)) / shoulder_width / dt
                moving |= speed >= self.movement_threshold
        self.previous_wrists = current
        self.previous_t = frame.t
        return visible, raised, moving

    def _reset_segment(self) -> None:
        self.state = "idle"
        self.mode = "idle"
        self.candidate_since = None
        self.rest_since = None
        self.mode_candidate = None
        self.mode_candidate_since = None
        self.stable_candidate = None
        self.stable_since = None
        self.last_provisional = -1e9
        self.emitted = False

    def _update_state(self, frame: Frame, visible: bool, raised: bool, moving: bool) -> str | None:
        active = visible and (raised or moving)
        event = None
        if self.state == "idle":
            if active:
                self.state = "starting"
                self.candidate_since = frame.t
            return None
        if self.state == "starting":
            if not active:
                self._reset_segment()
                return None
            if self.candidate_since is None:
                self.candidate_since = frame.t
            if frame.t - self.candidate_since >= self.start_hold_s:
                self.state = "signing"
                self.mode = "dynamic" if moving else "static"
                self.rest_since = None
                event = "sign_started"
            else:
                return None
        elif active:
            if self.state == "ending":
                self.state = "signing"
                self.rest_since = None
        elif self.state == "signing":
            self.state = "ending"
            self.rest_since = frame.t
        elif (self.state == "ending" and self.rest_since is not None
              and frame.t - self.rest_since >= self.stop_hold_s):
            self._reset_segment()
            return "segment_reset"

        if self.state in ("signing", "ending"):
            mode_candidate = "dynamic" if moving else "static" if raised else self.mode
            if mode_candidate == self.mode:
                self.mode_candidate = None
                self.mode_candidate_since = None
            elif self.mode_candidate != mode_candidate:
                self.mode_candidate = mode_candidate
                self.mode_candidate_since = frame.t
            elif self.mode_candidate_since is not None and frame.t - self.mode_candidate_since >= 0.45:
                self.mode = mode_candidate
                self.mode_candidate = None
                self.mode_candidate_since = None
        return event

    def push(self, frame: Frame) -> dict | None:
        visible, raised, moving = self._activity(frame)
        event = self._update_state(frame, visible, raised, moving)
        if event == "sign_started":
            return {"event": event, "state": "signing"}
        if self.state not in ("signing", "ending") or self.mode != "static":
            self.stable_candidate = None
            self.stable_since = None
            return None

        extracted = _feature_from_frame(frame, self.width, self.height)
        match = self.model.predict_feature(*extracted) if extracted is not None else {"prediction": None}
        candidate = match.get("prediction")
        if candidate is None:
            self.stable_candidate = None
            self.stable_since = None
            return None
        if candidate != self.stable_candidate:
            self.stable_candidate = candidate
            self.stable_since = frame.t
        if self.stable_since is None or frame.t - self.stable_since < self.prediction_hold_s:
            if frame.t - self.last_provisional >= 0.5:
                self.last_provisional = frame.t
                return {"event": "provisional", "state": "signing", "guess": {
                    "sign": candidate, "label": candidate.replace("_", " ").title(), "confidence": 0.0,
                }}
            return None
        if self.emitted:
            return None

        result = {
            "status": "ok", "sign": candidate, "label": candidate.replace("_", " ").title(),
            "confidence": max(0.0, 1.0 - match["distance"] / max(match["threshold"], 1e-6)),
            "top_k": [{"sign": candidate, "label": candidate.replace("_", " ").title(),
                       "confidence": max(0.0, 1.0 - match["distance"] / max(match["threshold"], 1e-6))}],
            "frames_received": 1, "model": "intro_static", "score_type": "static_hold",
            "distance": match["distance"], "threshold": match["threshold"],
        }
        self.emitted = True
        return {
            "event": "sign_ended", "state": "signing", "result": result,
            "stats": {"frames": 1, "hand_rate": 1.0, "body_reference": frame.pose_present},
            "t0": self.stable_since, "t1": frame.t,
        }
