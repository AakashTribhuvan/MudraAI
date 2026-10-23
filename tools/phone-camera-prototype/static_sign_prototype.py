"""Prepare and run a small, static-pose sign matching experiment.

This is an exploratory still-image prototype, not a dynamic ISL sign recognizer.
Source groups inferred from filenames are conservative guesses until reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "training_data" / "Frames_Word_Level"
MANIFEST_DIR = ROOT / "training_data" / "manifests"
DERIVED_DIR = ROOT / "training_data" / "derived"
REVIEW_DIR = ROOT / "training_data" / "review"
MODEL_DIR = ROOT / "training_data" / "models"
MANIFEST_PATH = MANIFEST_DIR / "selected_static_samples.csv"
FEATURES_PATH = DERIVED_DIR / "selected_static_features.npz"
MODEL_PATH = MODEL_DIR / "static_pose_medoids.npz"
REPORT_PATH = MODEL_DIR / "static_pose_evaluation.json"
CONFUSION_PATH = MODEL_DIR / "static_pose_confusion.csv"
LABELS = (
    "I_ME_MINE_MY",
    "YOU",
    "FOOD",
    "HELP",
    "WHAT",
    "HOW",
    "COLLEGE_SCHOOL",
    "NICE",
    "THANK",
    "DO",
    "WATER",
    "NOT",
    "HURT",
    "CRYING",
    "THAT",
    "REALLY",
    "OLD_AGE",
    "ANGRY",
    "SOME ONE",
    "TELL",
)
INTRO_LABELS = tuple(
    dict.fromkeys(
        (
            *LABELS,
            "HELLO_HI",
            "GOOD",
            "FROM",
            "TODAY",
            "MEET",
            "FRIEND",
            "PLEASE",
            "SORRY",
            "WELCOME",
            "NAME",
            "UNDERSTAND",
            "WANT",
        )
    )
)
LABEL_PROFILES = {
    "default": (
        LABELS,
        MANIFEST_PATH,
        FEATURES_PATH,
        REVIEW_DIR,
        MODEL_PATH,
        REPORT_PATH,
        CONFUSION_PATH,
    ),
    "intro": (
        INTRO_LABELS,
        MANIFEST_DIR / "intro_static_samples.csv",
        DERIVED_DIR / "intro_static_features.npz",
        REVIEW_DIR / "intro_static",
        MODEL_DIR / "intro_static_pose_medoids.npz",
        MODEL_DIR / "intro_static_pose_evaluation.json",
        MODEL_DIR / "intro_static_pose_confusion.csv",
    ),
}
REQUIRED_HAND_COUNTS = {"HELP": 2}
REQUIRED_FACE_ANCHORS = {"FOOD"}
FEATURE_SCHEMA = "mudra_static_pose_v10"
HAND_FEATURE_DIM = 72
FEATURE_DIM = 32 + 2 * HAND_FEATURE_DIM
HAND_PRESENCE_OFFSET = 66
HAND_FACE_ALIGNMENT_OFFSET = 70
FACE_ANCHOR_FLAG_INDICES = (
    32 + HAND_FEATURE_DIM - 1,
    32 + 2 * HAND_FEATURE_DIM - 1,
)
POSE_INDICES = (11, 12, 13, 14, 15, 16, 23, 24)
MANIFEST_FIELDS = (
    "sample_id",
    "augmentation",
    "augmentation_parent_sample_id",
    "label",
    "source_folder_label",
    "verified_gloss",
    "sign_motion",
    "annotation_status",
    "review_status",
    "manual_reviewer",
    "review_date",
    "review_notes",
    "relative_path",
    "sha256",
    "filename",
    "source_clip_id",
    "frame_index_or_timestamp",
    "signer_id",
    "session_id",
    "source_origin",
    "license_or_consent_ref",
    "inferred_source_group",
    "source_group",
    "group_basis",
    "group_review_status",
    "width",
    "height",
    "extractor_name",
    "extractor_version",
    "feature_schema",
    "coordinate_convention",
    "pose_detected",
    "torso_anchor_min_visibility",
    "torso_anchor_source",
    "face_anchor_source",
    "detected_hands",
    "largest_hand_diagonal_px",
    "minimum_hand_diagonal_px",
    "tracking_status",
    "tracking_reason",
    "split",
    "feature_row",
)
MANUAL_FIELDS = (
    "label",
    "verified_gloss",
    "sign_motion",
    "annotation_status",
    "review_status",
    "manual_reviewer",
    "review_date",
    "review_notes",
    "source_clip_id",
    "frame_index_or_timestamp",
    "signer_id",
    "session_id",
    "source_origin",
    "license_or_consent_ref",
    "source_group",
    "group_review_status",
)


def preserve_reviewed_fields(
    record: dict[str, str],
    previous: dict[str, str],
) -> dict[str, str]:
    for field in MANUAL_FIELDS:
        if previous.get(field):
            record[field] = previous[field]
    return record


@dataclass(frozen=True)
class ExtractedSample:
    feature: np.ndarray
    pose_detected: bool
    anchor_min_visibility: float
    torso_anchor_source: str
    face_anchor_source: str
    detected_hands: int
    largest_hand_diagonal_px: float
    minimum_hand_diagonal_px: float
    tracking_status: str
    tracking_reason: str
    width: int
    height: int


def filename_group(filename: str) -> str:
    """Group numbered image families together without claiming video provenance."""
    stem = Path(filename).stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem).strip()
    return stem.casefold() or Path(filename).stem.casefold()


def grouped_splits(records: list[dict[str, str]], seed: int = 41) -> dict[str, str]:
    """Assign whole inferred groups to train/validation/test deterministically."""
    groups_by_label: dict[str, set[str]] = defaultdict(set)
    for record in records:
        source_group = record.get("source_group") or record["inferred_source_group"]
        groups_by_label[record["label"]].add(source_group)

    assignments: dict[str, str] = {}
    for label, group_set in groups_by_label.items():
        groups = sorted(group_set)
        groups.sort(
            key=lambda group: hashlib.sha256(f"{seed}:{label}:{group}".encode()).hexdigest()
        )
        count = len(groups)
        if count <= 1:
            test_count = 0
            validation_count = 0
        elif count == 2:
            test_count = 1
            validation_count = 0
        else:
            test_count = max(1, round(count * 0.2))
            validation_count = max(1, round(count * 0.15))
            while test_count + validation_count >= count:
                if validation_count > 1:
                    validation_count -= 1
                elif test_count > 1:
                    test_count -= 1
                else:
                    break

        train_end = count - test_count - validation_count
        validation_end = count - test_count
        for index, group in enumerate(groups):
            if index < train_end:
                split = "train"
            elif index < validation_end:
                split = "validation"
            else:
                split = "test"
            assignments[f"{label}:{group}"] = split
    return assignments


def _pose_face_anchor(
    pose_xyz: np.ndarray,
    pose_visibility: np.ndarray,
) -> tuple[np.ndarray | None, float, str]:
    if pose_visibility[0] < 0.35:
        return None, 0.0, "unavailable"
    for left_index, right_index, source in (
        (7, 8, "pose_nose_ears"),
        (2, 5, "pose_nose_eyes"),
    ):
        if min(pose_visibility[left_index], pose_visibility[right_index]) < 0.35:
            continue
        face_width = float(
            np.linalg.norm((pose_xyz[left_index] - pose_xyz[right_index])[:2])
        )
        if math.isfinite(face_width) and face_width > 1e-6:
            return pose_xyz[0], face_width, source
    return None, 0.0, "unavailable"


def _feature_has_face_anchor(feature: np.ndarray) -> bool:
    return any(feature[index] >= 0.5 for index in FACE_ANCHOR_FLAG_INDICES)


def _torso_anchor(
    pose_xyz: np.ndarray,
    pose_visibility: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, str]:
    left_shoulder, right_shoulder = pose_xyz[11], pose_xyz[12]
    shoulder_visibility = float(min(pose_visibility[11], pose_visibility[12]))
    shoulder_width = float(np.linalg.norm((left_shoulder - right_shoulder)[:2]))
    if not math.isfinite(shoulder_width) or shoulder_width <= 1e-6:
        raise ValueError("Shoulder width is missing or degenerate.")

    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    hips_visible = min(pose_visibility[23], pose_visibility[24]) >= 0.45
    if hips_visible:
        left_hip, right_hip = pose_xyz[23], pose_xyz[24]
        hip_midpoint = (left_hip + right_hip) / 2
        anchor_min_visibility = float(
            min(shoulder_visibility, pose_visibility[23], pose_visibility[24])
        )
        anchor_source = "shoulders_and_hips"
    else:
        hip_midpoint = shoulder_midpoint + np.asarray(
            (0.0, shoulder_width * 2.0, 0.0), dtype=np.float32
        )
        shoulder_axis = (right_shoulder - left_shoulder) / 2
        left_hip, right_hip = hip_midpoint - shoulder_axis, hip_midpoint + shoulder_axis
        anchor_min_visibility = shoulder_visibility
        anchor_source = "shoulders_estimated"
    return (
        (shoulder_midpoint + hip_midpoint) / 2,
        shoulder_width,
        left_hip,
        right_hip,
        anchor_min_visibility,
        anchor_source,
    )


def build_pose_feature(
    pose_xyz: np.ndarray,
    pose_visibility: np.ndarray,
    hands_by_detection: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, int, str]:
    """Return pose features, anchor confidence, hand count, and torso anchor source."""
    if pose_xyz.shape != (33, 3) or pose_visibility.shape != (33,):
        raise ValueError("Pose data must have 33 xyz points and 33 visibility values.")

    (
        torso_center,
        shoulder_width,
        left_hip,
        right_hip,
        anchor_min_visibility,
        torso_anchor_source,
    ) = _torso_anchor(pose_xyz, pose_visibility)
    hips_visible = min(pose_visibility[23], pose_visibility[24]) >= 0.45
    left_shoulder, right_shoulder = pose_xyz[11], pose_xyz[12]

    face_center, face_width, _ = _pose_face_anchor(pose_xyz, pose_visibility)

    body_features: list[float] = []
    body_confidence: list[float] = []
    for index in POSE_INDICES:
        visibility = float(pose_visibility[index])
        if index in (23, 24) and not hips_visible:
            visibility = 0.0
        body_confidence.append(visibility)
        if index in (11, 12, 23, 24):
            if index in (23, 24) and not hips_visible:
                hip = left_hip if index == 23 else right_hip
                body_features.extend(((hip - torso_center) / shoulder_width).tolist())
            elif visibility > 0:
                body_features.extend(
                    ((pose_xyz[index] - torso_center) / shoulder_width).tolist()
                )
            else:
                body_features.extend((0.0, 0.0, 0.0))

    arm_segments = ((11, 13), (13, 15), (12, 14), (14, 16))
    for start, end in arm_segments:
        vector = pose_xyz[end] - pose_xyz[start]
        vector_length = float(np.linalg.norm(vector))
        if min(pose_visibility[start], pose_visibility[end]) < 0.1 or vector_length <= 1e-6:
            body_features.extend((0.0, 0.0, 0.0))
        else:
            body_features.extend((vector / vector_length).tolist())
    body_features.extend(body_confidence)

    body_right_axis = right_shoulder - left_shoulder
    body_right_axis[2] = 0.0
    body_right_axis = body_right_axis / max(
        float(np.linalg.norm(body_right_axis[:2])),
        1e-6,
    )
    detected_hands = sorted(
        hands_by_detection.values(),
        key=lambda hand_xyz: float(
            np.dot(hand_xyz[0] - torso_center, body_right_axis)
        ),
    )
    hand_blocks: list[list[float]] = []
    for hand_xyz in detected_hands[:2]:
        if hand_xyz.shape != (21, 3):
            raise ValueError("Each hand must contain 21 xyz points.")
        wrist = hand_xyz[0]
        palm_width = float(np.linalg.norm((hand_xyz[5] - hand_xyz[17])[:2]))
        if not math.isfinite(palm_width) or palm_width <= 1e-6:
            hand_blocks.append([0.0] * HAND_FEATURE_DIM)
            continue
        wrist_in_body = (wrist - torso_center) / shoulder_width
        hand_shape = (hand_xyz - wrist) / palm_width
        block = wrist_in_body.tolist()
        block.extend(hand_shape.reshape(-1).tolist())
        block.append(1.0)
        if face_center is None:
            block.extend((0.0,) * 5)
        else:
            wrist_from_face = (wrist - face_center) / face_width
            index_direction = hand_xyz[8] - hand_xyz[5]
            face_direction = face_center - wrist
            index_length = float(np.linalg.norm(index_direction))
            face_distance = float(np.linalg.norm(face_direction))
            alignment = (
                float(np.dot(index_direction / index_length, face_direction / face_distance))
                if index_length > 1e-6 and face_distance > 1e-6
                else 0.0
            )
            block.extend(wrist_from_face.tolist())
            block.extend((alignment, 1.0))
        hand_blocks.append(block)

    while len(hand_blocks) < 2:
        hand_blocks.append([0.0] * HAND_FEATURE_DIM)
    hand_features = [value for block in hand_blocks for value in block]

    feature = np.asarray(body_features + hand_features, dtype=np.float32)
    if feature.shape != (FEATURE_DIM,):
        raise AssertionError(f"Unexpected feature shape: {feature.shape}")
    return feature, anchor_min_visibility, len(detected_hands), torso_anchor_source


def _pose_array(pose_landmarks: Any, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.zeros((33, 3), dtype=np.float32)
    visibility = np.zeros(33, dtype=np.float32)
    for index, landmark in enumerate(pose_landmarks):
        xyz[index] = (landmark.x * width, landmark.y * height, landmark.z * width)
        visibility[index] = float(getattr(landmark, "visibility", 0.0) or 0.0)
    return xyz, visibility


def _hand_arrays(
    result: Any,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    return _hand_arrays_from_landmarks(
        result.hand_landmarks,
        width,
        height,
    )


def _hand_arrays_from_landmarks(
    hand_landmarks: list[Any],
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    by_detection: dict[str, np.ndarray] = {}
    for index, hand in enumerate(hand_landmarks):
        coordinates = np.asarray(
            [
                (landmark.x * width, landmark.y * height, landmark.z * width)
                for landmark in hand
            ],
            dtype=np.float32,
        )
        by_detection[f"hand_{index}"] = coordinates
    return by_detection


def _hand_diagonals(hand_landmarks: list[Any], width: int, height: int) -> list[float]:
    return [
        float(
            np.linalg.norm(
                np.ptp(
                    np.asarray(
                        [(landmark.x * width, landmark.y * height) for landmark in hand],
                        dtype=np.float32,
                    ),
                    axis=0,
                )
            )
        )
        for hand in hand_landmarks
    ]


class StaticPoseRecognizer:
    """Classify one live pose with the exploratory still-image medoid model."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(f"Static-pose model does not exist: {model_path}")
        with np.load(model_path, allow_pickle=False) as archive:
            self.classes = archive["classes"].astype(str).tolist()
            self.prototypes = archive["prototypes"].astype(np.float32)
            self.required_hand_counts = (
                archive["required_hand_counts"].astype(np.int32)
                if "required_hand_counts" in archive
                else np.zeros(len(self.classes), dtype=np.int32)
            )
            self.required_face_anchors = (
                archive["required_face_anchors"].astype(bool)
                if "required_face_anchors" in archive
                else np.asarray(
                    [label in REQUIRED_FACE_ANCHORS for label in self.classes],
                    dtype=bool,
                )
            )
            self.thresholds = archive["rejection_thresholds"].astype(np.float32)
            self.threshold_sources = (
                archive["threshold_sources"].astype(str).tolist()
                if "threshold_sources" in archive
                else ["unknown"] * len(self.classes)
            )
            feature_schema = archive["feature_schema"].astype(str).tolist()
        if feature_schema != [FEATURE_SCHEMA]:
            raise ValueError(f"Unsupported feature schema in model: {feature_schema}")
        if (
            self.prototypes.shape != (len(self.classes), FEATURE_DIM)
            or self.required_hand_counts.shape != (len(self.classes),)
            or self.required_face_anchors.shape != (len(self.classes),)
            or self.thresholds.shape != (len(self.classes),)
            or len(self.threshold_sources) != len(self.classes)
        ):
            raise ValueError("Static-pose model arrays have inconsistent shapes.")

    def predict_feature(
        self,
        feature: np.ndarray,
        detected_hand_count: int | None = None,
    ) -> dict[str, Any]:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape != (FEATURE_DIM,) or not np.isfinite(feature).all():
            raise ValueError(f"Live feature must contain {FEATURE_DIM} finite values.")
        distances = np.linalg.norm(self.prototypes - feature, axis=1)
        eligible = np.ones(len(self.classes), dtype=bool)
        if detected_hand_count is not None:
            eligible = self.required_hand_counts == detected_hand_count
        if not eligible.any():
            return {
                "prediction": None,
                "reason": f"no_class_supports_{detected_hand_count}_tracked_hands",
            }
        distances[~eligible] = np.inf
        best_index = int(np.argmin(distances))
        nearest_label = self.classes[best_index]
        if (
            self.required_face_anchors[best_index]
            and not _feature_has_face_anchor(feature)
        ):
            return {
                "prediction": None,
                "nearest_label": nearest_label,
                "reason": f"{nearest_label}_requires_face_anchor",
                "required_hand_count": int(self.required_hand_counts[best_index]),
            }
        distance = float(distances[best_index])
        threshold = float(self.thresholds[best_index])
        calibrated = math.isfinite(threshold)
        accepted = prediction_is_accepted(distance, threshold)
        return {
            "prediction": self.classes[best_index] if accepted else None,
            "nearest_label": self.classes[best_index],
            "distance": distance,
            "threshold": threshold if calibrated else None,
            "threshold_source": self.threshold_sources[best_index],
            "required_hand_count": int(self.required_hand_counts[best_index]),
            "reason": (
                None
                if accepted
                else "distance_exceeds_experimental_threshold"
                if calibrated
                else "no_reliable_validation_threshold"
            ),
        }

    def supported_learning_profiles(self) -> dict[str, set[int]]:
        profiles: dict[str, set[int]] = {}
        for label, hand_count, threshold in zip(
            self.classes,
            self.required_hand_counts,
            self.thresholds,
        ):
            if math.isfinite(float(threshold)):
                profiles.setdefault(label, set()).add(int(hand_count))
        return profiles

    def expected_hand_positions(
        self,
        label: str,
        hand_count: int,
        pose_xyz: np.ndarray,
        pose_visibility: np.ndarray,
    ) -> list[tuple[float, float]]:
        profile_indices = [
            index
            for index, (model_label, model_hand_count) in enumerate(
                zip(self.classes, self.required_hand_counts)
            )
            if model_label == label and int(model_hand_count) == hand_count
        ]
        if not profile_indices:
            return []

        torso_center, shoulder_width, _, _, _, _ = _torso_anchor(
            pose_xyz,
            pose_visibility,
        )
        prototype = self.prototypes[profile_indices[0]]
        positions: list[tuple[float, float]] = []
        for slot in range(hand_count):
            block_start = 32 + slot * HAND_FEATURE_DIM
            if prototype[block_start + HAND_PRESENCE_OFFSET] < 0.5:
                continue
            wrist = prototype[block_start : block_start + 3]
            position = torso_center[:2] + wrist[:2] * shoulder_width
            positions.append((float(position[0]), float(position[1])))
        return positions

    def predict_landmarks(
        self,
        pose_landmarks: list[Any] | None,
        hand_landmarks: list[Any],
        width: int,
        height: int,
    ) -> dict[str, Any]:
        if not pose_landmarks:
            return {"prediction": None, "reason": "pose_not_detected"}
        pose_xyz, pose_visibility = _pose_array(pose_landmarks, width, height)
        _, _, face_anchor_source = _pose_face_anchor(pose_xyz, pose_visibility)
        hands_by_detection = _hand_arrays_from_landmarks(
            hand_landmarks,
            width,
            height,
        )
        detected_hand_count = len(hand_landmarks)
        largest_hand = max(_hand_diagonals(hand_landmarks, width, height), default=0.0)
        minimum_hand = max(30.0, math.hypot(width, height) * 0.015)
        try:
            feature, anchor_visibility, _, torso_anchor_source = build_pose_feature(
                pose_xyz,
                pose_visibility,
                hands_by_detection,
            )
        except ValueError as exc:
            return {"prediction": None, "reason": str(exc).replace(" ", "_")}
        status, reason = _tracking_result(
            True,
            anchor_visibility,
            detected_hand_count,
            largest_hand,
            minimum_hand,
            [],
        )
        if status != "accepted":
            return {"prediction": None, "reason": reason}
        result = self.predict_feature(feature, detected_hand_count=detected_hand_count)
        result["tracking_status"] = status
        result["torso_anchor_source"] = torso_anchor_source
        result["face_anchor_source"] = face_anchor_source
        return result


class StaticPredictionStabilizer:
    """Require one label to persist before showing it as a live result."""

    def __init__(self, hold_s: float = 0.45):
        if not math.isfinite(hold_s) or hold_s <= 0:
            raise ValueError("Prediction hold time must be a positive finite number.")
        self.hold_s = hold_s
        self.label: str | None = None
        self.candidate: str | None = None
        self.candidate_since: float | None = None

    def reset(self) -> None:
        self.label = None
        self.candidate = None
        self.candidate_since = None

    def update(
        self,
        timestamp: float,
        candidate: str | None,
        active: bool,
    ) -> tuple[str | None, str]:
        if not active:
            self.reset()
            return None, "inactive"
        if candidate is None:
            self.reset()
            return None, "unknown"
        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = timestamp
            self.label = None
            return None, "confirming"
        if self.candidate_since is None:
            self.candidate_since = timestamp
        if timestamp - self.candidate_since < self.hold_s:
            return None, "confirming"
        self.label = candidate
        return self.label, "recognized"


def _tracking_result(
    pose_detected: bool,
    anchor_visibility: float,
    hands: int,
    largest_hand_diagonal_px: float,
    minimum_hand_diagonal_px: float,
    errors: list[str],
) -> tuple[str, str]:
    reasons = list(errors)
    if not pose_detected:
        reasons.append("pose_not_detected")
    elif anchor_visibility < 0.45:
        reasons.append("torso_anchor_visibility_below_0.45")
    if hands == 0:
        reasons.append("no_hand_detected")
    elif largest_hand_diagonal_px < minimum_hand_diagonal_px:
        reasons.append("hand_too_small_for_tracking")
    return ("rejected", ";".join(reasons)) if reasons else ("accepted", "tracking_gate_passed")


def mirrored_hand_count_mismatch_reason(
    original_hand_count: int,
    mirrored_hand_count: int,
) -> str | None:
    if original_hand_count == mirrored_hand_count:
        return None
    return (
        "mirror_hand_count_mismatch:"
        f"original_{original_hand_count}_mirrored_{mirrored_hand_count}"
    )


def extract_image(
    path: Path,
    pose_detector: Any,
    hand_detector: Any,
    mirror: bool = True,
) -> ExtractedSample:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        zero_feature = np.zeros(FEATURE_DIM, dtype=np.float32)
        return ExtractedSample(
            feature=zero_feature,
            pose_detected=False,
            anchor_min_visibility=0.0,
            torso_anchor_source="unavailable",
            face_anchor_source="unavailable",
            detected_hands=0,
            largest_hand_diagonal_px=0.0,
            minimum_hand_diagonal_px=0.0,
            tracking_status="rejected",
            tracking_reason="image_decode_failed",
            width=0,
            height=0,
        )

    if mirror:
        image = cv2.flip(image, 1)
    height, width = image.shape[:2]
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    )
    pose_result = pose_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)

    errors: list[str] = []
    pose_detected = bool(pose_result.pose_landmarks)
    hands_by_detection = _hand_arrays(
        hand_result,
        width,
        height,
    )
    detected_hand_count = len(hand_result.hand_landmarks)
    hand_diagonals = [
        float(
            np.linalg.norm(
                np.ptp(
                    np.asarray(
                        [(landmark.x * width, landmark.y * height) for landmark in hand],
                        dtype=np.float32,
                    ),
                    axis=0,
                )
            )
        )
        for hand in hand_result.hand_landmarks
    ]
    largest_hand_diagonal_px = max(hand_diagonals, default=0.0)
    minimum_hand_diagonal_px = max(30.0, math.hypot(width, height) * 0.015)
    if pose_detected:
        pose_xyz, visibility = _pose_array(pose_result.pose_landmarks[0], width, height)
        _, _, face_anchor_source = _pose_face_anchor(pose_xyz, visibility)
        try:
            feature, anchor_visibility, _, torso_anchor_source = build_pose_feature(
                pose_xyz,
                visibility,
                hands_by_detection,
            )
        except ValueError as exc:
            errors.append(str(exc).replace(" ", "_"))
            feature = np.zeros(FEATURE_DIM, dtype=np.float32)
            anchor_visibility = 0.0
            torso_anchor_source = "unavailable"
    else:
        feature = np.zeros(FEATURE_DIM, dtype=np.float32)
        anchor_visibility = 0.0
        torso_anchor_source = "unavailable"
        face_anchor_source = "unavailable"

    status, reason = _tracking_result(
        pose_detected,
        anchor_visibility,
        detected_hand_count,
        largest_hand_diagonal_px,
        minimum_hand_diagonal_px,
        errors,
    )
    return ExtractedSample(
        feature=feature,
        pose_detected=pose_detected,
        anchor_min_visibility=anchor_visibility,
        torso_anchor_source=torso_anchor_source,
        face_anchor_source=face_anchor_source,
        detected_hands=detected_hand_count,
        largest_hand_diagonal_px=largest_hand_diagonal_px,
        minimum_hand_diagonal_px=minimum_hand_diagonal_px,
        tracking_status=status,
        tracking_reason=reason,
        width=width,
        height=height,
    )


def _sample_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detectors() -> tuple[Any, Any]:
    pose_model = ROOT / "models" / "pose_landmarker_lite.task"
    hand_model = ROOT / "models" / "hand_landmarker.task"
    if not pose_model.is_file() or not hand_model.is_file():
        missing = [str(path) for path in (pose_model, hand_model) if not path.is_file()]
        raise FileNotFoundError("Missing MediaPipe model asset(s): " + ", ".join(missing))
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    return (
        vision.PoseLandmarker.create_from_options(pose_options),
        vision.HandLandmarker.create_from_options(hand_options),
    )


def _contact_sheet(records: list[dict[str, str]], page_path: Path) -> None:
    columns, rows = 4, 6
    cell_width, cell_height = 340, 260
    sheet = np.full((rows * cell_height, columns * cell_width, 3), (248, 246, 239), dtype=np.uint8)
    for index, record in enumerate(records[: columns * rows]):
        image = cv2.imread(str(ROOT / record["relative_path"]))
        if image is None:
            continue
        cell_x = (index % columns) * cell_width
        cell_y = (index // columns) * cell_height
        thumb_width, thumb_height = cell_width - 16, cell_height - 54
        scale = min(thumb_width / image.shape[1], thumb_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        x = cell_x + (cell_width - resized.shape[1]) // 2
        y = cell_y + 8 + (thumb_height - resized.shape[0]) // 2
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        color = (34, 118, 90) if record["tracking_status"] == "accepted" else (45, 65, 190)
        cv2.putText(
            sheet,
            record["filename"][:44],
            (cell_x + 8, cell_y + cell_height - 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (31, 43, 51),
            1,
            cv2.LINE_AA,
        )
        state = f'{record["inferred_source_group"]} | {record["tracking_status"]}'
        cv2.putText(
            sheet,
            state[:48],
            (cell_x + 8, cell_y + cell_height - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    page_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(page_path), sheet):
        raise OSError(f"Could not save review contact sheet: {page_path}")


def prepare_dataset(profile: str = "default") -> None:
    try:
        labels, manifest_path, features_path, review_dir, *_ = LABEL_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown static vocabulary profile: {profile}") from exc
    records: list[dict[str, str]] = []
    feature_rows: list[np.ndarray] = []
    seen_hashes: dict[str, str] = {}
    prior_rows: dict[str, dict[str, str]] = {}
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            prior_rows = {
                row["sample_id"]: row for row in csv.DictReader(stream) if row.get("sample_id")
            }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    pose_detector, hand_detector = _detectors()
    with pose_detector, hand_detector:
        for label in labels:
            folder = IMAGE_ROOT / label
            if not folder.is_dir():
                print(f"Warning: selected label folder is missing: {folder}", file=sys.stderr)
                continue
            for path in sorted(folder.glob("*.jpg"), key=lambda item: item.name.casefold()):
                relative_path = path.relative_to(ROOT).as_posix()
                sample_hash = _sample_sha256(path)
                duplicate_of = seen_hashes.get(sample_hash, "")
                seen_hashes.setdefault(sample_hash, relative_path)
                base_sample_id = f"{label.lower()}::{path.name}"
                original_hand_count: int | None = None
                for augmentation, mirror in (("original", False), ("mirrored", True)):
                    extracted = extract_image(
                        path,
                        pose_detector,
                        hand_detector,
                        mirror=mirror,
                    )
                    sample_id = (
                        base_sample_id
                        if augmentation == "original"
                        else f"{base_sample_id}::mirrored"
                    )
                    status = extracted.tracking_status
                    reason = extracted.tracking_reason
                    if augmentation == "original":
                        original_hand_count = extracted.detected_hands
                    if duplicate_of:
                        status = "rejected"
                        reason = "exact_duplicate:" + duplicate_of
                    elif augmentation == "mirrored":
                        if original_hand_count is None:
                            raise RuntimeError(
                                "Mirrored sample has no preceding original hand count."
                            )
                        mismatch_reason = mirrored_hand_count_mismatch_reason(
                            original_hand_count,
                            extracted.detected_hands,
                        )
                        if mismatch_reason is not None:
                            status = "rejected"
                            reason = mismatch_reason
                    record = {
                        "sample_id": sample_id,
                        "augmentation": augmentation,
                        "augmentation_parent_sample_id": (
                            "" if augmentation == "original" else base_sample_id
                        ),
                        "label": label,
                        "source_folder_label": label,
                        "verified_gloss": "",
                        "sign_motion": "unknown",
                        "annotation_status": "folder_label_unverified",
                        "review_status": "needs_review",
                        "manual_reviewer": "",
                        "review_date": "",
                        "review_notes": "",
                        "relative_path": relative_path,
                        "sha256": sample_hash,
                        "filename": path.name,
                        "source_clip_id": "",
                        "frame_index_or_timestamp": "",
                        "signer_id": "",
                        "session_id": "",
                        "source_origin": "",
                        "license_or_consent_ref": "",
                        "inferred_source_group": filename_group(path.name),
                        "source_group": filename_group(path.name),
                        "group_basis": "filename_family_before_trailing_parenthesized_number",
                        "group_review_status": "inferred_unverified",
                        "width": str(extracted.width),
                        "height": str(extracted.height),
                        "extractor_name": "MediaPipe Pose Landmarker Lite + Hand Landmarker",
                        "extractor_version": str(getattr(mp, "__version__", "unknown")),
                        "feature_schema": FEATURE_SCHEMA,
                        "coordinate_convention": (
                            "original_and_horizontal_flip_variants;"
                            "hand_landmarks_only_no_handedness_features_or_gate;"
                            "hands_sorted_by_body_lateral_position;"
                            "live_input_selfie_mirrored;"
                            "image_px_xyz_z_scaled_by_width;torso_centered;"
                            "shoulder_width_scaled;arm_bones_unit_directions;"
                            "hand_shape_palm_width_scaled;"
                            "pose_nose_face_anchor_ears_or_eyes;"
                            "hand_wrist_face_relative_in_face_widths;"
                            "index_direction_dot_wrist_to_nose_direction"
                        ),
                        "pose_detected": str(extracted.pose_detected).lower(),
                        "torso_anchor_min_visibility": f"{extracted.anchor_min_visibility:.4f}",
                        "torso_anchor_source": extracted.torso_anchor_source,
                        "face_anchor_source": extracted.face_anchor_source,
                        "detected_hands": str(extracted.detected_hands),
                        "largest_hand_diagonal_px": f"{extracted.largest_hand_diagonal_px:.2f}",
                        "minimum_hand_diagonal_px": f"{extracted.minimum_hand_diagonal_px:.2f}",
                        "tracking_status": status,
                        "tracking_reason": reason,
                        "split": "",
                        "feature_row": str(len(feature_rows)),
                    }
                    previous = prior_rows.get(
                        sample_id,
                        prior_rows.get(base_sample_id, {}),
                    )
                    preserve_reviewed_fields(record, previous)
                    records.append(record)
                    feature_rows.append(extracted.feature)
                    print(
                        f'{record["tracking_status"]:8} {augmentation:8} {relative_path}'
                    )

    assignments = grouped_splits(records)
    for record in records:
        source_group = record.get("source_group") or record["inferred_source_group"]
        record["split"] = assignments.get(
            f'{record["label"]}:{source_group}', "train"
        )

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    features = (
        np.stack(feature_rows)
        if feature_rows
        else np.empty((0, FEATURE_DIM), dtype=np.float32)
    )
    np.savez_compressed(
        features_path,
        sample_ids=np.asarray([record["sample_id"] for record in records], dtype=np.str_),
        features=features,
    )
    for label in labels:
        label_records = [
            record
            for record in records
            if record["source_folder_label"] == label
            and record["augmentation"] == "original"
        ]
        for page_number, start in enumerate(range(0, len(label_records), 24), start=1):
            _contact_sheet(
                label_records[start : start + 24],
                review_dir / f"{label.lower()}_page_{page_number:02d}.jpg",
            )

    counts: dict[str, dict[str, int]] = {}
    for label in labels:
        class_records = [record for record in records if record["label"] == label]
        counts[label] = {
            "samples": len(class_records),
            "source_images": sum(item["augmentation"] == "original" for item in class_records),
            "mirrored_augmentations": sum(
                item["augmentation"] == "mirrored" for item in class_records
            ),
            "accepted_tracking": sum(item["tracking_status"] == "accepted" for item in class_records),
            "rejected_tracking": sum(item["tracking_status"] == "rejected" for item in class_records),
            "inferred_filename_groups": len(
                {item["source_group"] for item in class_records}
            ),
        }
    print(f"\nCSV: {manifest_path}")
    print(f"Features: {features_path}")
    print(f"Review sheets: {review_dir}")
    print(json.dumps(counts, indent=2))


def _load_dataset(
    manifest_path: Path = MANIFEST_PATH,
    features_path: Path = FEATURES_PATH,
) -> tuple[list[dict[str, str]], np.ndarray]:
    if not manifest_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(
            f"Run `{Path(__file__).name} prepare` before training; missing "
            f"{manifest_path} or {features_path}."
        )
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    with np.load(features_path, allow_pickle=False) as archive:
        feature_ids = archive["sample_ids"].astype(str)
        features = archive["features"].astype(np.float32)
    if len(records) != len(feature_ids) or len(records) != len(features):
        raise ValueError("Manifest and feature archive row counts do not match.")
    for index, record in enumerate(records):
        if record["sample_id"] != feature_ids[index]:
            raise ValueError(f"Manifest/feature sample order mismatch at row {index}.")
    return records, features


def load_learning_examples(
    eligible_profiles: dict[str, set[int]] | None = None,
) -> list[tuple[str, Path, int]]:
    """Select one accepted, unaugmented reference image per inferred source group."""
    if not MANIFEST_PATH.is_file():
        return []
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    selected: dict[tuple[str, int, str], Path] = {}
    for record in records:
        if (
            record.get("label") not in LABELS
            or (
                eligible_profiles is not None
                and int(record.get("detected_hands", "0"))
                not in eligible_profiles.get(record.get("label", ""), set())
            )
            or record.get("augmentation") != "original"
            or record.get("tracking_status") != "accepted"
        ):
            continue
        group = record.get("source_group") or record.get("inferred_source_group", "")
        if not group:
            continue
        image_path = ROOT / record["relative_path"]
        if not image_path.is_file():
            continue
        hand_count = int(record["detected_hands"])
        selected.setdefault((record["label"], hand_count, group), image_path)
    return [
        (label, selected[(label, hand_count, group)], hand_count)
        for label, hand_count, group in sorted(selected)
    ]


def _medoid(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 1:
        return vectors[0]
    center = np.median(vectors, axis=0)
    distances = np.linalg.norm(vectors - center, axis=1)
    return vectors[int(np.argmin(distances))]


def prediction_is_accepted(distance: float, threshold: float) -> bool:
    return math.isfinite(threshold) and distance <= threshold


def train_prototype(profile: str = "default") -> None:
    try:
        labels, manifest_path, features_path, _, model_path, report_path, confusion_path = (
            LABEL_PROFILES[profile]
        )
    except KeyError as exc:
        raise ValueError(f"Unknown static vocabulary profile: {profile}") from exc
    records, features = _load_dataset(manifest_path, features_path)
    accepted_indices = [
        index for index, record in enumerate(records) if record["tracking_status"] == "accepted"
    ]
    train_indices = [
        index for index in accepted_indices if records[index]["split"] == "train"
    ]
    validation_indices = [
        index
        for index in accepted_indices
        if records[index]["split"] == "validation"
        and records[index]["augmentation"] == "original"
    ]
    test_indices = [
        index
        for index in accepted_indices
        if records[index]["split"] == "test"
        and records[index]["augmentation"] == "original"
    ]

    model_hand_counts: list[int] = []
    medoids: list[np.ndarray] = []
    model_classes: list[str] = []
    profile_train_indices: dict[tuple[str, int], list[int]] = {}
    for label in sorted({records[index]["label"] for index in train_indices}):
        available_hand_counts = sorted(
            {
                int(records[index]["detected_hands"])
                for index in train_indices
                if records[index]["label"] == label
            }
        )
        required_count = REQUIRED_HAND_COUNTS.get(label)
        if required_count is not None:
            available_hand_counts = [
                hand_count
                for hand_count in available_hand_counts
                if hand_count == required_count
            ]
        for hand_count in available_hand_counts:
            profile_indices = [
                index
                for index in train_indices
                if records[index]["label"] == label
                and int(records[index]["detected_hands"]) == hand_count
            ]
            if not profile_indices:
                continue
            profile_train_indices[(label, hand_count)] = profile_indices
            model_classes.append(label)
            model_hand_counts.append(hand_count)
            medoids.append(_medoid(features[profile_indices]))
    if len(model_classes) < 2:
        raise ValueError(
            "Need accepted training samples for at least two label/hand-count profiles; "
            "review tracking results and source-group assignments."
        )

    prototypes = np.stack(medoids).astype(np.float32)
    thresholds: list[float] = []
    threshold_sources: list[str] = []
    for (label, hand_count), prototype in zip(profile_train_indices, prototypes):
        profile_train = profile_train_indices[(label, hand_count)]
        groups = sorted({records[index]["source_group"] for index in profile_train})
        group_held_out_distances: list[float] = []
        if len(groups) >= 3:
            for group in groups:
                held_out = [
                    index
                    for index in profile_train
                    if records[index]["source_group"] == group
                ]
                fit = [
                    index
                    for index in profile_train
                    if records[index]["source_group"] != group
                ]
                if not fit:
                    continue
                fold_medoid = _medoid(features[fit])
                group_held_out_distances.extend(
                    float(np.linalg.norm(features[index] - fold_medoid))
                    for index in held_out
                )
        if group_held_out_distances:
            thresholds.append(float(np.percentile(group_held_out_distances, 90)))
            threshold_sources.append("inferred_group_holdout_p90_experimental")
            continue

        class_validation = [
            index
            for index in validation_indices
            if records[index]["label"] == label
            and int(records[index]["detected_hands"]) == hand_count
        ]
        validation_groups = {records[index]["source_group"] for index in class_validation}
        if len(validation_groups) >= 2:
            distances = [
                float(np.linalg.norm(features[index] - prototype))
                for index in class_validation
            ]
            thresholds.append(float(np.percentile(distances, 95)))
            threshold_sources.append("held_out_validation_p95")
        else:
            thresholds.append(float("nan"))
            threshold_sources.append("unavailable")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        classes=np.asarray(model_classes, dtype=np.str_),
        prototypes=prototypes,
        required_hand_counts=np.asarray(model_hand_counts, dtype=np.int32),
        required_face_anchors=np.asarray(
            [label in REQUIRED_FACE_ANCHORS for label in model_classes],
            dtype=np.bool_,
        ),
        rejection_thresholds=np.asarray(thresholds, dtype=np.float32),
        threshold_sources=np.asarray(threshold_sources, dtype=np.str_),
        feature_schema=np.asarray([FEATURE_SCHEMA], dtype=np.str_),
    )

    unique_classes = sorted(set(model_classes))
    confusion = {
        true: {predicted: 0 for predicted in (*unique_classes, "UNKNOWN")}
        for true in unique_classes
    }
    correct = 0
    evaluated = 0
    for index in test_indices:
        true_label = records[index]["label"]
        if true_label not in confusion:
            continue
        distances = np.linalg.norm(prototypes - features[index], axis=1)
        hand_count = int(records[index]["detected_hands"])
        eligible = np.asarray(model_hand_counts) == hand_count
        if not eligible.any():
            predicted = "UNKNOWN"
            confusion[true_label][predicted] += 1
            evaluated += 1
            continue
        distances[~eligible] = np.inf
        best_index = int(np.argmin(distances))
        requires_face_anchor = (
            model_classes[best_index] in REQUIRED_FACE_ANCHORS
            and not _feature_has_face_anchor(features[index])
        )
        predicted = "UNKNOWN" if requires_face_anchor else model_classes[best_index]
        confusion[true_label][predicted] += 1
        evaluated += 1
        correct += int(predicted == true_label)

    class_group_counts = {}
    split_counts = {}
    for label in unique_classes:
        label_records = [
            record for record in records if record["label"] == label
        ]
        class_group_counts[label] = len(
            {record["source_group"] for record in label_records}
        )
        split_counts[label] = {
            split: sum(
                record["label"] == label
                and record["split"] == split
                and record["tracking_status"] == "accepted"
                for record in records
            )
            for split in ("train", "validation", "test")
        }

    report = {
        "task": "still-image static-pose nearest-medoid baseline",
        "warning": (
            "Not a dynamic sign recognizer. Folder labels are user-confirmed but have not been "
            "independently reviewed by an ISL-fluent signer; filename-derived source groups "
            "remain unverified. Scores are exploratory and do not establish generalization. "
            "Inferred-group holdout rejection thresholds are demo-only estimates. "
            "Every source image's horizontal mirror shares its source-group split."
        ),
        "feature_schema": FEATURE_SCHEMA,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "model": str(model_path.relative_to(ROOT)),
        "classes": unique_classes,
        "profiles": [
            {"label": label, "hand_count": hand_count}
            for label, hand_count in zip(model_classes, model_hand_counts)
        ],
        "accepted_samples": len(accepted_indices),
        "accepted_original_samples": sum(
            records[index]["augmentation"] == "original"
            for index in accepted_indices
        ),
        "accepted_mirrored_augmentations": sum(
            records[index]["augmentation"] == "mirrored"
            for index in accepted_indices
        ),
        "class_inferred_filename_group_counts": class_group_counts,
        "accepted_samples_by_split": split_counts,
        "test_samples_evaluated": evaluated,
        "test_accuracy": correct / evaluated if evaluated else None,
        "rejection_threshold_by_profile": {
            f"{label}:{hand_count}_hands": (
                None if math.isnan(float(threshold)) else float(threshold)
            )
            for label, hand_count, threshold in zip(
                model_classes,
                model_hand_counts,
                thresholds,
            )
        },
        "rejection_threshold_source_by_profile": {
            f"{label}:{hand_count}_hands": source
            for label, hand_count, source in zip(
                model_classes,
                model_hand_counts,
                threshold_sources,
            )
        },
        "required_hand_count_by_class": {
            label: sorted(
                {
                    hand_count
                    for model_label, hand_count in zip(model_classes, model_hand_counts)
                    if model_label == label
                }
            )
            for label in unique_classes
        },
        "required_face_anchor_by_class": {
            label: label in REQUIRED_FACE_ANCHORS for label in unique_classes
        },
        "confusion_matrix": confusion,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with confusion_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_label", *unique_classes, "UNKNOWN"])
        for label in unique_classes:
            writer.writerow(
                [label, *[confusion[label][predicted] for predicted in (*unique_classes, "UNKNOWN")]]
            )
    print(json.dumps(report, indent=2))
    print(f"\nModel: {model_path}")
    print(f"Report: {report_path}")
    print(f"Confusion matrix: {confusion_path}")


def predict_image(image_path: Path, model_path: Path) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Prototype model does not exist: {model_path}")
    with np.load(model_path, allow_pickle=False) as archive:
        classes = archive["classes"].astype(str).tolist()
        prototypes = archive["prototypes"].astype(np.float32)
        required_hand_counts = (
            archive["required_hand_counts"].astype(np.int32)
            if "required_hand_counts" in archive
            else np.zeros(len(classes), dtype=np.int32)
        )
        required_face_anchors = (
            archive["required_face_anchors"].astype(bool)
            if "required_face_anchors" in archive
            else np.asarray(
                [label in REQUIRED_FACE_ANCHORS for label in classes],
                dtype=bool,
            )
        )
        thresholds = archive["rejection_thresholds"].astype(np.float32)
        threshold_sources = archive["threshold_sources"].astype(str).tolist()
        feature_schema = archive["feature_schema"].astype(str).tolist()
    if feature_schema != [FEATURE_SCHEMA]:
        raise ValueError(f"Unsupported feature schema in model: {feature_schema}")

    pose_detector, hand_detector = _detectors()
    with pose_detector, hand_detector:
        sample = extract_image(image_path, pose_detector, hand_detector)
    if sample.tracking_status != "accepted":
        raise ValueError(f"Image rejected by tracking gate: {sample.tracking_reason}")
    distances = np.linalg.norm(prototypes - sample.feature, axis=1)
    eligible = required_hand_counts == sample.detected_hands
    if not eligible.any():
        print(
            json.dumps(
                {
                    "image": str(image_path),
                    "tracking_status": sample.tracking_status,
                    "torso_anchor_source": sample.torso_anchor_source,
                    "face_anchor_source": sample.face_anchor_source,
                    "detected_hands": sample.detected_hands,
                    "prediction": "unknown",
                    "rejection_reason": (
                        f"no_class_supports_{sample.detected_hands}_tracked_hands"
                    ),
                },
                indent=2,
            )
        )
        return
    distances[~eligible] = np.inf
    best_index = int(np.argmin(distances))
    if (
        required_face_anchors[best_index]
        and not _feature_has_face_anchor(sample.feature)
    ):
        print(
            json.dumps(
                {
                    "image": str(image_path),
                    "tracking_status": sample.tracking_status,
                    "face_anchor_source": sample.face_anchor_source,
                    "detected_hands": sample.detected_hands,
                    "prediction": "unknown",
                    "nearest_label": classes[best_index],
                    "rejection_reason": "FOOD_requires_face_anchor",
                },
                indent=2,
            )
        )
        return
    threshold = float(thresholds[best_index])
    calibrated = math.isfinite(threshold)
    recognized = prediction_is_accepted(float(distances[best_index]), threshold)
    result = {
        "image": str(image_path),
        "tracking_status": sample.tracking_status,
        "torso_anchor_source": sample.torso_anchor_source,
        "face_anchor_source": sample.face_anchor_source,
        "detected_hands": sample.detected_hands,
        "prediction": classes[best_index] if recognized else "unknown",
        "nearest_label": classes[best_index],
        "distance": float(distances[best_index]),
        "rejection_threshold": threshold if calibrated else None,
        "rejection_threshold_source": threshold_sources[best_index],
        "rejection_reason": (
            None
            if recognized
            else "distance_exceeds_validation_threshold"
            if calibrated
            else "no_reliable_validation_threshold"
        ),
        "warning": "Static-pose experiment only; dynamic sign meaning is not evaluated.",
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="extract features, create the CSV, and make review sheets",
    )
    prepare_parser.add_argument(
        "--profile",
        choices=tuple(LABEL_PROFILES),
        default="default",
        help="Vocabulary/artifact profile to prepare",
    )
    train_parser = subparsers.add_parser(
        "train",
        help="fit/evaluate grouped nearest-medoid prototypes",
    )
    train_parser.add_argument(
        "--profile",
        choices=tuple(LABEL_PROFILES),
        default="default",
        help="Vocabulary/artifact profile to train",
    )
    predict_parser = subparsers.add_parser("predict-image", help="match a still image to the prototype")
    predict_parser.add_argument("image", type=Path)
    predict_parser.add_argument("--model", type=Path, default=MODEL_PATH)
    arguments = parser.parse_args()

    try:
        if arguments.command == "prepare":
            prepare_dataset(arguments.profile)
        elif arguments.command == "train":
            train_prototype(arguments.profile)
        else:
            predict_image(arguments.image, arguments.model)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
