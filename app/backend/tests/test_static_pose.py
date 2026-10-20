from __future__ import annotations

from pathlib import Path

import numpy as np
from mudra_ml.streaming import Frame

from mudra_api.static_pose import FEATURE_DIM, StaticPoseModel, _feature_from_frame

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mudra_intro_static_pose_medoids_v1.npz"


def make_frame() -> Frame:
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[0, :3] = (0.5, 0.15, 0.0)
    pose[2, :3] = (0.46, 0.16, 0.0)
    pose[5, :3] = (0.54, 0.16, 0.0)
    pose[7, :3] = (0.44, 0.15, 0.0)
    pose[8, :3] = (0.56, 0.15, 0.0)
    pose[11, :3] = (0.34, 0.3, 0.0)
    pose[12, :3] = (0.66, 0.3, 0.0)
    pose[13, :3] = (0.3, 0.42, 0.0)
    pose[14, :3] = (0.7, 0.42, 0.0)
    pose[15, :3] = (0.28, 0.54, 0.0)
    pose[16, :3] = (0.72, 0.54, 0.0)
    pose[23, :3] = (0.4, 0.68, 0.0)
    pose[24, :3] = (0.6, 0.68, 0.0)

    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = (0.5, 0.56, 0.0)
    hand[5] = (0.45, 0.5, 0.0)
    hand[17] = (0.55, 0.5, 0.0)
    hand[8] = (0.44, 0.39, 0.0)
    hand[12] = (0.5, 0.38, 0.0)
    hand[16] = (0.54, 0.4, 0.0)
    hand[20] = (0.58, 0.42, 0.0)
    for index in range(21):
        if not hand[index].any():
            hand[index] = (0.48 + (index % 4) * 0.01, 0.48 - (index // 4) * 0.01, 0.0)
    hands = np.zeros((2, 21, 3), dtype=np.float32)
    hands[0] = hand
    return Frame(0.0, hands, np.asarray((True, False)), pose, True)


def test_intro_model_has_separate_live_threshold_gate():
    model = StaticPoseModel(MODEL_PATH)
    assert len(set(model.classes)) == 32
    assert len(model.live_labels) == 10
    assert {"HELLO_HI", "GOOD"}.isdisjoint(model.live_labels)

    eligible_index = next(
        index for index, label in enumerate(model.classes)
        if label == "FROM" and np.isfinite(model.thresholds[index])
    )
    result = model.predict_feature(
        model.prototypes[eligible_index],
        int(model.hand_counts[eligible_index]),
    )
    assert result["prediction"] == "FROM"


def test_intro_model_rejects_preview_only_prototypes():
    model = StaticPoseModel(MODEL_PATH)
    index = model.classes.index("HELLO_HI")
    result = model.predict_feature(model.prototypes[index], int(model.hand_counts[index]))
    assert result["prediction"] is None


def test_static_feature_extraction_matches_model_feature_shape():
    extracted = _feature_from_frame(make_frame(), 640, 480)
    assert extracted is not None
    feature, hand_count = extracted
    assert feature.shape == (FEATURE_DIM,)
    assert np.isfinite(feature).all()
    assert hand_count == 1
