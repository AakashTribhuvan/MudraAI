import unittest
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from static_sign_prototype import (
    FACE_ANCHOR_FLAG_INDICES,
    FEATURE_DIM,
    FEATURE_SCHEMA,
    HAND_FACE_ALIGNMENT_OFFSET,
    HAND_PRESENCE_OFFSET,
    INTRO_LABELS,
    LABEL_PROFILES,
    LABELS,
    StaticPoseRecognizer,
    StaticPredictionStabilizer,
    build_pose_feature,
    extract_image,
    filename_group,
    grouped_splits,
    mirrored_hand_count_mismatch_reason,
    prediction_is_accepted,
    preserve_reviewed_fields,
    _hand_arrays_from_landmarks,
    _tracking_result,
)


class StaticSignPrototypeTests(unittest.TestCase):
    def test_intro_profile_has_separate_training_artifacts_and_added_vocabulary(self):
        self.assertTrue(set(LABELS).issubset(INTRO_LABELS))
        self.assertTrue({"HELLO_HI", "GOOD", "FROM", "NAME"}.issubset(INTRO_LABELS))
        self.assertNotEqual(LABEL_PROFILES["default"][1], LABEL_PROFILES["intro"][1])
        self.assertNotEqual(LABEL_PROFILES["default"][4], LABEL_PROFILES["intro"][4])

    def test_mirror_augmentation_with_changed_hand_count_is_rejected(self):
        self.assertEqual(
            mirrored_hand_count_mismatch_reason(1, 2),
            "mirror_hand_count_mismatch:original_1_mirrored_2",
        )
        self.assertEqual(mirrored_hand_count_mismatch_reason(2, 2), None)

    def test_filename_number_suffixes_share_a_conservative_group(self):
        self.assertEqual(filename_group("YOU (3).jpg"), "you")
        self.assertEqual(filename_group("YOU-2 (3).jpg"), "you-2")
        self.assertEqual(filename_group("ME-333.jpg"), "me-333")

    def test_grouped_split_never_separates_a_filename_family(self):
        records = [
            {"label": "A", "inferred_source_group": "a"},
            {"label": "A", "inferred_source_group": "a"},
            {"label": "A", "inferred_source_group": "b"},
            {"label": "A", "inferred_source_group": "c"},
            {"label": "A", "inferred_source_group": "d"},
            {"label": "B", "inferred_source_group": "one"},
            {"label": "B", "inferred_source_group": "two"},
        ]
        splits = grouped_splits(records)

        self.assertEqual(len({splits[f'A:{group}'] for group in ("a", "b", "c", "d")}), 3)
        self.assertEqual(len({splits[f'B:{group}'] for group in ("one", "two")}), 2)
        self.assertEqual(grouped_splits(records), splits)

    def test_mirrored_augmentations_share_the_source_split(self):
        records = [
            {"label": "YOU", "source_group": "frame-family", "augmentation": variant}
            for variant in ("original", "mirrored")
        ] + [
            {"label": "YOU", "source_group": f"family-{index}", "augmentation": "original"}
            for index in range(1, 6)
        ]
        splits = grouped_splits(records)

        self.assertEqual(len(splits), 6)
        self.assertIn("YOU:frame-family", splits)
        self.assertEqual(
            LABELS,
            (
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
            ),
        )

    def test_reviewed_source_group_can_override_filename_guess(self):
        records = [
            {
                "label": "A",
                "inferred_source_group": "frame-a",
                "source_group": "same-source-clip",
            },
            {
                "label": "A",
                "inferred_source_group": "frame-b",
                "source_group": "same-source-clip",
            },
            {"label": "A", "inferred_source_group": "other", "source_group": "other"},
            {"label": "A", "inferred_source_group": "last", "source_group": "last"},
        ]
        splits = grouped_splits(records)

        self.assertIn("A:same-source-clip", splits)
        self.assertNotIn("A:frame-a", splits)
        self.assertNotIn("A:frame-b", splits)

    def test_prepare_preserves_manual_review_metadata_but_refreshes_computed_fields(self):
        record = {
            "label": "I_ME_MINE_MY",
            "verified_gloss": "",
            "review_status": "needs_review",
            "source_group": "i",
            "tracking_status": "accepted",
        }
        previous = {
            "label": "I",
            "verified_gloss": "first-person singular",
            "review_status": "approved",
            "source_group": "clip-17",
            "tracking_status": "rejected",
        }

        preserve_reviewed_fields(record, previous)

        self.assertEqual(record["label"], "I")
        self.assertEqual(record["verified_gloss"], "first-person singular")
        self.assertEqual(record["review_status"], "approved")
        self.assertEqual(record["source_group"], "clip-17")
        self.assertEqual(record["tracking_status"], "accepted")

    def test_feature_is_invariant_to_image_translation_and_global_scale(self):
        pose_xyz = np.zeros((33, 3), dtype=np.float32)
        pose_visibility = np.ones(33, dtype=np.float32)
        points = {
            11: (100, 100, 0),
            12: (200, 100, 0),
            13: (90, 160, 0),
            14: (210, 160, 0),
            15: (80, 220, 0),
            16: (220, 220, 0),
            23: (110, 260, 0),
            24: (190, 260, 0),
        }
        for index, point in points.items():
            pose_xyz[index] = point
        hand = np.asarray(
            [(70 + index, 215 + index * 2, 0) for index in range(21)],
            dtype=np.float32,
        )
        hands = {"left": hand}
        original, _, _, _ = build_pose_feature(pose_xyz, pose_visibility, hands)

        transformed_pose = pose_xyz * 2 + np.asarray((500, 300, 80), dtype=np.float32)
        transformed_hand = hand * 2 + np.asarray((500, 300, 80), dtype=np.float32)
        transformed, _, _, _ = build_pose_feature(
            transformed_pose,
            pose_visibility,
            {"left": transformed_hand},
        )

        np.testing.assert_allclose(original, transformed, atol=1e-6)

    def test_feature_retains_hand_location_relative_to_torso(self):
        pose_xyz = np.zeros((33, 3), dtype=np.float32)
        visibility = np.ones(33, dtype=np.float32)
        pose_xyz[11] = (100, 100, 0)
        pose_xyz[12] = (200, 100, 0)
        pose_xyz[23] = (110, 260, 0)
        pose_xyz[24] = (190, 260, 0)
        hand = np.asarray(
            [(80 + index, 150 + index, 0) for index in range(21)],
            dtype=np.float32,
        )
        first, _, _, _ = build_pose_feature(
            pose_xyz, visibility, {"left": hand}
        )
        moved_hand = hand + np.asarray((50, 0, 0), dtype=np.float32)
        second, _, _, _ = build_pose_feature(
            pose_xyz, visibility, {"left": moved_hand}
        )

        self.assertGreater(np.linalg.norm(first - second), 0.1)

    def test_face_relative_hand_features_detect_inward_vs_outward_index_direction(self):
        pose_xyz = np.zeros((33, 3), dtype=np.float32)
        visibility = np.zeros(33, dtype=np.float32)
        for index, point in {
            0: (100, 60, 0),
            7: (80, 40, 0),
            8: (120, 40, 0),
            11: (70, 110, 0),
            12: (130, 110, 0),
            23: (80, 260, 0),
            24: (120, 260, 0),
        }.items():
            pose_xyz[index] = point
            visibility[index] = 1.0

        hand_toward_face = np.zeros((21, 3), dtype=np.float32)
        hand_toward_face[0] = (100, 110, 0)
        hand_toward_face[5] = (105, 100, 0)
        hand_toward_face[8] = (100, 75, 0)
        hand_toward_face[17] = (95, 100, 0)
        hand_away_from_face = hand_toward_face.copy()
        hand_away_from_face[8] = (110, 130, 0)

        toward, _, _, _ = build_pose_feature(
            pose_xyz,
            visibility,
            {"left": hand_toward_face},
        )
        away, _, _, _ = build_pose_feature(
            pose_xyz,
            visibility,
            {"left": hand_away_from_face},
        )

        self.assertGreater(toward[32 + HAND_FACE_ALIGNMENT_OFFSET], 0.8)
        self.assertLess(away[32 + HAND_FACE_ALIGNMENT_OFFSET], -0.8)

    def test_face_relative_features_remain_invariant_to_screen_translation_and_scale(self):
        pose_xyz = np.zeros((33, 3), dtype=np.float32)
        visibility = np.zeros(33, dtype=np.float32)
        points = {
            0: (150, 60, 0),
            7: (130, 40, 0),
            8: (170, 40, 0),
            11: (100, 120, 0),
            12: (200, 120, 0),
            23: (120, 280, 0),
            24: (180, 280, 0),
        }
        for index, point in points.items():
            pose_xyz[index] = point
            visibility[index] = 1.0
        hand = np.asarray(
            [(145 + index % 5, 130 + index // 5, 0) for index in range(21)],
            dtype=np.float32,
        )
        hand[5] = (150, 125, 0)
        hand[8] = (150, 100, 0)
        hand[17] = (140, 125, 0)
        first, _, _, _ = build_pose_feature(
            pose_xyz, visibility, {"left": hand}
        )

        offset = np.asarray((500, 250, 80), dtype=np.float32)
        moved_pose = pose_xyz * 1.7 + offset
        moved_hand = hand * 1.7 + offset
        second, _, _, _ = build_pose_feature(
            moved_pose, visibility, {"left": moved_hand}
        )

        np.testing.assert_allclose(first, second, atol=3e-6)

    def test_arm_direction_features_ignore_limb_length_differences(self):
        pose_xyz = np.zeros((33, 3), dtype=np.float32)
        visibility = np.ones(33, dtype=np.float32)
        pose_xyz[11] = (100, 100, 0)
        pose_xyz[12] = (200, 100, 0)
        pose_xyz[13] = (90, 160, 0)
        pose_xyz[14] = (210, 160, 0)
        pose_xyz[15] = (80, 220, 0)
        pose_xyz[16] = (220, 220, 0)
        pose_xyz[23] = (110, 260, 0)
        pose_xyz[24] = (190, 260, 0)
        first, _, _, _ = build_pose_feature(pose_xyz, visibility, {})

        longer_limbs = pose_xyz.copy()
        for shoulder, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
            upper = pose_xyz[elbow] - pose_xyz[shoulder]
            lower = pose_xyz[wrist] - pose_xyz[elbow]
            longer_limbs[elbow] = pose_xyz[shoulder] + upper * 2
            longer_limbs[wrist] = longer_limbs[elbow] + lower * 3
        second, _, _, _ = build_pose_feature(longer_limbs, visibility, {})

        np.testing.assert_allclose(first[:32], second[:32], atol=1e-6)

    def test_tracking_gate_uses_hand_size_not_handedness_confidence(self):
        self.assertEqual(
            _tracking_result(True, 0.9, 1, 100, 35, []),
            ("accepted", "tracking_gate_passed"),
        )
        self.assertEqual(
            _tracking_result(True, 0.9, 1, 20, 35, []),
            ("rejected", "hand_too_small_for_tracking"),
        )
        self.assertEqual(
            _tracking_result(True, 0.9, 1, 40, 35, []),
            ("accepted", "tracking_gate_passed"),
        )

    def test_pose_anchor_quality_and_detected_hand_count_are_reported(self):
        pose_xyz = np.zeros((33, 3), dtype=np.float32)
        pose_xyz[11] = (100, 100, 0)
        pose_xyz[12] = (200, 100, 0)
        pose_xyz[23] = (110, 260, 0)
        pose_xyz[24] = (190, 260, 0)
        visibility = np.ones(33, dtype=np.float32)
        visibility[24] = 0.3
        hand = np.ones((21, 3), dtype=np.float32)
        hand[:, 0] = np.linspace(0, 10, 21)
        hand[:, 1] = np.linspace(0, 20, 21)
        _, anchor_visibility, hand_count, _ = build_pose_feature(
            pose_xyz, visibility, {"hand_0": hand}
        )

        self.assertAlmostEqual(anchor_visibility, 1.0)
        self.assertEqual(hand_count, 1)

    def test_missing_hips_use_a_shoulders_only_torso_estimate(self):
        pose_xyz = np.zeros((33, 3), dtype=np.float32)
        visibility = np.zeros(33, dtype=np.float32)
        pose_xyz[11] = (100, 100, 0)
        pose_xyz[12] = (200, 100, 0)
        pose_xyz[13] = (80, 160, 0)
        pose_xyz[14] = (220, 160, 0)
        pose_xyz[15] = (70, 220, 0)
        pose_xyz[16] = (230, 220, 0)
        visibility[[11, 12, 13, 14, 15, 16]] = 0.9

        feature, anchor_visibility, _, anchor_source = build_pose_feature(
            pose_xyz, visibility, {}
        )

        self.assertEqual(anchor_source, "shoulders_estimated")
        self.assertAlmostEqual(anchor_visibility, 0.9)
        self.assertTrue(np.isfinite(feature).all())

    def test_missing_validation_threshold_must_not_be_treated_as_confident_prediction(self):
        threshold = float("nan")
        self.assertFalse(math.isfinite(threshold))
        self.assertFalse(prediction_is_accepted(0.1, threshold))
        self.assertTrue(prediction_is_accepted(0.1, 0.2))
        self.assertFalse(prediction_is_accepted(0.3, 0.2))

    def test_live_recognizer_returns_only_threshold_accepted_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.npz"
            np.savez_compressed(
                model_path,
                classes=np.asarray(["YOU", "HELP"]),
                prototypes=np.stack((np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM))).astype(np.float32),
                required_hand_counts=np.asarray([1, 2], dtype=np.int32),
                rejection_thresholds=np.asarray([1.0, np.nan], dtype=np.float32),
                threshold_sources=np.asarray(
                    ["held_out_validation_p95", "unavailable"]
                ),
                feature_schema=np.asarray([FEATURE_SCHEMA]),
            )
            recognizer = StaticPoseRecognizer(model_path)

            accepted = recognizer.predict_feature(np.zeros(FEATURE_DIM, dtype=np.float32))
            rejected = recognizer.predict_feature(np.ones(FEATURE_DIM, dtype=np.float32))
            one_hand_pose = recognizer.predict_feature(
                np.ones(FEATURE_DIM, dtype=np.float32),
                detected_hand_count=1,
            )
            two_hand_pose = recognizer.predict_feature(
                np.ones(FEATURE_DIM, dtype=np.float32),
                detected_hand_count=2,
            )

        self.assertEqual(accepted["prediction"], "YOU")
        self.assertEqual(accepted["threshold_source"], "held_out_validation_p95")
        self.assertIsNone(rejected["prediction"])
        self.assertEqual(rejected["reason"], "no_reliable_validation_threshold")
        self.assertIsNone(one_hand_pose["prediction"])
        self.assertEqual(one_hand_pose["nearest_label"], "YOU")
        self.assertEqual(two_hand_pose["nearest_label"], "HELP")
        self.assertEqual(two_hand_pose["required_hand_count"], 2)

    def test_expected_positions_come_from_exact_hand_count_prototype(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.npz"
            single_hand = np.zeros(FEATURE_DIM, dtype=np.float32)
            single_hand[32 : 35] = (-0.2, 0.1, 0.0)
            single_hand[32 + HAND_PRESENCE_OFFSET] = 1.0
            two_hand = np.zeros(FEATURE_DIM, dtype=np.float32)
            two_hand[32 : 35] = (-0.5, 0.2, 0.0)
            two_hand[32 + HAND_PRESENCE_OFFSET] = 1.0
            two_hand[32 + 72 : 32 + 75] = (0.5, 0.4, 0.0)
            two_hand[32 + 72 + HAND_PRESENCE_OFFSET] = 1.0
            np.savez_compressed(
                model_path,
                classes=np.asarray(["HELP", "HELP"]),
                prototypes=np.stack((single_hand, two_hand)),
                required_hand_counts=np.asarray([1, 2], dtype=np.int32),
                rejection_thresholds=np.asarray([10.0, 10.0], dtype=np.float32),
                threshold_sources=np.asarray(["test", "test"]),
                feature_schema=np.asarray([FEATURE_SCHEMA]),
            )
            recognizer = StaticPoseRecognizer(model_path)

        pose_xyz = np.zeros((33, 3), dtype=np.float32)
        visibility = np.ones(33, dtype=np.float32)
        pose_xyz[11] = (100, 100, 0)
        pose_xyz[12] = (200, 100, 0)
        pose_xyz[23] = (100, 300, 0)
        pose_xyz[24] = (200, 300, 0)
        positions = recognizer.expected_hand_positions(
            "HELP",
            2,
            pose_xyz,
            visibility,
        )

        self.assertEqual(len(positions), 2)
        np.testing.assert_allclose(positions, [(100, 220), (200, 240)], atol=1e-5)
        self.assertEqual(recognizer.supported_learning_profiles(), {"HELP": {1, 2}})

    def test_food_prediction_requires_a_face_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.npz"
            np.savez_compressed(
                model_path,
                classes=np.asarray(["FOOD", "YOU"]),
                prototypes=np.stack(
                    (np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM))
                ).astype(np.float32),
                required_hand_counts=np.ones(2, dtype=np.int32),
                rejection_thresholds=np.asarray([100.0, 100.0], dtype=np.float32),
                threshold_sources=np.asarray(["experimental", "experimental"]),
                feature_schema=np.asarray([FEATURE_SCHEMA]),
            )
            recognizer = StaticPoseRecognizer(model_path)
            no_face = recognizer.predict_feature(
                np.zeros(FEATURE_DIM, dtype=np.float32),
                detected_hand_count=1,
            )
            face_feature = np.zeros(FEATURE_DIM, dtype=np.float32)
            face_feature[FACE_ANCHOR_FLAG_INDICES[0]] = 1.0
            with_face = recognizer.predict_feature(
                face_feature,
                detected_hand_count=1,
            )

        self.assertIsNone(no_face["prediction"])
        self.assertEqual(no_face["reason"], "FOOD_requires_face_anchor")
        self.assertEqual(with_face["prediction"], "FOOD")

    def test_live_prediction_is_shown_only_after_a_stable_static_pose(self):
        stabilizer = StaticPredictionStabilizer(hold_s=0.4)

        self.assertEqual(stabilizer.update(1.0, "YOU", active=True), (None, "confirming"))
        self.assertEqual(stabilizer.update(1.2, "YOU", active=True), (None, "confirming"))
        self.assertEqual(stabilizer.update(1.41, "YOU", active=True), ("YOU", "recognized"))
        self.assertEqual(stabilizer.update(1.5, None, active=True), (None, "unknown"))
        self.assertEqual(stabilizer.update(2.0, "WHAT", active=False), (None, "inactive"))

    def test_static_extractor_uses_the_live_selfie_mirror_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "orientation.png"
            image = np.zeros((12, 20, 3), dtype=np.uint8)
            image[:, :10] = (0, 0, 255)
            image[:, 10:] = (255, 0, 0)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            captured = []

            class FakePoseDetector:
                def detect(self, mp_image):
                    captured.append(mp_image.numpy_view().copy())
                    return type("PoseResult", (), {"pose_landmarks": []})()

            class FakeHandDetector:
                def detect(self, mp_image):
                    return type(
                        "HandResult",
                        (),
                        {"hand_landmarks": [], "handedness": []},
                    )()

            extracted = extract_image(
                image_path,
                FakePoseDetector(),
                FakeHandDetector(),
            )
            unmirrored_pixels = []

            class CapturePoseDetector:
                def detect(self, mp_image):
                    unmirrored_pixels.append(mp_image.numpy_view().copy())
                    return type("PoseResult", (), {"pose_landmarks": []})()

            unmirrored = extract_image(
                image_path,
                CapturePoseDetector(),
                FakeHandDetector(),
                mirror=False,
            )

        self.assertEqual(extracted.width, 20)
        self.assertEqual(extracted.height, 12)
        self.assertEqual(captured[0][0, 0].tolist(), [0, 0, 255])
        self.assertEqual(unmirrored.width, 20)
        self.assertEqual(unmirrored_pixels[0][0, 0].tolist(), [255, 0, 0])

    def test_hand_features_are_extracted_from_landmarks_without_handedness(self):
        hand = [
            type(
                "Landmark",
                (),
                {
                    "x": 0.3 + (index % 5) * 0.02,
                    "y": 0.4 + (index // 5) * 0.02,
                    "z": 0.0,
                },
            )()
            for index in range(21)
        ]
        extracted_hands = _hand_arrays_from_landmarks([hand], 100, 100)
        pose = np.zeros((33, 3), dtype=np.float32)
        visibility = np.ones(33, dtype=np.float32)
        pose[11] = (30, 20, 0)
        pose[12] = (70, 20, 0)
        pose[23] = (35, 80, 0)
        pose[24] = (65, 80, 0)
        feature, _, hand_count, _ = build_pose_feature(
            pose, visibility, extracted_hands
        )

        self.assertEqual(list(extracted_hands), ["hand_0"])
        self.assertEqual(hand_count, 1)
        self.assertEqual(feature[32 + HAND_PRESENCE_OFFSET], 1.0)

    def test_two_detections_fill_both_spatial_feature_slots(self):
        hands = [
            [
                type(
                    "Landmark",
                    (),
                    {
                        "x": offset + (index % 5) * 0.02,
                        "y": 0.4 + (index // 5) * 0.02,
                        "z": 0.0,
                    },
                )()
                for index in range(21)
            ]
            for offset in (0.25, 0.65)
        ]
        hands_by_detection = _hand_arrays_from_landmarks(hands, 640, 480)
        pose = np.zeros((33, 3), dtype=np.float32)
        visibility = np.ones(33, dtype=np.float32)
        pose[11] = (200, 100, 0)
        pose[12] = (440, 100, 0)
        pose[23] = (240, 380, 0)
        pose[24] = (400, 380, 0)

        feature, _, hand_count, _ = build_pose_feature(
            pose, visibility, hands_by_detection
        )

        self.assertEqual(hand_count, 2)
        self.assertEqual(
            feature[
                [
                    32 + HAND_PRESENCE_OFFSET,
                    32 + 72 + HAND_PRESENCE_OFFSET,
                ]
            ].tolist(),
            [1.0, 1.0],
        )

    def test_live_hand_count_preserves_two_detections(self):
        pose = [
            type(
                "Landmark",
                (),
                {
                    "x": 0.5,
                    "y": 0.5,
                    "z": 0.0,
                    "visibility": 1.0,
                },
            )()
            for _ in range(33)
        ]
        pose[11].x, pose[11].y = 0.4, 0.35
        pose[12].x, pose[12].y = 0.6, 0.35
        pose[23].x, pose[23].y = 0.43, 0.7
        pose[24].x, pose[24].y = 0.57, 0.7
        hands = [
            [
                type(
                    "Landmark",
                    (),
                    {
                        "x": offset + (index % 5) * 0.04,
                        "y": 0.4 + (index // 5) * 0.04,
                        "z": 0.0,
                    },
                )()
                for index in range(21)
            ]
            for offset in (0.2, 0.6)
        ]
        recognizer = object.__new__(StaticPoseRecognizer)
        with patch.object(
            StaticPoseRecognizer,
            "predict_feature",
            return_value={"prediction": None},
        ) as predict_feature:
            result = recognizer.predict_landmarks(
                pose,
                hands,
                width=640,
                height=480,
            )

        self.assertEqual(result["tracking_status"], "accepted")
        self.assertEqual(predict_feature.call_args.kwargs["detected_hand_count"], 2)


if __name__ == "__main__":
    unittest.main()
