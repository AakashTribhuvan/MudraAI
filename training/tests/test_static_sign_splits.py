from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from static_sign_prototype import grouped_splits


class StaticSignSplitTests(unittest.TestCase):
    def test_five_source_groups_split_four_to_one(self) -> None:
        records = [
            {"label": "HELLO", "inferred_source_group": f"source-{index}"}
            for index in range(5)
        ]

        splits = grouped_splits(records)

        self.assertEqual(list(splits.values()).count("train"), 4)
        self.assertEqual(list(splits.values()).count("test"), 1)

    def test_all_augmentations_from_a_source_stay_in_one_split(self) -> None:
        records = [
            {
                "label": "HELLO",
                "source_group": "source-a",
                "inferred_source_group": "source-a",
                "augmentation": augmentation,
            }
            for augmentation in ("original", "mirrored")
        ] + [
            {
                "label": "HELLO",
                "source_group": f"source-{index}",
                "inferred_source_group": f"source-{index}",
                "augmentation": "original",
            }
            for index in range(4)
        ]

        splits = grouped_splits(records)

        self.assertEqual(len(splits), 5)
        self.assertIn(splits["HELLO:source-a"], {"train", "test"})
        self.assertEqual(
            grouped_splits(records),
            splits,
        )

    def test_single_source_group_is_not_misrepresented_as_test_data(self) -> None:
        records = [{"label": "HELLO", "inferred_source_group": "one-recording"}]

        self.assertEqual(
            grouped_splits(records),
            {"HELLO:one-recording": "train"},
        )

    def test_test_split_targets_one_fifth_of_images_without_splitting_groups(self) -> None:
        records = [
            {"label": "HELLO", "source_group": "large", "sample_id": f"large-{index}"}
            for index in range(2)
        ] + [
            {"label": "HELLO", "source_group": group, "sample_id": group}
            for group in ("small-a", "small-b", "small-c")
        ]

        splits = grouped_splits(records)

        test_groups = {group for group, split in splits.items() if split == "test"}
        self.assertEqual(sum(splits[group] == "test" for group in splits), 1)
        self.assertEqual(len(test_groups), 1)
        self.assertGreaterEqual(len({group for group, split in splits.items() if split == "train"}), 1)


if __name__ == "__main__":
    unittest.main()
