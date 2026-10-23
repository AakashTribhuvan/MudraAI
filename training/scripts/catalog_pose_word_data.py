"""Create static/dynamic candidate indexes for extracted ISL word landmarks.

The split is a motion heuristic over MediaPipe landmark sequences, not a
linguistic annotation: a sign variant or a noisy clip can land in review.
"""

from __future__ import annotations

import argparse
import csv
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))

from mudra_ml.preprocessing.sequence import load_sequence  # noqa: E402

STATIC_LABEL_OVERRIDES = {"help", "you"}  # Project-defined held-pose targets.


def score(path: Path) -> tuple[float, float, float, int]:
    seq = load_sequence(path)
    # Shoulder width makes the motion score less sensitive to camera distance.
    shoulders = seq.pose[:, [11, 12], :2]
    shoulder_width = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    scale = np.maximum(shoulder_width, 0.05)
    wrists = seq.hands[:, :, 0, :2]
    present = seq.hand_present
    shoulder_midpoint = shoulders.mean(axis=1)
    relative = (wrists - shoulder_midpoint[:, None, :]) / scale[:, None, None]
    step = np.linalg.norm(np.diff(relative, axis=0), axis=-1)
    valid_pair = present[1:] & present[:-1]
    values = step[valid_pair]
    if values.size < 4:
        return float("nan"), float("nan"), float("nan"), int(values.size)
    active_hands = int(np.sum(valid_pair.sum(axis=0) >= 4))
    duration = max(float(seq.timestamps_ms[-1] - seq.timestamps_ms[0]) / 1000.0, 1e-3)
    path_per_second = float(values.sum() / max(active_hands, 1) / duration)
    # Median frame motion discounts single-frame tracking jumps; p90 catches
    # deliberate movement that the median alone could hide.
    return float(np.median(values)), float(np.quantile(values, 0.9)), path_per_second, int(values.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--landmarks",
        type=Path,
        default=ROOT / "data/processed/landmarks/common_word_videos_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/external/pose_word_datasets/indexes",
    )
    args = parser.parse_args()
    records: list[dict[str, object]] = []
    manifest_path = ROOT / "data/metadata/manifest_common_word_videos_v1.csv"
    source_path = ROOT / "data/external/isl_common_word_videos_v1/manifest.csv"
    provenance: dict[str, dict[str, str]] = {}
    if manifest_path.exists() and source_path.exists():
        with manifest_path.open(newline="", encoding="utf-8-sig") as f:
            indexed = {row["sample_id"]: row for row in csv.DictReader(f)}
        with source_path.open(newline="", encoding="utf-8-sig") as f:
            by_name = {Path(row["local_path"]).name: row for row in csv.DictReader(f)}
        for sample_id, row in indexed.items():
            source = by_name.get(Path(row["raw_path"]).name, {})
            provenance[sample_id] = {
                "source_dataset": source.get("dataset", ""),
                "license": source.get("license", ""),
                "repository": source.get("repository", ""),
                "split": row.get("split", ""),
            }
    for path in sorted(args.landmarks.glob("*/*.npz")):
        median_step, p90_step, path_per_second, valid_pairs = score(path)
        sample_provenance = provenance.get(path.stem, {})
        records.append(
            {
                "label": path.parent.name.lower(),
                "sample_id": path.stem,
                "landmarks_path": path.relative_to(ROOT).as_posix(),
                **sample_provenance,
                "median_wrist_step_per_shoulder_width": median_step,
                "p90_wrist_step_per_shoulder_width": p90_step,
                "mean_wrist_path_per_second_shoulder_widths": path_per_second,
                "valid_adjacent_hand_pairs": valid_pairs,
            }
        )

    label_scores: dict[str, float] = {}
    for label in sorted({str(row["label"]) for row in records}):
        values = [
            float(row["mean_wrist_path_per_second_shoulder_widths"])
            for row in records
            if row["label"] == label and np.isfinite(float(row["mean_wrist_path_per_second_shoulder_widths"]))
        ]
        if values:
            label_scores[label] = float(np.median(values))
    low_cut, high_cut = np.quantile(list(label_scores.values()), [0.33, 0.67])
    label_buckets = {
        label: (
            "static_candidate" if label in STATIC_LABEL_OVERRIDES else
            "static_candidate" if value <= low_cut else
            "dynamic_candidate" if value >= high_cut else
            "needs_review"
        )
        for label, value in label_scores.items()
    }
    for row in records:
        row["bucket"] = label_buckets.get(str(row["label"]), "needs_review")
        row["label_median_path_speed"] = label_scores.get(str(row["label"]), float("nan"))
        row["annotation_status"] = (
            "project-defined held-pose target; clip motion may disagree; review required"
            if str(row["label"]) in STATIC_LABEL_OVERRIDES
            else "relative motion candidate; human review required"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    for bucket in ("static_candidate", "dynamic_candidate", "needs_review"):
        selected = [row for row in records if row["bucket"] == bucket]
        with (args.output / f"{bucket}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    summary: dict[str, object] = {
        "source": "data/processed/landmarks/common_word_videos_v1",
        "method": "hand wrist path relative to shoulder midpoint, normalized by shoulder width and clip duration",
        "thresholds": {
            "static_candidate": f"label median path speed at or below corpus 33rd percentile ({low_cut:.6f})",
            "dynamic_candidate": f"label median path speed at or above corpus 67th percentile ({high_cut:.6f})",
            "otherwise": "needs_review",
        },
        "warning": "These are relative-motion candidates, not validated ISL lexical categories. A low-motion recording can still be a dynamic sign captured incompletely; review sign variants with an ISL-fluent signer.",
        "project_static_overrides": sorted(STATIC_LABEL_OVERRIDES),
        "counts": {
            bucket: sum(row["bucket"] == bucket for row in records)
            for bucket in ("static_candidate", "dynamic_candidate", "needs_review")
        },
        "labels": sorted({str(row["label"]) for row in records}),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
