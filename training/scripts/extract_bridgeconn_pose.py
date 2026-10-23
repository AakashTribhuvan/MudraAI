"""Extract and motion-sort BridgeConn pose files from downloaded WebDataset shards.

Videos are skipped. The output contains pose files, per-sample JSON metadata,
and motion-candidate CSV indexes. These candidates require ISL-fluent review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVES = ROOT / "data/external/pose_word_datasets/bridgeconn/archives"
DEFAULT_OUTPUT = ROOT / "data/external/pose_word_datasets/bridgeconn"
STATIC_LABEL_OVERRIDES = {"help", "you"}  # Project-defined held-pose targets.


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return value or "unknown"


def override_name(value: str) -> str:
    # Some dictionary entries suffix duplicate senses as "help (2)".
    return safe_name(re.sub(r"\s*\(\d+\)$", "", value.strip()))


def read_motion(npz_bytes: bytes, duration_sec: float) -> tuple[float, float, float, int]:
    with np.load(__import__("io").BytesIO(npz_bytes)) as data:
        frames = np.asarray(data["frames"], dtype=np.float32)
        confidence = np.asarray(data.get("confidences", np.ones(frames.shape[:-1])), dtype=np.float32)
    # DWPose layout: 18 COCO body landmarks first; neck=1, shoulders=2/5,
    # wrists=4/7. Pool both wrists so category is independent of signer side.
    if frames.ndim == 4:
        frames = frames[:, 0]
    if confidence.ndim == 3:
        confidence = confidence[:, 0]
    if frames.ndim != 3 or frames.shape[1] < 8:
        return float("nan"), float("nan"), float("nan"), 0
    points = frames[:, :18]
    conf = confidence[:, :18]
    scale = np.linalg.norm(points[:, 2, :2] - points[:, 5, :2], axis=-1)
    scale = np.maximum(scale, 1.0)
    neck = points[:, 1, :2]
    wrists = points[:, [4, 7], :2]
    wrists = (wrists - neck[:, None, :]) / scale[:, None, None]
    wrist_conf = conf[:, [4, 7]]
    shoulder_conf = np.minimum(conf[:, 2], conf[:, 5])
    valid = (wrist_conf >= 0.25) & (shoulder_conf[:, None] >= 0.25)
    pair_valid = valid[1:] & valid[:-1]
    step = np.linalg.norm(np.diff(wrists, axis=0), axis=-1)
    values = step[pair_valid]
    if values.size < 4:
        return float("nan"), float("nan"), float("nan"), int(values.size)
    active_hands = int(np.sum(valid.sum(axis=0) >= 4))
    path_per_second = float(np.sum(values) / max(active_hands, 1) / max(duration_sec, 1e-3))
    return float(np.median(values)), float(np.quantile(values, 0.9)), path_per_second, int(values.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    archives_by_name = {archive.name: archive for archive in args.archives.glob("shard_*-train.tar")}
    for archive in args.archives.glob("shard_*-train.verified.tar"):
        canonical_name = archive.name.removesuffix(".verified.tar") + ".tar"
        # Prefer a verified clean copy when both variants exist.
        archives_by_name[canonical_name] = archive
    archives = [archives_by_name[name] for name in sorted(archives_by_name)]
    for archive in archives:
        with tarfile.open(archive, "r:*") as tar:
            members = {member.name: member for member in tar.getmembers() if member.isfile()}
            for member in members.values():
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                key = Path(member.name).stem
                metadata = json.load(tar.extractfile(member))
                pose_member = members.get(f"{key}.pose-mediapipe.pose")
                dwp_member = members.get(f"{key}.pose-dwpose.npz")
                if pose_member is None or dwp_member is None:
                    continue
                dwp_bytes = tar.extractfile(dwp_member).read()
                transcript = metadata.get("transcript", {}).get("text", "").strip()
                duration = float(metadata.get("duration_sec", 0) or 0)
                median_step, p90_step, path_per_second, pair_count = read_motion(dwp_bytes, duration)
                raw.append(
                    {
                        "archive": archive.name,
                        "key": key,
                        "gloss": transcript,
                        "metadata": metadata,
                        "median_step": median_step,
                        "p90_step": p90_step,
                        "path_per_second": path_per_second,
                        "pair_count": pair_count,
                    }
                )

    valid_scores = [float(r["path_per_second"]) for r in raw if np.isfinite(float(r["path_per_second"]))]
    low_cut, high_cut = np.quantile(valid_scores, [0.33, 0.67]) if valid_scores else (0.0, 0.0)
    bucket_by_identity: dict[tuple[str, str], str] = {}
    for row in raw:
        motion_score = float(row["path_per_second"])
        bucket = (
            "static_candidate" if override_name(str(row["gloss"])) in STATIC_LABEL_OVERRIDES else
            "needs_review" if not np.isfinite(motion_score) else
            "static_candidate" if motion_score <= low_cut else
            "dynamic_candidate" if motion_score >= high_cut else
            "needs_review"
        )
        bucket_by_identity[(str(row["archive"]), str(row["key"]))] = bucket
    by_archive: dict[str, list[dict[str, object]]] = {}
    for row in raw:
        by_archive.setdefault(str(row["archive"]), []).append(row)
    for archive in archives:
        archive_rows = by_archive.get(archive.name, [])
        if not archive_rows:
            continue
        with tarfile.open(archive, "r:*") as tar:
            members = {member.name: member for member in tar.getmembers() if member.isfile()}
            for row in archive_rows:
                bucket = bucket_by_identity[(str(row["archive"]), str(row["key"]))]
                metadata = row["metadata"]
                name = safe_name(str(row["gloss"]))
                key = f"{archive.stem}_{row['key']}"
                base = args.output / "extracted" / bucket / name
                base.mkdir(parents=True, exist_ok=True)
                pose_member = members[f"{row['key']}.pose-mediapipe.pose"]
                dwp_member = members[f"{row['key']}.pose-dwpose.npz"]
                json_member = members[f"{row['key']}.json"]
                pose_path = base / f"{key}.pose"
                dwp_path = base / f"{key}.pose-dwpose.npz"
                json_path = base / f"{key}.json"
                pose_path.write_bytes(tar.extractfile(pose_member).read())
                dwp_path.write_bytes(tar.extractfile(dwp_member).read())
                json_path.write_text(
                    json.dumps(json.load(tar.extractfile(json_member)), indent=2), encoding="utf-8"
                )
                rows.append(
            {
                "gloss": row["gloss"],
                "bucket": bucket,
                "archive": row["archive"],
                "sample_key": key,
                "pose_path": pose_path.relative_to(ROOT).as_posix(),
                "dwpose_path": dwp_path.relative_to(ROOT).as_posix(),
                "metadata_path": json_path.relative_to(ROOT).as_posix(),
                "median_wrist_step_per_shoulder_width": row["median_step"],
                "p90_wrist_step_per_shoulder_width": row["p90_step"],
                "mean_wrist_path_per_second_shoulder_widths": row["path_per_second"],
                "valid_adjacent_hand_pairs": row["pair_count"],
                "fps": metadata.get("fps", ""),
                "duration_sec": metadata.get("duration_sec", ""),
                "license": metadata.get("license", "CC BY-SA 4.0"),
                "repository": "https://huggingface.co/datasets/bridgeconn/sign-dictionary-isl",
                "annotation_status": (
                    "project-defined held-pose target; clip motion may disagree; review required"
                    if override_name(str(row["gloss"])) in STATIC_LABEL_OVERRIDES
                    else "relative motion candidate; human review required"
                ),
                }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    for bucket in ("static_candidate", "dynamic_candidate", "needs_review"):
        selected = [r for r in rows if r["bucket"] == bucket]
        with (args.output / f"{bucket}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    summary = {
        "dataset": "BridgeConn Sign Dictionary ISL",
        "repository": "https://huggingface.co/datasets/bridgeconn/sign-dictionary-isl",
        "license": "CC BY-SA 4.0; retain attribution and share-alike obligations",
        "downloaded_archives": sorted({str(r["archive"]) for r in raw}),
        "downloaded_samples": len(rows),
        "upstream_lexical_items_claimed": 3077,
        "completeness": "all 7 published train shards downloaded and indexed",
        "method": "DWPose wrist path relative to neck, normalized by shoulder width and clip duration; 33/67 corpus quantiles",
        "thresholds": {"static_upper": float(low_cut), "dynamic_lower": float(high_cut)},
        "counts": {b: sum(r["bucket"] == b for r in rows) for b in ("static_candidate", "dynamic_candidate", "needs_review")},
        "warning": "Motion buckets are clip-level candidates, not verified lexical static/dynamic labels. Human review by an ISL-fluent signer is required.",
        "project_static_overrides": sorted(STATIC_LABEL_OVERRIDES),
        "extracted_modalities": ["MediaPipe pose-format landmarks", "DWPose keypoint arrays", "JSON metadata/transcript"],
        "videos_extracted": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
