"""Join extracted common-word landmarks to their source metadata and make signer-aware splits.

Supplementary clips without verified signer IDs are kept in training only. ISL500
samples are split by signer, allowing a genuine signer-held-out validation/test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from mudra_ml.config import project_path
from mudra_ml.data.manifest import read_manifest, write_manifest
from mudra_ml.data.splits import check_no_leakage, split_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, default=Path("data/metadata/manifest_common_word_videos_v1.csv"))
    parser.add_argument("--source-manifest", type=Path, default=Path("data/external/isl_common_word_videos_v1/manifest.csv"))
    parser.add_argument("--name", default="common_word_videos")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extracted = read_manifest(project_path(args.extracted))
    with project_path(args.source_manifest).open(newline="", encoding="utf-8-sig") as f:
        source_rows = list(csv.DictReader(f))
    source_root = project_path(args.source_manifest).parent
    source_by_path = {
        str((source_root / Path(row["local_path"])).resolve()).casefold(): row
        for row in source_rows
    }

    joined = []
    missing_provenance = []
    for row in extracted:
        source = source_by_path.get(str(Path(row.raw_path).resolve()).casefold())
        if source is None:
            missing_provenance.append(row.raw_path)
            continue
        row.source = source["dataset"]
        row.split = ""
        if source["dataset"] == "ISL500" and source["signer"].startswith("User"):
            row.signer_id = f"ISL500:{source['signer']}"
            row.group_id = f"ISL500:signer:{source['signer']}"
        else:
            # INCLUDE's aggregate field is a clip ID; CISLR's is a semantic category,
            # neither identifies a signer. Keep these clips training-only.
            row.signer_id = ""
            row.group_id = f"{source['dataset']}:clip:{row.sample_id}"
            row.split = "train"
        joined.append(row)

    if missing_provenance:
        raise SystemExit(f"Missing source metadata for {len(missing_provenance)} extracted clips")
    if not joined:
        raise SystemExit("No extracted clips joined to source metadata")

    # Require the full 15-signer ISL500 cohort for every evaluated label.
    signer_counts: dict[str, set[str]] = {}
    for row in joined:
        if row.source == "ISL500" and row.signer_id:
            signer_counts.setdefault(row.label, set()).add(row.signer_id)
    eligible = {label for label, signers in signer_counts.items() if len(signers) >= 15}
    joined = [r for r in joined if r.label in eligible]
    if not eligible:
        raise SystemExit("No labels have 15 usable ISL500 signers after extraction")

    # For 15 signers, use exactly 10/2/3 for train/validation/test instead of
    # relying on a greedy row-balancer that can leave only two people in test.
    signer_ids = sorted({r.signer_id for r in joined if r.source == "ISL500" and r.signer_id})
    random.Random(args.seed).shuffle(signer_ids)
    signer_split = {
        **{s: "train" for s in signer_ids[:10]},
        **{s: "val" for s in signer_ids[10:12]},
        **{s: "test" for s in signer_ids[12:15]},
    }
    for row in joined:
        if row.source == "ISL500":
            row.split = signer_split[row.signer_id]
        else:
            row.split = "train"
    # Keep flagged clips in held-out test so the report reflects difficult videos;
    # exclude them from training and calibration to avoid teaching on tracking failures.
    usable = [r for r in joined if not r.quality_flag or r.split == "test"]
    check_no_leakage(usable)
    report = split_report(usable)
    report["evaluation_protocol"] = "ISL500 signer-disjoint; supplementary sources train-only"
    report["labels"] = sorted(eligible)
    report["source_counts"] = dict(Counter(r.source for r in usable))
    report["excluded_flagged_train_or_val"] = len(joined) - len(usable)

    out_dir = project_path("data/metadata/splits") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(usable, out_dir / "all.csv")
    for split in ("train", "val", "test"):
        write_manifest([r for r in usable if r.split == split], out_dir / f"{split}.csv")
    (out_dir / "classes.json").write_text(json.dumps(sorted(eligible), indent=2), encoding="utf-8")
    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Splits: {out_dir}")


if __name__ == "__main__":
    main()
