"""Assemble a local-only, signer-indexed set of exact ISL500 word clips."""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = ROOT / "data" / "external" / "signline_intro" / "videos"
SENTENCE_ROOT = ROOT / "data" / "external" / "isl500_sentence_subset" / "videos"
COMMON_MANIFEST = ROOT / "data" / "external" / "isl_common_word_videos_v1" / "manifest.csv"
COMMON_ROOT = ROOT / "data" / "external" / "isl_common_word_videos_v1"
EXTRA_MANIFEST = ROOT / "data" / "external" / "signline_intro_terms" / "manifest.csv"
EXTRA_ROOT = ROOT / "data" / "external" / "signline_intro_terms"
COMMON_TRAIN_SPLIT = ROOT / "data" / "metadata" / "splits" / "common_word_videos" / "train.csv"

SENTENCE_LABELS = {
    "FOOD", "HELLO", "HELP", "I", "MYSELF", "NICE", "NO", "STUDENT",
    "THANKYOU", "THIS", "WATER", "WE", "WHAT", "YOU",
}
COMMON_LABELS = {"DRINK", "EAT", "FRIEND", "GO", "PLEASE", "SCHOOL", "TODAY", "WHERE", "YES"}
EXTRA_LABELS = {"GOOD", "GOOD_MORNING", "GOOD_AFTERNOON"}


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == source.stat().st_size:
            return
        raise FileExistsError(f"Refusing to replace a different staged clip: {target}")
    try:
        os.link(source, target)
    except OSError:
        # Cross-volume fallback. These files are ignored local training data.
        import shutil
        shutil.copy2(source, target)


def main() -> None:
    records: dict[tuple[str, str, str], dict] = {}
    for source in sorted(SENTENCE_ROOT.glob("User*/*")):
        label = source.name.split("__", 1)[0].upper()
        signer = source.parent.name
        if label not in SENTENCE_LABELS or not source.is_file():
            continue
        records[(label, signer, source.name)] = {"label": label, "signer": signer, "source": "ISL500/ISL-DATA", "raw_path": source}

    if not COMMON_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing common-word manifest: {COMMON_MANIFEST}")
    with COMMON_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            label = row.get("normalized_word", "").replace(" ", "_").upper()
            signer = row.get("signer", "")
            if label not in COMMON_LABELS or not signer.startswith("User") or row.get("dataset") != "ISL500":
                continue
            rel = Path(row.get("local_path", ""))
            source = (COMMON_ROOT / rel).resolve()
            if source.is_file():
                records.setdefault((label, signer, source.name), {"label": label, "signer": signer, "source": "ISL500/ISL-DATA", "raw_path": source})

    if not EXTRA_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing extra terms manifest: {EXTRA_MANIFEST}")
    with EXTRA_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            label = {"goodmorning": "GOOD_MORNING", "goodafternoon": "GOOD_AFTERNOON"}.get(
                row["word"], row["word"].upper()
            )
            signer = row["signer"]
            source = (EXTRA_ROOT / row["local_path"]).resolve()
            if label in EXTRA_LABELS and source.is_file():
                records[(label, signer, source.name)] = {"label": label, "signer": signer, "source": "ISL500/ISL-DATA", "raw_path": source}

    # More accepted training clips from the existing CISLR and INCLUDE collection improve
    # signer/camera diversity. Only its recorded train split is used; ISL500 examples here
    # are deliberately omitted because the held-out split below owns Users 013-015.
    if COMMON_TRAIN_SPLIT.is_file():
        with COMMON_TRAIN_SPLIT.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                source_name = row.get("source", "")
                label = row.get("label", "").upper().replace("THANK_YOU", "THANKYOU")
                if source_name not in {"CISLR", "INCLUDE"} or label not in SENTENCE_LABELS | COMMON_LABELS | EXTRA_LABELS:
                    continue
                source = Path(row.get("raw_path", ""))
                if not source.is_absolute():
                    source = (ROOT / source).resolve()
                if source.is_file():
                    signer = f"Extra_{source_name}"
                    key = (label, signer, source.name)
                    records[key] = {"label": label, "signer": signer, "source": source_name, "raw_path": source}

    expected = {"FOOD", "HELLO", "HELP", "I", "MYSELF", "NICE", "NO", "STUDENT", "THANKYOU", "THIS", "WATER", "WE", "WHAT", "YOU", *COMMON_LABELS, "GOOD", "GOOD_MORNING", "GOOD_AFTERNOON"}
    counts = {label: sum(item["label"] == label and item["signer"].startswith("User") for item in records.values())
              for label in sorted(expected)}
    if set(counts) != expected or any(count != 15 for count in counts.values()):
        raise RuntimeError(f"Need exactly 15 ISL500 signer clips per model class; found {counts}")

    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for (label, signer, _), record in sorted(records.items()):
        source = Path(record["raw_path"])
        name = source.name if label not in {"GOOD_MORNING", "GOOD_AFTERNOON"} and not signer.startswith("Extra_") else f"{label}__{source.name}"
        staged = VIDEO_ROOT / signer / name
        link(source, staged)
        manifest.append({"label": label, "signer_id": signer, "source": record["source"], "raw_path": str(staged)})
    out = VIDEO_ROOT.parent / "source_manifest.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Staged {len(manifest)} clips across {len(expected)} labels and 15 signers: {out}")
    print("\n".join(f"{label}: {count}" for label, count in counts.items()))


if __name__ == "__main__":
    main()
