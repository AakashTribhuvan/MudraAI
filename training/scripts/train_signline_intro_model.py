"""Prepare and train the expanded 26-word SignLine introduction model.

Run: app/backend/.venv/Scripts/python.exe training/scripts/train_signline_intro_model.py prepare
     app/backend/.venv/Scripts/python.exe training/scripts/train_signline_intro_model.py train
Clips, tracking arrays, model, and reports remain local under ignored data/ and experiments/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "scripts"))
import train_isl500_word_model as pipeline

pipeline.DATA_ROOT = ROOT / "data" / "external" / "signline_intro"
pipeline.VIDEO_ROOT = pipeline.DATA_ROOT / "videos"
pipeline.TRACK_ROOT = ROOT / "data" / "processed" / "signline_intro" / "landmarks"
pipeline.MANIFEST_PATH = ROOT / "data" / "metadata" / "signline_intro" / "manifest.csv"
pipeline.SPLIT_PATH = ROOT / "data" / "metadata" / "splits" / "signline_intro" / "split.csv"
pipeline.REPORT_ROOT = ROOT / "experiments" / "reports" / "signline_intro"
pipeline.MODEL_ROOT = ROOT / "data" / "processed" / "signline_intro" / "model"
pipeline.WORDS = (
    "DRINK", "EAT", "FOOD", "FRIEND", "GO", "GOOD", "GOOD_AFTERNOON", "GOOD_MORNING",
    "HELLO", "HELP", "I", "MYSELF", "NICE", "NO", "PLEASE", "SCHOOL", "STUDENT",
    "THANKYOU", "THIS", "TODAY", "WATER", "WE", "WHAT", "WHERE", "YES", "YOU",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="extract and save matching body/face/hand landmarks")
    prepare_parser.add_argument("--workers", type=int, default=4)
    prepare_parser.add_argument("--max-fps", type=float, default=15)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument("--include-extra-training", action="store_true",
                                help="include staged CISLR/INCLUDE clips; grouped CV currently favors ISL500-only")
    commands.add_parser("train", help="select and train by signer-grouped validation")
    args = parser.parse_args()
    if args.command == "prepare":
        # The original 14-class ISL500 pilot already has tracker-identical local landmarks.
        # Reuse those arrays by hard link so preparation only tracks genuinely new classes.
        old_tracks = ROOT / "data" / "processed" / "isl500_sentence_subset" / "landmarks"
        for video in pipeline.VIDEO_ROOT.glob("User*/*"):
            label = video.name.split("__", 1)[0].upper()
            if label not in {"FOOD", "HELLO", "HELP", "I", "MYSELF", "NICE", "NO", "STUDENT", "THANKYOU", "THIS", "WATER", "WE", "WHAT", "YOU"}:
                continue
            sample = f"{video.parent.name}_{video.stem}.npz"
            source = old_tracks / label / sample
            target = pipeline.TRACK_ROOT / label / sample
            if source.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    import shutil
                    shutil.copy2(source, target)
        pipeline.prepare(args)
        # The shared training code writes a generic pilot summary; label this run precisely.
        summary_path = pipeline.REPORT_ROOT / "tracking_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update({"model": "SignLine introduction words", "classes": list(pipeline.WORDS)})
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        pipeline.train(args)
        report_path = pipeline.REPORT_ROOT / "model_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["task"] = "SignLine requested introduction + everyday ISL words"
        report["source"] = "390 selected ISL500/ISL-DATA videos across 26 glosses and 15 signers"
        report["words"] = list(pipeline.WORDS)
        report["limitations"] = [
            "The 26 model classes have one clip per class from each of 15 source signers; the locked test contains three clips per class.",
            "Whole-clip hand coverage was below 0.5 on 349 of 390 source clips; the pipeline crops to tracked active spans, but missed hand frames remain a likely accuracy limit.",
            "Held-out clip accuracy is not a live webcam estimate; camera and unseen signer performance may differ.",
            "The exact requested sentence labels ARE, FROM, PUNE, INVENTORS, AND, IS, SIGN, and LINE are not included in this model because the available 15-signer clips do not contain them. STUDENT is the source gloss used for the requested plural Students.",
            "Source glosses and regional ISL variants require review by an ISL-fluent signer.",
            "A video label is an isolated gloss, not an English sentence translation.",
        ]
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Updated report: {report_path}")


if __name__ == "__main__":
    main()
