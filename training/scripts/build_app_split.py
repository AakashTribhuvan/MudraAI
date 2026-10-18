"""Add the app's recorded words (Teach a sign, data/metadata/manifest_app.csv) to a word split set.

Words with at least --min-takes good recordings join the training data: per word, the last take goes to
test and the one before it to validation (by recording order), the rest to train. Takes recorded by
people listed in --holdout-signers all go to test (the honest "new person" test once teammates record).
Without signer variety the test only measures new takes by the same person in the same session.

    python training/scripts/build_app_split.py --base combined_words_v3 --name combined_words_v4_app
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict

from mudra_ml.config import project_path
from mudra_ml.data.manifest import read_manifest, write_manifest
from mudra_ml.data.splits import check_no_leakage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="combined_words_v3")
    ap.add_argument("--name", default="combined_words_v4_app")
    ap.add_argument("--min-takes", type=int, default=5)
    ap.add_argument("--holdout-signers", nargs="*", default=[])
    args = ap.parse_args()

    splits = project_path("data/metadata/splits")
    base = {s: read_manifest(splits / args.base / f"{s}.csv") for s in ("train", "val", "test")}
    app = [r for r in read_manifest(project_path("data/metadata/manifest_app.csv")) if not r.quality_flag]
    holdout = {f"app:{s.lower()}" for s in args.holdout_signers}
    by_word = defaultdict(list)
    for r in app:
        by_word[r.label].append(r)
    added = {"train": [], "val": [], "test": []}
    report = {}
    for word, rows in sorted(by_word.items()):
        if len(rows) < args.min_takes:
            continue
        own = [r for r in rows if r.signer_id not in holdout]
        for r in rows:
            r.group_id = f"{r.group_id}:take:{r.sample_id}"  # one sitting: takes are split individually
        assign = {r.sample_id: "train" for r in own}
        if len(own) >= 3:
            assign[own[-1].sample_id], assign[own[-2].sample_id] = "test", "val"
        for r in rows:
            r.split = "test" if r.signer_id in holdout else assign[r.sample_id]
            added[r.split].append(r)
        report[word] = {s: sum(r.split == s for r in rows) for s in ("train", "val", "test")}

    out = splits / args.name
    out.mkdir(parents=True, exist_ok=True)
    every = []
    for s in ("train", "val", "test"):
        rows = base[s] + added[s]
        write_manifest(rows, out / f"{s}.csv")
        every += rows
    check_no_leakage(every)
    classes = sorted({r.label for r in every})
    (out / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    if (splits / args.base / "label_fixes.json").exists():
        shutil.copy(splits / args.base / "label_fixes.json", out / "label_fixes.json")
    (out / "app_words.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    new = sorted(set(report) - {r.label for s in base.values() for r in s})
    print(f"{len(classes)} classes ({len(new)} new from the app: {', '.join(new) or '-'}); app takes added "
          f"{ {s: len(v) for s, v in added.items()} } -> {out}")


if __name__ == "__main__":
    main()
