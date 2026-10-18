"""Hold many words out of a split set entirely, to measure custom words at a large vocabulary.

The model is trained on the remaining words only; the held-out words' clips go to ``novel.csv`` (keeping
their train / val / test assignment: train clips are the recordings a user would "teach", val and test
clips are what they later sign). ``eval_fewshot.py`` then adds 25 ... N of them as custom words and
measures how often they are recognised, mixed up with each other, or steal known words.

    python training/scripts/make_fewshot_split.py --source combined_words_v3 --holdout 150 --name fewshot_scale150
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mudra_ml.config import project_path
from mudra_ml.data.manifest import read_manifest, write_manifest

SEED = 7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="combined_words_v3")
    ap.add_argument("--holdout", type=int, default=150)
    ap.add_argument("--name", default="fewshot_scale150")
    ap.add_argument("--min-train", type=int, default=5, help="held-out words need this many train clips to teach from")
    args = ap.parse_args()

    src = project_path("data/metadata/splits") / args.source
    rows = {s: read_manifest(src / f"{s}.csv") for s in ("train", "val", "test")}
    labels = sorted({r.label for rs in rows.values() for r in rs})
    train_counts = {lab: sum(r.label == lab for r in rows["train"]) for lab in labels}
    has_queries = {r.label for r in rows["val"]} & {r.label for r in rows["test"]}
    eligible = [lab for lab in labels if train_counts[lab] >= args.min_train and lab in has_queries]
    rng = np.random.default_rng(SEED)
    held = sorted(rng.choice(eligible, size=min(args.holdout, len(eligible)), replace=False).tolist())
    held_set = set(held)

    out = project_path("data/metadata/splits") / args.name
    out.mkdir(parents=True, exist_ok=True)
    novel = []
    for split, rs in rows.items():
        write_manifest([r for r in rs if r.label not in held_set], out / f"{split}.csv")
        for r in rs:
            if r.label in held_set:
                r.split = split
                novel.append(r)
    write_manifest(novel, out / "novel.csv")
    known = sorted(set(labels) - held_set)
    (out / "classes.json").write_text(json.dumps(known, indent=2), encoding="utf-8")
    (out / "heldout_words.json").write_text(json.dumps(held, indent=2), encoding="utf-8")
    print(f"{len(known)} known words, {len(held)} held out ({len(novel)} clips) -> {out}")


if __name__ == "__main__":
    main()
