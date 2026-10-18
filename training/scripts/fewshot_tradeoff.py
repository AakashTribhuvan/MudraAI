"""Validation trade-off of the known-word boundary strictness (known_k) at fixed custom-word barriers.

For each known_k, reports on VALIDATION (half the held-out words taught, k=5) how many known words are still
answered correctly and how many never-taught signs are answered anyway. Used to pick the live strictness
under a product rule fixed in advance: known-word answers may drop by at most --max-known-drop.

    python training/scripts/fewshot_tradeoff.py --model experiments/models/fewshot_scale_cos \
        --split-dir data/metadata/splits/fewshot_scale150 --report experiments/reports/fewshot_scale_cos_known.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_fewshot_scale import SEED, encode, model_decisions, run

from mudra_ml.config import project_path
from mudra_ml.data.manifest import read_manifest
from mudra_ml.fewshot import FewShotConfig, KnownWordBoundaries
from mudra_ml.streaming import ModelBundle, SegmenterConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--report", required=True, help="eval_fewshot_scale.py report with the chosen custom-word barriers")
    ap.add_argument("--max-known-drop", type=float, default=0.03)
    args = ap.parse_args()
    chosen = FewShotConfig(**json.loads(Path(project_path(args.report)).read_text(encoding="utf-8"))["chosen"])
    bundle = ModelBundle(project_path(args.model))
    seg = SegmenterConfig.load(project_path("training/configs/streaming.json"))
    threshold = seg.commit_threshold if seg.commit_threshold is not None else bundle.threshold
    split = project_path(args.split_dir)
    held = json.loads((split / "heldout_words.json").read_text(encoding="utf-8"))
    order = list(np.random.default_rng(SEED).permutation(held))
    novel_rows = read_manifest(split / "novel.csv")
    pool = encode(bundle, [r for r in novel_rows if r.split == "train"])
    val = {"novel": encode(bundle, [r for r in novel_rows if r.split == "val"]), "known": encode(bundle, read_manifest(split / "val.csv"))}
    train = encode(bundle, read_manifest(split / "train.csv"))
    words = order[: len(held) // 2]
    rows = []
    for kk in [None, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        kb = None if kk is None else KnownWordBoundaries(train["labels"], train["embs"], kk)
        dec = {part: model_decisions(bundle, val[part], threshold, kb) for part in ("novel", "known")}
        m = run(bundle, chosen, pool, val, dec, words, 5, 3, seed=0)
        rows.append({"known_k": kk, **m})
        print(f"known_k={kk}: known correct {m['known_correct']:.1%}  made-up {m['made_up']:.1%}  "
              f"taught recognised {m['recognised']:.1%}  mixup {m['mixup']:.1%}  stolen {m['stolen']:.2%}", flush=True)
    base = rows[0]["known_correct"]
    ok = [r for r in rows[1:] if base - r["known_correct"] <= args.max_known_drop]
    pick = min(ok, key=lambda r: r["made_up"]) if ok else None
    print(f"rule: known correct may drop at most {args.max_known_drop:.0%} from {base:.1%} -> "
          f"known_k = {pick['known_k'] if pick else None}")


if __name__ == "__main__":
    main()
