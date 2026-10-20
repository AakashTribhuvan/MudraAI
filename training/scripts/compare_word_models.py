"""Compare exported word models on one split at the live commit threshold, separately for the app's recorded
takes (Teach a sign: test and validation takes) and for every other test clip.

A new model is only worth deploying if it gets the recorded words right AND does not lose accuracy elsewhere.
--bank also scores "<model>+bank": the model plus the live taught-word bank (mudra_ml.fewshot.CustomWordBank, as
the server builds it: new words, and known words with >= 3 takes), built from the split's TRAIN takes only.

    python training/scripts/compare_word_models.py --models v9 v10b v10c --split combined_words_v5_app_r4
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np
from mudra_ml.config import project_path
from mudra_ml.fewshot import CustomWordBank, FewShotConfig
from mudra_ml.preprocessing.sequence import load_sequence
from mudra_ml.streaming import ModelBundle, SegmenterConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["v9", "v10"])
    ap.add_argument("--split", default="combined_words_v5_app")
    ap.add_argument("--bank", action="store_true")
    args = ap.parse_args()
    thr = SegmenterConfig.load(project_path("training/configs/streaming.json")).commit_threshold
    bundles = {v: ModelBundle(project_path(f"app/backend/models/isl_mudra_combined_{v}_bilstm")) for v in args.models}
    banks = {}
    if args.bank:  # the server's rule (recordings.py rebuild_bank): new words always, known words from 3 takes
        train = {}
        for r in csv.DictReader(open(project_path(f"data/metadata/splits/{args.split}/train.csv"), encoding="utf-8")):
            if r["source"] == "app":
                train[r["sample_id"].split("#")[0]] = r
        cfg = FewShotConfig.load(project_path("training/configs/fewshot.json"))
        for v, b in list(bundles.items()):
            ex: dict[str, list] = defaultdict(list)
            for r in train.values():
                seq = load_sequence(project_path(r["landmarks_path"]))
                ex[r["label"]].append(b.probs_and_embeddings([seq], int(seq.meta["width"]), int(seq.meta["height"]))[1][0])
            bank = CustomWordBank(cfg)
            bank.set_examples({k: np.stack(x) for k, x in ex.items() if k not in b.classes or len(x) >= 3})
            banks[v] = bank
    res: dict[str, Counter] = defaultdict(Counter)
    misses: dict[str, list] = defaultdict(list)
    for part in ("test", "val"):
        for r in csv.DictReader(open(project_path(f"data/metadata/splits/{args.split}/{part}.csv"), encoding="utf-8")):
            app = r["source"] == "app"
            if part == "val" and not app:
                continue
            seq = load_sequence(project_path(r["landmarks_path"]))
            w, h = int(seq.meta.get("width", 0)), int(seq.meta.get("height", 0))
            group = f"recorded {part}" if app else "other test"
            res[group]["clips"] += 1
            for v, b in bundles.items():
                probs, emb = b.probs_and_embeddings([seq], w, h)
                d = b.decide(probs[0], threshold=thr)
                variants = [(v, d)] + ([(f"{v}+bank", banks[v].decide(emb[0], d) or d)] if v in banks else [])
                for name, dd in variants:
                    right = dd["status"] == "ok" and dd["sign"] == r["label"]
                    res[group][f"{name} right"] += right
                    res[group][f"{name} wrong"] += dd["status"] == "ok" and dd["sign"] != r["label"]
                    if app and not right:
                        misses[name].append(f"{r['label']}->{dd['top_k'][0]['sign']}@{dd['top_k'][0]['confidence']:.2f}")
    report = {"split": args.split, "threshold": thr, "groups": {k: dict(v) for k, v in res.items()}, "recorded_misses": misses}
    out = project_path("experiments/reports/word_models_" + "_".join(args.models) + ".json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    names = [n for v in bundles for n in ([v, f"{v}+bank"] if v in banks else [v])]
    for group, c in res.items():
        print(f"{group} ({c['clips']} clips): " + " | ".join(f"{v} {c[v + ' right']} right, {c[v + ' wrong']} wrong" for v in names))
    for v in names:
        print(f"{v} missed recorded takes: {misses[v] or 'none'}")


if __name__ == "__main__":
    main()
