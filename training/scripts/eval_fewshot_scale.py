"""Custom words at a large vocabulary: mix-ups, made-up answers and stolen known words as N grows.

Needs a model trained without many words (``make_fewshot_split.py``, e.g. 150 of 314 held out). For
N = 25 ... all held-out words, N of them are "taught" from k recorded clips (their train clips). Then:

- taught queries (their val / test clips): recognised / mixed up with another taught word (``mixup``) /
  taken as a known word / missed;
- never-taught queries (held-out words beyond the first N): any answer is made up (``made_up``);
- known-word clips: ``stolen`` when a taught word replaces the model's answer.

The barrier settings (min similarity, model-confidence ceiling, margin, per-word boundary radius_k,
outlier_z, and known_k: the boundary around the trained model's own words, KnownWordBoundaries) are chosen
on validation with half the held-out words taught and k=5, and written to --config-out; test numbers are
then reported for every N and k. Objective on validation:
recognised + known_correct - 3 x (mixup + taken_as_known) - 3 x made_up - 10 x stolen
(known_correct was added with the known-word boundary, which could otherwise reject everything).

    python training/scripts/eval_fewshot_scale.py --model app/backend/models/fewshot_scale_ce \
        --split-dir data/metadata/splits/fewshot_scale150 --out experiments/reports/fewshot_scale_ce.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from mudra_ml.config import project_path
from mudra_ml.data.manifest import read_manifest
from mudra_ml.fewshot import CustomWordBank, FewShotConfig, KnownWordBoundaries, normalise
from mudra_ml.preprocessing.sequence import load_sequence
from mudra_ml.streaming import ModelBundle, SegmenterConfig

SEED = 11


def encode(bundle: ModelBundle, rows) -> dict:
    probs, embs = [], []
    for r in rows:
        s = load_sequence(project_path(r.landmarks_path))
        p, e = bundle.probs_and_embeddings([s], int(s.meta.get("width", 0)), int(s.meta.get("height", 0)))
        probs.append(p[0])
        embs.append(e[0])
    return {"labels": np.array([r.label for r in rows]), "probs": np.stack(probs), "embs": normalise(np.stack(embs))}


def model_decisions(bundle: ModelBundle, part: dict, threshold: float,
                    known: KnownWordBoundaries | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(status ok?, sign, top known-word confidence) per clip, as the live path decides; with ``known``,
    a confident answer outside that word's boundary becomes "not sure"."""
    ok, sign, conf = [], [], []
    for p, e in zip(part["probs"], part["embs"]):
        d = bundle.decide(p, threshold=threshold)
        accepted = d["status"] == "ok" and (known is None or known.inside(d["sign"], e))
        ok.append(accepted)
        sign.append(d["sign"] if accepted else "")
        conf.append(d["top_k"][0]["confidence"] if d["top_k"] else 0.0)
    return np.array(ok), np.array(sign), np.array(conf)


def outcome(bank: CustomWordBank, cfg: FewShotConfig, q: dict, decided) -> dict:
    """Vectorised CustomWordBank.decide over query clips (same rule)."""
    ok, sign, conf = decided
    sims = q["embs"] @ bank.protos.T
    order = np.argsort(-sims, axis=1)
    best = order[:, 0]
    best_sim = sims[np.arange(len(sims)), best]
    second = sims[np.arange(len(sims)), order[:, 1]] if sims.shape[1] > 1 else np.full(len(sims), -1.0)
    custom = (best_sim >= bank.floors[best]) & (best_sim - second >= cfg.margin) & ~(ok & (conf >= cfg.max_model_confidence))
    custom_label = np.array(bank.labels)[best]
    answer = np.where(custom, custom_label, np.where(ok, sign, ""))
    return {"custom": custom, "answer": answer}


def rates(bank, cfg, data, decided) -> dict:
    t = outcome(bank, cfg, data["taught"], decided["taught"])
    taught_labels = data["taught"]["labels"]
    correct = t["answer"] == taught_labels
    mixup = t["custom"] & ~correct
    as_known = ~t["custom"] & (t["answer"] != "")
    out = {"recognised": float(correct.mean()), "mixup": float(mixup.mean()), "taken_as_known": float(as_known.mean()),
           "missed": float((t["answer"] == "").mean())}
    if len(data["untaught"]["labels"]):
        u = outcome(bank, cfg, data["untaught"], decided["untaught"])
        out["made_up"] = float((u["answer"] != "").mean())
        out["made_up_custom"] = float(u["custom"].mean())
    else:  # every held-out word is taught: nothing left to make up
        out["made_up"] = out["made_up_custom"] = None
    k = outcome(bank, cfg, data["known"], decided["known"])
    out["stolen"] = float(k["custom"].mean())
    out["known_correct"] = float((k["answer"] == data["known"]["labels"]).mean())
    return out


def subset(d: dict, mask: np.ndarray) -> dict:
    return {key: v[mask] for key, v in d.items()}


def prepare(pool: dict, split_data: dict, words: list[str], k: int, rng) -> tuple[dict, dict]:
    examples = {}
    for w in words:
        idx = np.flatnonzero(pool["labels"] == w)
        examples[w] = pool["embs"][rng.choice(idx, size=min(k, len(idx)), replace=False)]
    novel = split_data["novel"]
    taught = np.isin(novel["labels"], words)
    return examples, {"taught": subset(novel, taught), "untaught": subset(novel, ~taught), "known": split_data["known"]}


def run(bundle, cfg, pool, split_data, decided_all, words, k, draws, seed) -> dict:
    rng = np.random.default_rng(seed)
    novel = split_data["novel"]
    taught = np.isin(novel["labels"], words)
    decided = {"taught": tuple(x[taught] for x in decided_all["novel"]),
               "untaught": tuple(x[~taught] for x in decided_all["novel"]), "known": decided_all["known"]}
    results = []
    for _ in range(draws):
        examples, data = prepare(pool, split_data, words, k, rng)
        bank = CustomWordBank(cfg)
        bank.set_examples(examples)
        results.append(rates(bank, cfg, data, decided))
    return {key: None if results[0][key] is None else round(float(np.mean([r[key] for r in results])), 4)
            for key in results[0]}


def objective(m: dict) -> float:
    return (m["recognised"] + m["known_correct"] - 3 * (m["mixup"] + m["taken_as_known"])
            - 3 * m["made_up"] - 10 * m["stolen"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--n-words", nargs="+", type=int, default=[25, 50, 100, 150])
    ap.add_argument("--shots", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--config-out", default=None, help="write the chosen barrier settings here (fewshot.json format)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = ModelBundle(project_path(args.model))
    seg = SegmenterConfig.load(project_path("training/configs/streaming.json"))
    threshold = seg.commit_threshold if seg.commit_threshold is not None else bundle.threshold
    split = project_path(args.split_dir)
    held = json.loads((split / "heldout_words.json").read_text(encoding="utf-8"))
    order = list(np.random.default_rng(SEED).permutation(held))
    novel_rows = read_manifest(split / "novel.csv")
    pool = encode(bundle, [r for r in novel_rows if r.split == "train"])
    sets = {}
    for s in ("val", "test"):
        sets[s] = {"novel": encode(bundle, [r for r in novel_rows if r.split == s]),
                   "known": encode(bundle, read_manifest(split / f"{s}.csv"))}
    known_train = encode(bundle, read_manifest(split / "train.csv"))
    known_ks = [None, 1.5, 2.0, 2.5, 3.0, 4.0]
    boundaries = {kk: None if kk is None else KnownWordBoundaries(known_train["labels"], known_train["embs"], kk)
                  for kk in known_ks}
    decided_by_k = {kk: {s: {part: model_decisions(bundle, sets[s][part], threshold, boundaries[kk])
                             for part in ("novel", "known")} for s in sets} for kk in known_ks}
    print(f"{bundle.name}: {len(held)} held-out words; teach pool {len(pool['labels'])}, "
          f"val novel {len(sets['val']['novel']['labels'])} known {len(sets['val']['known']['labels'])}", flush=True)

    n_max = max(args.n_words)
    words_max = order[:n_max]
    grid = itertools.product([0.5, 0.6, 0.7, 0.8, 0.9], [0.8, 0.95, 1.01], [0.0, 0.02, 0.05, 0.1],
                             [None, 1.0, 2.0, 3.0], [None, 2.0], known_ks)
    tune_words = words_max[: n_max // 2] if n_max == len(held) else words_max
    best, best_obj, best_m, best_kk = None, -1e9, None, None
    for sim, conf, margin, radius, outz, kk in grid:
        cfg = FewShotConfig(sim, conf, margin, radius_k=radius, outlier_z=outz)
        m = run(bundle, cfg, pool, sets["val"], decided_by_k[kk]["val"], tune_words, 5, args.draws, seed=0)
        if objective(m) > best_obj:
            best, best_obj, best_m, best_kk = cfg, objective(m), m, kk
    baseline = FewShotConfig.load(project_path("training/configs/fewshot.json"))
    current = run(bundle, baseline, pool, sets["val"], decided_by_k[None]["val"], tune_words, 5, args.draws, seed=0)
    print(f"chosen on validation: {best.to_dict()} known_k={best_kk} -> {best_m}", flush=True)
    print(f"current settings on validation: {current}", flush=True)
    if args.config_out:
        Path(project_path(args.config_out)).write_text(json.dumps(best.to_dict(), indent=2))
    decided = decided_by_k[best_kk]

    report = {"model": bundle.name, "threshold": threshold, "chosen": best.to_dict(), "known_k": best_kk,
              "validation": best_m, "validation_current_config": current,
              "objective": "recognised + known_correct - 3*(mixup + taken_as_known) - 3*made_up - 10*stolen",
              "note": "tuned with half the held-out words taught (the rest act as never-taught signs)",
              "test": {}, "test_current_config": {}}
    for n in args.n_words:
        for k in args.shots:
            for name, cfg, dec in (("test", best, decided), ("test_current_config", baseline, decided_by_k[None])):
                m = run(bundle, cfg, pool, sets["test"], dec["test"], order[:n], k, args.draws, seed=1)
                report[name][f"N{n}_k{k}"] = m
            m, c = report["test"][f"N{n}_k{k}"], report["test_current_config"][f"N{n}_k{k}"]
            pct = lambda v: "  n/a" if v is None else f"{v:5.1%}"
            print(f"test N={n:3d} k={k:2d} | barriers: recognised {pct(m['recognised'])} mixup {pct(m['mixup'])} "
                  f"made-up {pct(m['made_up'])} stolen {m['stolen']:.2%} known {pct(m['known_correct'])} | current: "
                  f"recognised {pct(c['recognised'])} mixup {pct(c['mixup'])} made-up {pct(c['made_up'])} "
                  f"stolen {c['stolen']:.2%} known {pct(c['known_correct'])}", flush=True)
    Path(project_path(args.out)).parent.mkdir(parents=True, exist_ok=True)
    Path(project_path(args.out)).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
