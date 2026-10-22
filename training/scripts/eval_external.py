"""Evaluate an exported model on another dataset's clips (new signers, cameras and rooms).

Only clips whose label is in the model's vocabulary are scored; the model still chooses
among all of its classes, exactly as in the app.

    python training/scripts/eval_external.py --model app/backend/models/isl_mudra_combined_v15b_bilstm \
        --manifest data/metadata/manifest_islwords.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mudra_ml.config import project_path
from mudra_ml.data.manifest import read_manifest
from mudra_ml.datasets.isl_dataset import FeatureConfig, sequence_features
from mudra_ml.models.sequence_model import ModelConfig, build_model
from mudra_ml.preprocessing.sequence import load_sequence, mirror_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", default=None, type=Path, help="optional JSON report path")
    parser.add_argument("--mirror", action="store_true", help="flip clips horizontally (test for mirrored recordings)")
    args = parser.parse_args()

    model_dir = project_path(args.model)
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    classes = card["classes"]
    index = {c: i for i, c in enumerate(classes)}
    fcfg = FeatureConfig.from_dict(card["feature_config"])
    mean, std = np.asarray(card["feature_mean"], np.float32), np.asarray(card["feature_std"], np.float32)
    model = build_model(ModelConfig.from_dict(card["model_config"]))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    rows = [r for r in read_manifest(project_path(args.manifest)) if r.landmarks_path and r.label in index]
    per_label = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0, "confident": 0, "confident_correct": 0})
    confusions = defaultdict(int)
    for r in rows:
        seq = load_sequence(Path(r.landmarks_path))
        feats = sequence_features(mirror_sequence(seq) if args.mirror else seq, fcfg)
        feats = (np.clip(feats, -fcfg.clip_value, fcfg.clip_value) - mean) / std
        with torch.no_grad():
            logits = model(torch.from_numpy(feats[None])).numpy()[0] / card["temperature"]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        order = np.argsort(-probs)
        stats = per_label[r.label]
        stats["n"] += 1
        stats["top1"] += int(order[0] == index[r.label])
        stats["top5"] += int(index[r.label] in order[:5])
        if probs[order[0]] >= card["uncertain_threshold"]:
            stats["confident"] += 1
            stats["confident_correct"] += int(order[0] == index[r.label])
        if order[0] != index[r.label]:
            confusions[(r.label, classes[order[0]])] += 1

    n = sum(s["n"] for s in per_label.values())
    tot = {k: sum(s[k] for s in per_label.values()) for k in ("top1", "top5", "confident", "confident_correct")}
    report = {
        "model": card["name"],
        "manifest": str(args.manifest),
        "clips": n,
        "top1": tot["top1"] / n,
        "top5": tot["top5"] / n,
        "answered_share": tot["confident"] / n,
        "accuracy_when_answered": tot["confident_correct"] / tot["confident"] if tot["confident"] else None,
        "per_label": {k: {**v, "top1_rate": v["top1"] / v["n"]} for k, v in sorted(per_label.items())},
        "top_confusions": [
            {"true": t, "predicted": p, "count": c} for (t, p), c in sorted(confusions.items(), key=lambda kv: -kv[1])[:12]
        ],
    }
    print(
        f"{n} clips | top-1 {report['top1']:.1%} | top-5 {report['top5']:.1%} | "
        f"answered (>= threshold) {report['answered_share']:.1%}"
        + (f" at {report['accuracy_when_answered']:.1%} accuracy" if report["accuracy_when_answered"] is not None else "")
    )
    for label, s in report["per_label"].items():
        print(f"  {label:16s} {s['top1']:3d}/{s['n']:<3d} top-1  {s['top5']:3d}/{s['n']:<3d} top-5")
    for c in report["top_confusions"][:8]:
        print(f"  confused {c['true']} -> {c['predicted']} x{c['count']}")
    if args.out:
        out = project_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
