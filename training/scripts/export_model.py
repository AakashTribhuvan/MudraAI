"""Package a trained run for the inference API.

Writes ``<out>/model.pt`` (state dict only, loadable with ``weights_only=True``) and
``<out>/model_card.json`` (everything else inference needs, plus provenance and limits).

The "uncertain" threshold is chosen on the VALIDATION split: the lowest calibrated
confidence (never below ``--min-threshold``) at which predictions kept on validation reach
``--target-accuracy``. Validation clips are complete signs; live sliding windows often are
not, which is why the floor exists.

    python training/scripts/export_model.py --run experiments/runs/mudra_words_<time> --data-source combined \
        --name isl_mudra_combined_v13b_bilstm
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from mudra_ml.config import project_path
from mudra_ml.datasets.isl_dataset import FeatureConfig, ISLLandmarkDataset
from mudra_ml.evaluation.evaluate import load_checkpoint_model, predict
from mudra_ml.evaluation.metrics import softmax

def limitations(num_classes: int) -> list[str]:
    return [
        f"Vocabulary is {num_classes} classes (including _BACKGROUND_); MUDRA app words such as NAMASTE, WATER and HELP are not included.",
        "Test clips come from the same signers and recording sessions as training clips (no dataset has signer IDs): "
        "90% of INCLUDE test clips have a same-word training take 1-2 file numbers away, and every team-word test clip a "
        "take recorded within 2 minutes. Test accuracy therefore measures new takes by known signers, not new people; "
        "on held-out INCLUDE sessions the same recipe reaches about 67%.",
        "Trained on isolated signs; live use cuts the stream into signs with the segmenter in streaming.py and classifies "
        "each whole sign once (no continuous sentence recognition).",
        "Temperature and threshold are fitted on the validation split, which shares signers and sessions with training; "
        "confidence may be over-confident on new people, cameras or lighting.",
    ]


def phrase_limitations(num_classes: int) -> list[str]:
    return [
        (f"{num_classes} whole phrases from 687 videos (about 5 training videos per phrase); 58% top-1 and 81% top-5 on "
         "held-out videos, so the app shows the top guesses for the user to choose, never one automatic answer."),
        "Test videos come from the same signers as training videos (no signer IDs); new people will be recognised less often.",
        "A phrase is classified as one whole clip (start to end); words inside a phrase are not recognised separately.",
        "Variant chosen on validation (seq_len 48, no background class); the test scores of the alternatives were also seen.",
    ]


DATA_SOURCES = {
    "phrases": {
        "name": r"Sentence videos (101 phrases, 687 videos, C:\ISL\Videos_Sentence_Level)",
        "pose_release": "extracted locally with MediaPipe Tasks (same pipeline as the webcam)",
        "license": "provided by the team; source and licence to be confirmed (possibly ISL-CSLTR, CC-BY-4.0)",
        "attribution": "MUDRA team data.",
    },
    "include": {
        "name": "INCLUDE (Sridhar et al., ACM Multimedia 2020)",
        "pose_release": "OpenHands v1 labelled ISLR poses, Zenodo record 6674324",
        "license": "CC-BY-4.0",
        "attribution": "Sridhar, Ganesan, Kumar, Khapra. INCLUDE: A Large Scale Dataset for Indian Sign Language "
        "Recognition. ACM MM 2020. Pose data via AI4Bharat OpenHands.",
    },
    "combined": {
        "name": "INCLUDE (263 words) + MUDRA isolated-word videos (61 words, C:\\ISL\\archive)",
        "pose_release": "INCLUDE via OpenHands (Zenodo 6674324); MUDRA words extracted locally with MediaPipe Tasks",
        "license": "INCLUDE: CC-BY-4.0. MUDRA word videos: provided by the team; source and licence to be confirmed",
        "attribution": "Sridhar et al., INCLUDE, ACM MM 2020; MUDRA team recordings.",
    },
}


def choose_threshold(probs: np.ndarray, labels: np.ndarray, target: float, minimum: float = 0.5) -> dict:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    table = []
    chosen = None
    for t in grid:
        keep = conf >= t
        acc = float(correct[keep].mean()) if keep.any() else None
        table.append({"threshold": float(t), "coverage": float(keep.mean()), "accuracy_on_kept": acc})
        if chosen is None and t >= minimum and acc is not None and acc >= target:
            chosen = float(t)
    if chosen is None:
        chosen = float(grid[-1])
    return {"threshold": chosen, "target_accuracy": target, "minimum_threshold": minimum, "validation_curve": table}


def per_word_test_accuracy(run_dir: Path, split_dir: Path | None = None) -> dict:
    """{word: {"correct": c, "clips": n}} from the run's held-out test predictions (empty if missing).

    Counts recordings, not rows: the INCLUDE video and pose-release copies of one recording share a
    ``group_id`` in the split's test.csv and are counted once, as correct only if every copy is.
    """
    import csv

    path = run_dir / "reports" / "test_predictions.csv"
    if not path.exists():
        return {}
    groups: dict = {}
    test_csv = Path(split_dir) / "test.csv" if split_dir else None
    if test_csv and test_csv.exists():
        with test_csv.open(encoding="utf-8") as f:
            groups = {r["sample_id"]: r["group_id"] for r in csv.DictReader(f)}
    recordings: dict = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["true"], groups.get(row["sample_id"], row["sample_id"]))
            recordings[key] = recordings.get(key, True) and row["true"] == row["predicted"]
    out: dict = {}
    for (word, _), ok in recordings.items():
        entry = out.setdefault(word, {"correct": 0, "clips": 0})
        entry["clips"] += 1
        entry["correct"] += int(ok)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-root", default="app/backend/models", type=Path)
    parser.add_argument("--target-accuracy", type=float, default=0.985)
    parser.add_argument(
        "--min-threshold", type=float, default=0.5,
        help="floor: validation clips are complete signs, live windows often are not, so never accept below this",
    )
    parser.add_argument("--data-source", default="include", choices=sorted(DATA_SOURCES))
    args = parser.parse_args()

    run_dir = project_path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    device = torch.device("cpu")
    model, ckpt = load_checkpoint_model(run_dir / "best.pt", device)
    feature_cfg = FeatureConfig.from_dict(ckpt["feature_config"])
    temperature = float(ckpt["temperature"])

    val = ISLLandmarkDataset(project_path(cfg["data"]["split_dir"]) / "val.csv", ckpt["classes"], feature_cfg)
    val.set_stats(np.asarray(ckpt["feature_mean"]), np.asarray(ckpt["feature_std"]))
    logits, labels = predict(model, DataLoader(val, batch_size=128), device, amp=False)
    threshold = choose_threshold(softmax(logits, temperature), labels, args.target_accuracy, args.min_threshold)

    out = project_path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    card = {
        "name": args.name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run": run_dir.name,
        "task": ("Whole-phrase" if args.data_source == "phrases" else "Isolated")
        + " Indian Sign Language " + ("phrase" if args.data_source == "phrases" else "word")
        + " recognition from MediaPipe hand + pose landmarks",
        "classes": ckpt["classes"],
        "model_config": ckpt["model_config"],
        "feature_config": ckpt["feature_config"],
        "feature_mean": np.asarray(ckpt["feature_mean"]).round(6).tolist(),
        "feature_std": np.asarray(ckpt["feature_std"]).round(6).tolist(),
        "temperature": temperature,
        "uncertain_threshold": threshold["threshold"],
        "threshold_selection": threshold,
        "metrics": {"validation": summary["val"], "test": summary["test"], "test_samples": summary["test_samples"]},
        "data": DATA_SOURCES[args.data_source],
        "limitations": (phrase_limitations if args.data_source == "phrases" else limitations)(len(ckpt["classes"])),
    }
    per_word = per_word_test_accuracy(run_dir, project_path(cfg["data"]["split_dir"]))
    if per_word:
        card["per_word_test"] = per_word
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    kept = next(r for r in threshold["validation_curve"] if r["threshold"] == threshold["threshold"])
    print(
        f"Exported {args.name} -> {out}\n"
        f"  classes {len(card['classes'])}, temperature {temperature:.3f}, uncertain below {threshold['threshold']:.2f} "
        f"(validation: keeps {kept['coverage']:.1%} of clips at {kept['accuracy_on_kept']:.1%} accuracy)"
    )


if __name__ == "__main__":
    main()
