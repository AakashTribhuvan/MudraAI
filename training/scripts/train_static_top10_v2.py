"""Compare small-data static sign classifiers with grouped validation.

This experiment reuses the locally extracted body/face/hand landmarks. Near-duplicate
photos are grouped by pHash before splitting so they cannot cross split boundaries.
Classifier selection uses cross-validation on the training partition; the held-out
partition is evaluated once after selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_static_top10 import IMAGE_ROOT, LABELS, SEED, TRACKING_PATH, _feature, _phash

ROOT = Path(__file__).resolve().parents[2]


def grouped(paths: list[str], image_root: Path) -> list[list[int]]:
    hashes = [_phash(image_root / Path(path)) for path in paths]
    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, left in enumerate(hashes):
        if left is None:
            continue
        for j in range(i):
            right = hashes[j]
            if right is not None and (left ^ right).bit_count() <= 4:
                parent[find(i)] = find(j)
    components: dict[int, list[int]] = {}
    for i in range(len(paths)):
        components.setdefault(find(i), []).append(i)
    return list(components.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path, default=TRACKING_PATH)
    parser.add_argument("--images", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/static_signs/top10_v2")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments/reports/static_signs/top10_v2")
    args = parser.parse_args()
    if not args.tracking.is_file():
        raise FileNotFoundError(f"Run track_static_images.py first: {args.tracking}")

    data = np.load(args.tracking, allow_pickle=False)
    labels = data["labels"].astype(str)
    paths = data["relative_paths"].astype(str)
    eligible = data["face_regions_present"] & data["torso_present"] & data["hand_present"].any(axis=1)
    eligible &= np.linalg.norm(data["palm_directions"], axis=2).max(axis=1) > 0

    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    group_ids: list[int] = []
    next_group = 0
    for label in LABELS:
        ids = np.where((labels == label) & eligible)[0]
        label_paths = [paths[i] for i in ids]
        components = grouped(label_paths, args.images)
        if len(components) < 5:
            raise ValueError(f"{label}: only {len(components)} distinct pHash groups; need at least 5")
        rng = random.Random(SEED + sum(label.encode("utf-8")))
        rng.shuffle(components)
        target = max(1, round(len(ids) * 0.2))
        test_groups: set[int] = set()
        test_size = 0
        for gi, component in enumerate(components):
            if test_size < target and len(components) - len(test_groups) > 2:
                test_groups.add(gi)
                test_size += len(component)
        for gi, component in enumerate(components):
            gid = next_group
            next_group += 1
            # Keep every photo from a duplicate component together; the lexicographically
            # first file is the representative so the report is easy to inspect.
            for local in component:
                source_row = int(ids[local])
                vector = _feature(data, source_row)
                if vector is None:
                    continue
                rows.append({"label": label, "relative_path": paths[source_row],
                             "split": "test" if gi in test_groups else "train",
                             "group_id": gid, "group_size": len(component)})
                features.append(vector)
                group_ids.append(gid)
        next_group += 1

    x = np.stack(features)
    y = np.asarray([r["label"] for r in rows])
    groups = np.asarray(group_ids)
    train_mask = np.asarray([r["split"] == "train" for r in rows])
    test_mask = ~train_mask
    train_x, train_y, train_groups = x[train_mask], y[train_mask], groups[train_mask]
    test_x, test_y = x[test_mask], y[test_mask]
    if len(set(train_y)) != len(LABELS) or len(set(test_y)) != len(LABELS):
        raise ValueError("Every label must have both training and test examples")

    # Per-class grouped folds, deterministic and disjoint. Fold construction only sees
    # training data; no held-out examples influence classifier choice.
    fold_count = min(4, min(len(set(train_groups[train_y == label])) for label in LABELS))
    if fold_count < 2:
        raise ValueError("Not enough independent groups per label for grouped CV")
    fold_for_group: dict[int, int] = {}
    for label in LABELS:
        label_group_ids = list(dict.fromkeys(train_groups[train_y == label].tolist()))
        random.Random(SEED + sum(label.encode("utf-8"))).shuffle(label_group_ids)
        for index, gid in enumerate(label_group_ids):
            fold_for_group[int(gid)] = index % fold_count

    candidates = {
        "nearest_centroid": make_pipeline(StandardScaler(), NearestCentroid()),
        "knn_1": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1, weights="distance")),
        "knn_3": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3, weights="distance")),
        "svm_linear_c1": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1, class_weight="balanced")),
        "svm_rbf_c1": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")),
        "svm_rbf_c10": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
    }
    cv_scores: dict[str, list[dict[str, float]]] = {name: [] for name in candidates}
    for fold in range(fold_count):
        val_mask = np.asarray([fold_for_group[int(gid)] == fold for gid in train_groups])
        fit_mask = ~val_mask
        for name, estimator in candidates.items():
            model = clone(estimator).fit(train_x[fit_mask], train_y[fit_mask])
            pred = model.predict(train_x[val_mask])
            cv_scores[name].append({"accuracy": float(accuracy_score(train_y[val_mask], pred)),
                                    "macro_f1": float(f1_score(train_y[val_mask], pred, labels=LABELS, average="macro", zero_division=0))})
    cv_means = {name: float(np.mean([fold["macro_f1"] for fold in scores])) for name, scores in cv_scores.items()}
    best_name = max(candidates, key=lambda name: (cv_means[name], np.mean([s["accuracy"] for s in cv_scores[name]])))
    model = clone(candidates[best_name]).fit(train_x, train_y)
    prediction = model.predict(test_x)
    test_accuracy = float(accuracy_score(test_y, prediction))
    test_macro_f1 = float(f1_score(test_y, prediction, labels=LABELS, average="macro", zero_division=0))
    cm = confusion_matrix(test_y, prediction, labels=LABELS)

    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    # Store a portable sklearn estimator via joblib alongside its feature schema.
    import joblib
    model_path = args.output / "static_top10_model_v2.joblib"
    joblib.dump({"estimator": model, "classes": LABELS,
                 "feature_schema": "mudra_static_top10_body_face_hands_v1"}, model_path)
    split_path = args.report / "top10_v2_split.csv"
    with split_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("label", "relative_path", "split", "group_id", "group_size"))
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.report / "static_top10_v2_report.json"
    report = {
        "task": "10-word static-sign classifier comparison with grouped holdout",
        "labels": list(LABELS), "training_examples": int(train_mask.sum()), "test_examples": int(test_mask.sum()),
        "examples_per_label": {label: {split: sum(r["label"] == label and r["split"] == split for r in rows)
                                          for split in ("train", "test")} for label in LABELS},
        "split": "approximately 80/20 per class, keeping pHash-connected groups together; seed 20261018",
        "grouped_cv_folds": fold_count, "cv_macro_f1_by_classifier": cv_means,
        "selected_classifier": best_name,
        "test_accuracy": test_accuracy, "test_macro_f1": test_macro_f1,
        "test_report": classification_report(test_y, prediction, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix_labels": list(LABELS), "confusion_matrix": cm.tolist(),
        "predictions": [{"truth": str(a), "prediction": str(b)} for a, b in zip(test_y, prediction)],
        "limitations": ["Single corpus; signer and recording-session independence are unknown.",
                        "This is still-image recognition, not a live-stream result.",
                        "pHash grouping reduces near-copy leakage but does not ensure signer-independent evaluation.",
                        "Folder labels/glosses should be reviewed by an ISL-fluent signer."],
        "artifacts": {"model": str(model_path), "split": str(split_path)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"train": int(train_mask.sum()), "test": int(test_mask.sum()),
                      "selected_classifier": best_name, "cv_macro_f1": cv_means[best_name],
                      "test_accuracy": test_accuracy, "test_macro_f1": test_macro_f1,
                      "model": str(model_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
