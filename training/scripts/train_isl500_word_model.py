"""Prepare and train a signer-independent pilot word model from the local ISL500 subset.

Run from the project root with app/backend/.venv/Scripts/python.exe. The video files and
landmark arrays remain under ignored data/ folders. Split policy: Users 013-015 are locked
as the final test signers; grouped 4-fold CV over Users 001-012 selects the classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "external" / "isl500_sentence_subset"
VIDEO_ROOT = DATA_ROOT / "videos"
TRACK_ROOT = ROOT / "data" / "processed" / "isl500_sentence_subset" / "landmarks"
MANIFEST_PATH = ROOT / "data" / "metadata" / "manifest_isl500_sentence_subset.csv"
SPLIT_PATH = ROOT / "data" / "metadata" / "splits" / "isl500_sentence_subset" / "split.csv"
REPORT_ROOT = ROOT / "experiments" / "reports" / "isl500_sentence_subset"
MODEL_ROOT = ROOT / "data" / "processed" / "isl500_sentence_subset" / "model"
ASSET_ROOT = ROOT / "app" / "frontend" / "public" / "models"
WORDS = (
    "FOOD", "HELP", "I", "MYSELF", "NICE", "NO", "THANKYOU", "WATER", "WHAT", "YOU",
    "HELLO", "WE", "STUDENT", "THIS",
)
TEST_SIGNERS = {"User013", "User014", "User015"}
SEQUENCE_LENGTH = 24
FEATURE_SCHEMA = "mudra_static_top10_body_face_hands_v1"
_EXTRACTOR = None


def _init_extractor(max_fps: float) -> None:
    global _EXTRACTOR
    from mudra_ml.preprocessing.landmark_extractor import LandmarkExtractor

    _EXTRACTOR = LandmarkExtractor(
        asset_dir=ASSET_ROOT,
        pose_model="pose_landmarker_lite.task",
        min_hand_detection_confidence=0.25,
        min_pose_detection_confidence=0.35,
        max_fps=max_fps,
        mirrored_input=False,
    )


def _palm_directions(hands: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Estimate side-canonicalized palm normals from tracked 3D hand landmarks."""
    result = np.zeros((len(hands), 2, 3), dtype=np.float32)
    for frame_index in range(len(hands)):
        for slot in range(2):
            if not present[frame_index, slot]:
                continue
            hand = hands[frame_index, slot]
            normal = np.cross(hand[5] - hand[0], hand[17] - hand[0])
            length = float(np.linalg.norm(normal))
            if not math.isfinite(length) or length < 1e-8:
                continue
            # Match the app tracker convention: normalize the winding across anatomical hands.
            result[frame_index, slot] = normal / length * (-1.0 if slot == 1 else 1.0)
    return result


def _track_one(row: dict, overwrite: bool) -> dict:
    from mudra_ml.preprocessing.landmark_extractor import LandmarkSequence, load_sequence

    source = Path(row["raw_path"])
    target = Path(row["landmarks_path"])
    if target.is_file() and not overwrite:
        sequence = load_sequence(target)
    else:
        if _EXTRACTOR is None:
            raise RuntimeError("Landmark extractor was not initialized")
        sequence = _EXTRACTOR.extract_video(source)
        sequence.meta.update({
            "label": row["label"], "source": "ISL500/ISL-DATA",
            "signer_id": row["signer_id"], "sample_id": row["sample_id"],
            "face_anchor_indices": {"eyes": [2, 5], "ears": [7, 8], "mouth": [9, 10]},
            "tracking_note": "PoseLandmarker includes eyes, ears, mouth, torso and arms; hands include finger joints.",
            "palm_direction_note": "A canonicalized 3D normal computed from the tracked hand points.",
        })
        sequence.save(target)
        # Keep the derived palm direction with each raw landmark sequence for the Learning view.
        with np.load(target, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["palm_directions"] = _palm_directions(sequence.hands, sequence.hand_present)
        np.savez_compressed(target, **payload)

    rates = sequence.detection_rates()
    palm = _palm_directions(sequence.hands, sequence.hand_present)
    valid_palm_rate = float((np.linalg.norm(palm, axis=2).max(axis=1) > 0).mean())
    return {
        **row,
        "num_frames": sequence.num_frames,
        "fps": float(sequence.meta.get("fps") or 0),
        "duration_s": sequence.num_frames / float(sequence.meta.get("fps") or 1),
        "any_hand_rate": rates["any_hand"],
        "pose_rate": rates["pose"],
        "palm_direction_rate": valid_palm_rate,
        "quality_flag": "" if rates["any_hand"] >= 0.5 and rates["pose"] >= 0.5 else "low_tracking_coverage",
    }


def prepare(args: argparse.Namespace) -> None:
    if not VIDEO_ROOT.is_dir():
        raise FileNotFoundError(f"Downloaded videos not found: {VIDEO_ROOT}")
    files = sorted(VIDEO_ROOT.glob("*/*"))
    files = [p for p in files if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi"}]
    rows = []
    for path in files:
        label = path.name.split("__", 1)[0].upper()
        signer = path.parent.name
        if signer.startswith("Extra_") and not getattr(args, "include_extra_training", False):
            continue
        if label not in WORDS or not (signer.startswith("User") or signer.startswith("Extra_")):
            continue
        sample_id = f"{signer}_{path.stem}"
        output = TRACK_ROOT / label / f"{sample_id}.npz"
        rows.append({
            "sample_id": sample_id, "label": label, "source": "isl500_sentence_subset", "kind": "video",
            "signer_id": signer, "group_id": f"isl500_sentence_subset:signer:{signer}",
            "raw_path": str(path), "landmarks_path": str(output),
        })
    expected = len(WORDS) * 15
    per_class = {label: sum(row["label"] == label and row["signer_id"].startswith("User") for row in rows) for label in WORDS}
    if any(count != 15 for count in per_class.values()) or len(rows) < expected:
        raise ValueError(f"Expected 15 locked ISL500 signer videos per class plus optional training-only source clips; found {per_class}, total={len(rows)}")

    TRACK_ROOT.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_extractor, initargs=(args.max_fps,)) as pool:
        futures = [pool.submit(_track_one, row, args.overwrite) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tracking ISL500 clips"):
            results.append(future.result())

    results.sort(key=lambda item: (item["label"], item["signer_id"]))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    split_rows = []
    for row in results:
        split_rows.append({
            "sample_id": row["sample_id"], "label": row["label"], "signer_id": row["signer_id"],
            "group_id": row["group_id"], "split": "test" if row["signer_id"] in TEST_SIGNERS else "train",
            "raw_path": row["raw_path"], "landmarks_path": row["landmarks_path"],
            "quality_flag": row["quality_flag"], "num_frames": row["num_frames"],
            "any_hand_rate": row["any_hand_rate"], "pose_rate": row["pose_rate"],
            "palm_direction_rate": row["palm_direction_rate"],
        })
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SPLIT_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "videos": len(results),
        "words": len(WORDS),
        "signers": 15,
        "locked_test_signers": sorted(TEST_SIGNERS),
        "train_signers": [f"User{i:03}" for i in range(1, 13)],
        "clips_with_low_tracking_coverage": sum(bool(r["quality_flag"]) for r in results),
        "mean_hand_detection_rate": float(np.mean([r["any_hand_rate"] for r in results])),
        "mean_pose_detection_rate": float(np.mean([r["pose_rate"] for r in results])),
        "mean_palm_direction_rate": float(np.mean([r["palm_direction_rate"] for r in results])),
        "manifest": str(MANIFEST_PATH), "split": str(SPLIT_PATH),
        "landmarks_dir": str(TRACK_ROOT),
    }
    (REPORT_ROOT / "tracking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _load_clip_features(row: dict, mirror: bool = False) -> tuple[np.ndarray, float]:
    """Build a fixed-length temporal descriptor using the backend's live body/face/hand feature."""
    sys.path.insert(0, str(ROOT / "app" / "backend"))
    from mudra_api.static_top10 import feature_from_frame
    from mudra_ml.preprocessing.frame_sampler import sample_indices
    from mudra_ml.preprocessing.sequence import load_sequence, mirror_sequence
    from mudra_ml.streaming import Frame

    sequence = load_sequence(Path(row["landmarks_path"]))
    if mirror:
        sequence = mirror_sequence(sequence)
    with np.load(row["landmarks_path"], allow_pickle=False) as archive:
        palm = archive["palm_directions"].copy()
    if mirror:
        palm = palm[:, ::-1].copy()
        palm[..., 0] *= -1
    width = int(sequence.meta.get("width") or 0)
    height = int(sequence.meta.get("height") or 0)
    # Dataset clips contain long rest sections. Crop to the tracked signing span before
    # uniform resampling so the classifier does not learn that every word is mostly idle.
    active = np.flatnonzero(sequence.hand_present.any(axis=1))
    if not len(active):
        raise ValueError(f"No usable tracking frames: {row['raw_path']}")
    context = max(1, round(sequence.num_frames * 0.04))
    start = max(0, int(active[0]) - context)
    end = min(sequence.num_frames - 1, int(active[-1]) + context)
    indices = sample_indices(end - start + 1, SEQUENCE_LENGTH) + start
    frames: list[np.ndarray | None] = []
    for index in indices:
        frame = Frame(
            t=float(sequence.timestamps_ms[index]) / 1000,
            hands=sequence.hands[index],
            present=sequence.hand_present[index],
            pose=sequence.pose[index],
            pose_present=bool(sequence.pose_present[index]),
        )
        frames.append(feature_from_frame(frame, palm[index], width, height))
    valid = np.asarray([frame is not None for frame in frames], dtype=bool)
    if not valid.any():
        raise ValueError(f"No usable tracking frames: {row['raw_path']}")
    features = np.stack([frame if frame is not None else np.zeros(229, np.float32) for frame in frames])
    # Fill brief detector dropouts from the nearest valid frame; preserve an explicit coverage score.
    for index in np.flatnonzero(~valid):
        nearest = np.flatnonzero(valid)[np.argmin(np.abs(np.flatnonzero(valid) - index))]
        features[index] = features[nearest]
    return features, float(valid.mean())


def _describe(sequence: np.ndarray, kind: str) -> np.ndarray:
    if kind.startswith("bins"):
        bins = int(kind.removeprefix("bins"))
        edges = np.linspace(0, len(sequence), bins + 1).round().astype(int)
        output = []
        for start, end in zip(edges[:-1], edges[1:]):
            block = sequence[start:max(start + 1, end)]
            output.extend((block.mean(axis=0), block.std(axis=0)))
        return np.concatenate(output).astype(np.float32)
    frames = int(kind.removeprefix("flat"))
    ids = np.rint(np.linspace(0, len(sequence) - 1, frames)).astype(int)
    return sequence[ids].reshape(-1).astype(np.float32)


def train(args: argparse.Namespace) -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Run this script with `prepare` first.")
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    usable = [row for row in rows if Path(row["landmarks_path"]).is_file()]
    feature_cache: dict[tuple[str, bool], tuple[np.ndarray, float]] = {}
    unusable: list[dict[str, str]] = []
    for row in tqdm(usable, desc="Building temporal features"):
        try:
            feature_cache[(row["sample_id"], False)] = _load_clip_features(row)
            feature_cache[(row["sample_id"], True)] = _load_clip_features(row, mirror=True)
        except ValueError as exc:
            unusable.append({"sample_id": row["sample_id"], "label": row["label"],
                             "signer": row["signer_id"], "reason": str(exc)})
    usable = [row for row in usable if (row["sample_id"], False) in feature_cache]

    train_rows = [row for row in usable if row["signer_id"] not in TEST_SIGNERS]
    test_rows = [row for row in usable if row["signer_id"] in TEST_SIGNERS]
    labels = sorted(WORDS)
    train_counts = {label: sum(row["label"] == label for row in train_rows) for label in labels}
    test_counts = {label: sum(row["label"] == label for row in test_rows) for label in labels}
    if any(count < 8 for count in train_counts.values()):
        raise ValueError(f"Some classes lack enough usable training clips: {train_counts}")
    if any(count < 1 for count in test_counts.values()):
        raise ValueError(f"A class has no usable clip from the held-out signers: {test_counts}")

    candidates = {}
    for descriptor in ("bins4", "bins6", "flat12"):
        for kernel, c in (("linear", 0.01), ("linear", 0.1), ("linear", 1.0),
                          ("rbf", 0.1), ("rbf", 1.0), ("rbf", 10.0)):
            candidates[f"{descriptor}_svm_{kernel}_c{c:g}"] = (descriptor, make_pipeline(
                StandardScaler(), SVC(kernel=kernel, C=c, gamma="scale", class_weight="balanced")
            ))
    candidates["bins4_knn1"] = ("bins4", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1, weights="distance")))
    candidates["bins4_knn3"] = ("bins4", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3, weights="distance")))

    train_groups = np.asarray([row["signer_id"] for row in train_rows])
    train_y = np.asarray([row["label"] for row in train_rows])
    cv = GroupKFold(n_splits=4)
    cv_scores: dict[str, list[dict[str, float]]] = {name: [] for name in candidates}
    for fold, (fit_ids, val_ids) in enumerate(cv.split(train_rows, train_y, train_groups), start=1):
        for name, (descriptor, estimator) in candidates.items():
            x_fit = []
            y_fit = []
            for idx in fit_ids:
                row = train_rows[idx]
                original, _ = feature_cache[(row["sample_id"], False)]
                flipped, _ = feature_cache[(row["sample_id"], True)]
                x_fit.extend((_describe(original, descriptor), _describe(flipped, descriptor)))
                y_fit.extend((row["label"], row["label"]))
            x_val = np.stack([_describe(feature_cache[(train_rows[idx]["sample_id"], False)][0], descriptor) for idx in val_ids])
            y_val = train_y[val_ids]
            model = clone(estimator).fit(np.stack(x_fit), np.asarray(y_fit))
            prediction = model.predict(x_val)
            cv_scores[name].append({
                "fold": fold,
                "accuracy": float(accuracy_score(y_val, prediction)),
                "macro_f1": float(f1_score(y_val, prediction, labels=labels, average="macro", zero_division=0)),
            })

    cv_means = {
        name: {
            "macro_f1": float(np.mean([fold["macro_f1"] for fold in scores])),
            "accuracy": float(np.mean([fold["accuracy"] for fold in scores])),
        }
        for name, scores in cv_scores.items()
    }
    best = max(candidates, key=lambda name: (cv_means[name]["macro_f1"], cv_means[name]["accuracy"]))
    descriptor, estimator = candidates[best]

    x_fit, y_fit = [], []
    for row in train_rows:
        original, _ = feature_cache[(row["sample_id"], False)]
        flipped, _ = feature_cache[(row["sample_id"], True)]
        x_fit.extend((_describe(original, descriptor), _describe(flipped, descriptor)))
        y_fit.extend((row["label"], row["label"]))
    fit_x, fit_y = np.stack(x_fit), np.asarray(y_fit)
    fit_groups = np.repeat(train_groups, 2)  # original + mirrored take stay in one signer fold
    calibration_splits = list(GroupKFold(n_splits=4).split(fit_x, fit_y, fit_groups))
    # Grouped out-of-fold calibration uses only the 12 training signers; the final three
    # signers stay untouched until the one final evaluation below.
    model = CalibratedClassifierCV(
        estimator=clone(estimator), method="sigmoid", cv=calibration_splits, ensemble=False
    ).fit(fit_x, fit_y)
    x_test = np.stack([_describe(feature_cache[(row["sample_id"], False)][0], descriptor) for row in test_rows])
    y_test = np.asarray([row["label"] for row in test_rows])
    prediction = model.predict(x_test)
    report = {
        "task": "ISL500 signer-independent 14-word pilot",
        "source": "ISL500/ISL-DATA Videos",
        "words": labels,
        "samples_total": len(usable),
        "excluded_untrackable_samples": unusable,
        "usable_train_clips_by_word": train_counts,
        "usable_test_clips_by_word": test_counts,
        "train_samples_original": len(train_rows),
        "train_samples_after_mirror_augmentation": len(y_fit),
        "test_samples": len(test_rows),
        "train_signers": sorted(set(train_groups)),
        "test_signers_locked_before_model_selection": sorted(TEST_SIGNERS),
        "split": "Users 001-012 for training and grouped 4-fold classifier selection; Users 013-015 held out untouched for final test.",
        "tracking": "MediaPipe HandLandmarker + PoseLandmarker Lite; 24 uniformly sampled frames; pose supplies eyes, ears, mouth, shoulders, arms and hips; hand joints and canonicalized 3D palm normals; backend live per-frame 229D feature.",
        "mirror_augmentation": "Each training clip has a mirrored counterpart; no test clip is augmented.",
        "selected_model": best,
        "selected_descriptor": descriptor,
        "confidence_calibration": "Sigmoid calibration on 4-fold out-of-fold predictions grouped by training signer; test signers excluded.",
        "grouped_cv_mean_by_candidate": cv_means,
        "grouped_cv_folds_by_candidate": cv_scores,
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_macro_f1": float(f1_score(y_test, prediction, labels=labels, average="macro", zero_division=0)),
        "test_classification_report": classification_report(y_test, prediction, labels=labels, output_dict=True, zero_division=0),
        "test_confusion_matrix_labels": labels,
        "test_confusion_matrix": confusion_matrix(y_test, prediction, labels=labels).tolist(),
        "test_predictions": [{"truth": str(actual), "prediction": str(guess), "signer": row["signer_id"], "sample_id": row["sample_id"]}
                             for actual, guess, row in zip(y_test, prediction, test_rows)],
        "mean_test_frame_coverage": float(np.mean([feature_cache[(row["sample_id"], False)][1] for row in test_rows])),
        "limitations": [
            "Only 15 examples per word are available; the independent test has three clips per word.",
            "Clip-level results are not live webcam results; unknown signers/cameras may differ.",
            "The source glosses and regional ISL variants need review by an ISL-fluent signer.",
            "I, MYSELF and NO are dataset-available pilot labels replacing missing requested labels in the first download list; they are not asserted to be gloss-equivalent to the missing words.",
            "A video label is an isolated-gloss class, not an English sentence translation.",
        ],
    }

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / "isl500_word_model_v1.joblib"
    joblib.dump({
        "estimator": model,
        "classes": labels,
        "feature_schema": FEATURE_SCHEMA,
        "sequence_length": SEQUENCE_LENGTH,
        "descriptor": descriptor,
        "words": labels,
        "source": "ISL500/ISL-DATA",
    }, model_path)
    report_path = REPORT_ROOT / "model_report.json"
    report["artifacts"] = {"model": str(model_path), "report": str(report_path), "split": str(SPLIT_PATH), "manifest": str(MANIFEST_PATH)}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "train_signers": len(set(train_groups)), "test_signers": len(TEST_SIGNERS),
        "train_clips": len(train_rows), "test_clips": len(test_rows),
        "selected_model": best, "descriptor": descriptor,
        "cv_macro_f1": cv_means[best]["macro_f1"], "cv_accuracy": cv_means[best]["accuracy"],
        "test_accuracy": report["test_accuracy"], "test_macro_f1": report["test_macro_f1"],
        "model": str(model_path), "report": str(report_path),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="extract and save video landmarks")
    prepare_parser.add_argument("--workers", type=int, default=4)
    prepare_parser.add_argument("--max-fps", type=float, default=15)
    prepare_parser.add_argument("--overwrite", action="store_true")
    train_parser = commands.add_parser("train", help="select and train a model using grouped signer CV")
    train_parser.set_defaults(func=train)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
