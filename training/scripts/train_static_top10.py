"""Train an isolated ten-word static-sign prototype from the phone-webcam tracker.

Inputs and outputs are local-only: the source photos and extracted person landmarks must not
be committed. The split uses five visually distinct, fully tracked images per class (four train,
one test) and holds all near-duplicate image groups together.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = ROOT.parent / "MudraPhoneWebcam" / "training_data" / "Frames_Word_Level"
TRACKING_PATH = ROOT / "data" / "processed" / "static_signs" / "tracked_landmarks_v1.npz"
OUTPUT_DIR = ROOT / "data" / "processed" / "static_signs" / "top10_v1"
REPORT_DIR = ROOT / "experiments" / "reports" / "static_signs" / "top10_v1"
LABELS = ("YOU", "I_ME_MINE_MY", "WHAT", "HELP", "FOOD", "WATER", "HOW", "NOT", "NICE", "THANK")
SEED = 20261018


def _phash(path: Path) -> int | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    gray = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.dct(gray)[:8, :8]
    median = float(np.median(low.ravel()[1:]))
    return sum(int(bit) << i for i, bit in enumerate((low > median).ravel()[1:]))


def _groups(paths: list[str]) -> list[list[int]]:
    """Connected components of perceptual hashes; threshold 4 catches near-identical copies."""
    hashes = [_phash(IMAGE_ROOT / Path(path)) for path in paths]
    parent = list(range(len(paths)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, left in enumerate(hashes):
        if left is None:
            continue
        for j in range(i):
            right = hashes[j]
            if right is not None and (left ^ right).bit_count() <= 4:
                parent[find(i)] = find(j)
    result: dict[int, list[int]] = {}
    for index in range(len(paths)):
        result.setdefault(find(index), []).append(index)
    return list(result.values())


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / length if np.isfinite(length) and length > 1e-7 else np.zeros_like(vector)


def _feature(archive: np.lib.npyio.NpzFile, row: int) -> np.ndarray | None:
    pose = archive["pose"][row].astype(np.float32)
    width, height = archive["image_sizes"][row].astype(np.float32)
    pose[:, :3] *= np.asarray((width, height, width), dtype=np.float32)
    if min(pose[11, 3], pose[12, 3]) < 0.35 or not archive["hand_present"][row].any():
        return None
    shoulder_mid = (pose[11, :3] + pose[12, :3]) / 2
    shoulder_width = float(np.linalg.norm(pose[12, :2] - pose[11, :2]))
    if shoulder_width < 1e-5:
        return None
    hip_mid = (pose[23, :3] + pose[24, :3]) / 2 if min(pose[23, 3], pose[24, 3]) >= 0.2 else shoulder_mid + (0, 2 * shoulder_width, 0)
    lateral = _unit(pose[12, :3] - pose[11, :3])
    upward = _unit(shoulder_mid - hip_mid)
    depth = _unit(np.cross(lateral, upward))
    basis = np.stack((lateral, upward, depth))
    body_points = (0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24)
    values: list[float] = []
    for index in body_points:
        point = pose[index, :3] - shoulder_mid
        values.extend((basis @ point / shoulder_width).tolist())
        values.append(float(pose[index, 3]))

    # Face anchors are pose points: eyes (2,5), ears (7,8), mouth (9,10). Face mesh
    # identity-like detail is intentionally excluded; only hand-to-region distances are used.
    anchors = {"eyes": (2, 5), "ears": (7, 8), "mouth": (9, 10)}
    hands = archive["hands"][row].astype(np.float32) * np.asarray((width, height, width), dtype=np.float32)
    present = archive["hand_present"][row]
    descriptions = []
    for slot in np.where(present)[0]:
        hand = hands[slot].astype(np.float32).copy()
        wrist = hand[0].copy()
        palm_width = float(np.linalg.norm(hand[5, :2] - hand[17, :2]))
        if palm_width < 1e-5:
            continue
        # No left/right hand label enters the feature. Slots are ordered by physical position.
        wrist3 = wrist - shoulder_mid
        lateral_position = float(np.dot(wrist3, lateral))
        hand_local = (hand - wrist) / palm_width
        # Palm-normal polarity is canonicalized by the tracker, while its 3D facing is retained.
        normal = archive["palm_directions"][row, slot].astype(np.float32)
        block = (basis @ wrist3 / shoulder_width).tolist()
        block.extend(hand_local.reshape(-1).tolist())
        block.extend(normal.tolist())
        extensions = []
        for tip, mcp in zip((4, 8, 12, 16, 20), (1, 5, 9, 13, 17)):
            length = float(np.linalg.norm(hand[tip] - wrist)) / palm_width
            base = float(np.linalg.norm(hand[mcp] - wrist)) / palm_width
            extensions.append(length / max(base, 1e-4))
        block.extend(extensions)
        block.append(float(sum(value > 1.35 for value in extensions)))
        for name, indices in anchors.items():
            visible = [pose[index, :2] for index in indices if pose[index, 3] >= 0.2]
            if not visible:
                block.extend((2.0, 0.0, 0.0))
                continue
            points = np.asarray(visible)
            distances = np.linalg.norm(hand[:, None, :2] - points[None, :, :], axis=2)
            minimum = float(distances.min()) / shoulder_width
            block.extend((minimum, float(minimum < 0.15), float(minimum < 0.25)))
        descriptions.append((lateral_position, block))
    descriptions.sort(key=lambda item: item[0])
    for _, block in descriptions[:2]:
        values.extend(block)
    while len(descriptions) < 2:
        # Fixed-width missing-hand block; no side label or dominant-hand flag.
        values.extend([0.0] * (3 + 63 + 3 + 5 + 1 + 9))
        descriptions.append((0.0, []))
    values.append(float(len(present.nonzero()[0])))
    result = np.asarray(values, dtype=np.float32)
    return result if np.isfinite(result).all() else None


def main() -> None:
    global IMAGE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path, default=TRACKING_PATH)
    parser.add_argument("--images", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    IMAGE_ROOT = args.images
    if not args.tracking.is_file():
        raise FileNotFoundError(f"Run track_static_images.py first: {args.tracking}")
    archive = np.load(args.tracking, allow_pickle=False)
    labels = archive["labels"].astype(str)
    paths = archive["relative_paths"].astype(str)
    eligible = archive["face_regions_present"] & archive["torso_present"] & archive["hand_present"].any(axis=1)
    eligible &= np.linalg.norm(archive["palm_directions"], axis=2).max(axis=1) > 0
    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    for label in LABELS:
        ids = np.where((labels == label) & eligible)[0]
        label_paths = [paths[index] for index in ids]
        groups = _groups(label_paths)
        if len(groups) < 5:
            raise ValueError(f"{label} has only {len(groups)} complete near-duplicate groups; need five")
        rng = random.Random(SEED + sum(label.encode("utf-8")))
        chosen = rng.sample(groups, 5)
        rng.shuffle(chosen)
        for group_number, group in enumerate(chosen):
            # Use a deterministic representative where a pHash component has multiple copies.
            local_index = sorted(group, key=lambda i: label_paths[i].casefold())[0]
            archive_row = int(ids[local_index])
            vector = _feature(archive, archive_row)
            if vector is None:
                raise ValueError(f"Could not form a feature for {label}/{paths[archive_row]}")
            split = "test" if group_number == 0 else "train"
            rows.append({"label": label, "relative_path": paths[archive_row], "split": split,
                         "group_size": len(group), "feature_row": len(vectors)})
            vectors.append(vector)
    x = np.stack(vectors)
    y = np.asarray([item["label"] for item in rows])
    train = np.asarray([item["split"] == "train" for item in rows])
    test = ~train
    # Scale from train only. Nearest-neighbour uses all four training examples per word.
    mean, std = x[train].mean(axis=0), x[train].std(axis=0)
    std[std < 1e-4] = 1.0
    scaled = (x - mean) / std
    predictions = []
    for index in np.where(test)[0]:
        distances = np.linalg.norm(scaled[train] - scaled[index], axis=1)
        predictions.append(str(y[train][int(np.argmin(distances))]))
    truth = y[test].tolist()
    correct = sum(prediction == label for prediction, label in zip(predictions, truth))

    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "static_top10_model_v1.npz"
    np.savez_compressed(model_path, classes=np.asarray(LABELS), train_features=scaled[train].astype(np.float32),
                        train_labels=y[train], feature_mean=mean, feature_std=std,
                        feature_schema=np.asarray(["mudra_static_top10_body_face_hands_v1"]))
    # The Learning page gets one held-out example per sign, separate from the model features.
    ref_path = args.output / "static_top10_learning_v1.npz"
    source_rows = np.asarray([int(np.where(paths == item["relative_path"])[0][0]) for item in rows if item["split"] == "test"])
    np.savez_compressed(ref_path, labels=y[test], pose=archive["pose"][source_rows],
                        image_sizes=archive["image_sizes"][source_rows],
                        hands=archive["hands"][source_rows], hand_present=archive["hand_present"][source_rows],
                        palm_directions=archive["palm_directions"][source_rows],
                        face_regions_present=archive["face_regions_present"][source_rows])
    split_path = args.report / "top10_split.csv"
    with split_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("label", "relative_path", "split", "group_size", "feature_row"))
        writer.writeheader()
        writer.writerows(rows)
    confusion = {label: {candidate: 0 for candidate in (*LABELS, "UNKNOWN")} for label in LABELS}
    for label, prediction in zip(truth, predictions):
        confusion[label][prediction] += 1
    report = {
        "task": "10-word still-image nearest-neighbour prototype",
        "schema": "mudra_static_top10_body_face_hands_v1",
        "labels": list(LABELS), "source_images_per_label": {label: 5 for label in LABELS},
        "split": "four train and one held-out test image from distinct pHash groups per label; seed 20261018",
        "training_examples": int(train.sum()), "test_examples": int(test.sum()),
        "test_correct": correct, "test_accuracy": correct / int(test.sum()),
        "predictions": [{"truth": label, "prediction": prediction} for label, prediction in zip(truth, predictions)],
        "confusion_matrix": confusion,
        "training_feature_families": ["torso-relative pose/hand locations", "face-relative eye/ear/mouth distances and contact flags",
                                      "21-point finger geometry", "five finger-extension ratios and count", "side-normalized palm normal"],
        "handedness_policy": "No dominant-hand identity or detector handedness category enters the model features; two hands are ordered by physical lateral position. Tracker uses detector side only to canonicalize palm-normal polarity.",
        "limitations": ["Only ten held-out photos: exploratory result, not an accuracy claim for live signing.",
                       "Images are from one directory corpus; signer/session independence is unknown.",
                       "pHash components reduce near-duplicate leakage but do not establish signer independence.",
                       "ISL glosses/folder labels need review by an ISL-fluent signer.",
                       "The separate Learning examples and trained model are ignored local artifacts, not deployed."],
        "artifacts": {"model": str(model_path), "learning_landmarks": str(ref_path), "split": str(split_path)},
    }
    (args.report / "static_top10_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"labels": list(LABELS), "train": int(train.sum()), "test": int(test.sum()),
                      "test_correct": correct, "test_accuracy": report["test_accuracy"],
                      "model": str(model_path), "learning_landmarks": str(ref_path),
                      "report": str(args.report / "static_top10_report.json")}, indent=2))


if __name__ == "__main__":
    main()
