"""Track body, face, hand, and palm-direction landmarks in static sign images.

The source images are read only. Landmark arrays, reports, and previews are written to
ignored local output folders so source photos and personal landmarks are never committed.
This is an extraction/inspection step; it does not train or replace a recognition model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_ROOT = ROOT.parent / "MudraPhoneWebcam" / "training_data" / "Frames_Word_Level"
MODEL_ROOT = ROOT / "app" / "frontend" / "public" / "models"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "static_signs" / "tracked_landmarks_v1.npz"
DEFAULT_REPORT_DIR = ROOT / "experiments" / "reports" / "static_signs" / "tracking"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FACE_POINTS = 478
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
)
POSE_EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24))
POSE_TRACK_POINTS = (2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24)


def _xyz(landmarks: Any) -> np.ndarray:
    return np.asarray([(point.x, point.y, point.z) for point in landmarks], dtype=np.float32)


def _pose(landmarks: Any, world: bool = False) -> np.ndarray:
    points = np.zeros((33, 4), dtype=np.float32)
    if not landmarks:
        return points
    for index, point in enumerate(landmarks[:33]):
        points[index] = (
            point.x,
            point.y,
            point.z,
            float(getattr(point, "visibility", 0.0) or 0.0),
        )
    return points


def _face(landmarks: Any) -> np.ndarray:
    points = np.zeros((MAX_FACE_POINTS, 4), dtype=np.float32)
    if not landmarks:
        return points
    count = min(len(landmarks), MAX_FACE_POINTS)
    for index, point in enumerate(landmarks[:count]):
        points[index] = (point.x, point.y, point.z, 1.0)
    return points


def _detect_face(face_model: Any, image_rgb: np.ndarray, pose_landmarks: Any) -> tuple[np.ndarray, bool]:
    """Retry face mesh on an enlarged head crop when the full upper-body frame is too small."""
    height, width = image_rgb.shape[:2]
    result = face_model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))
    if result.face_landmarks:
        return _face(result.face_landmarks[0]), False
    if not pose_landmarks or len(pose_landmarks) < 13:
        return _face([]), False

    visible = [index for index in (0, 2, 5, 7, 8, 9, 10)
               if float(getattr(pose_landmarks[index], "visibility", 0.0) or 0.0) >= 0.2]
    if len(visible) < 3:
        return _face([]), False
    face_x = [float(pose_landmarks[index].x) * width for index in visible]
    face_y = [float(pose_landmarks[index].y) * height for index in visible]
    shoulder_span = float(np.linalg.norm((
        (pose_landmarks[11].x - pose_landmarks[12].x) * width,
        (pose_landmarks[11].y - pose_landmarks[12].y) * height,
    )))
    face_width = max(float(np.ptp(face_x)), shoulder_span * 0.18, 48.0)
    x0 = max(0, int(min(face_x) - face_width * 0.65))
    x1 = min(width, int(max(face_x) + face_width * 0.65))
    y0 = max(0, int(min(face_y) - face_width * 0.45))
    y1 = min(height, int(max(face_y) + face_width * 0.45))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return _face([]), False

    crop = image_rgb[y0:y1, x0:x1]
    scale = min(512 / crop.shape[1], 512 / crop.shape[0])
    resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                                max(1, round(crop.shape[0] * scale))),
                         interpolation=cv2.INTER_CUBIC)
    crop_result = face_model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=resized))
    if not crop_result.face_landmarks:
        return _face([]), False

    crop_height, crop_width = crop.shape[:2]
    points = np.zeros((MAX_FACE_POINTS, 4), dtype=np.float32)
    for index, point in enumerate(crop_result.face_landmarks[0][:MAX_FACE_POINTS]):
        points[index] = (
            (x0 + point.x * crop_width) / width,
            (y0 + point.y * crop_height) / height,
            point.z * crop_width / width,
            1.0,
        )
    return points, True


def palm_normal(hand_world: np.ndarray, handedness: str = "") -> np.ndarray:
    """Unit normal to the palm plane; preserves orientation without guessing facing labels."""
    wrist = hand_world[0]
    index_base = hand_world[5]
    pinky_base = hand_world[17]
    normal = np.cross(index_base - wrist, pinky_base - wrist)
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    # Index-to-pinky winding reverses between anatomical hands. Normalize its polarity
    # so the same palm pose has the same direction on either hand.
    side_sign = -1.0 if handedness.casefold() == "right" else 1.0
    return (normal / norm * side_sign).astype(np.float32)


def _detectors() -> tuple[Any, Any, Any]:
    assets = {
        "pose": MODEL_ROOT / "pose_landmarker_lite.task",
        "hands": MODEL_ROOT / "hand_landmarker.task",
        "face": MODEL_ROOT / "face_landmarker.task",
    }
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MediaPipe task model(s): " + ", ".join(missing))

    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(assets["pose"])),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    ))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(assets["hands"])),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    ))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(assets["face"])),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    ))
    return pose, hands, face


def _annotate(
    image: np.ndarray,
    pose_points: np.ndarray,
    face_points: np.ndarray,
    hands: np.ndarray,
    hand_present: np.ndarray,
    palm_directions: np.ndarray,
    face_connections: list[Any],
) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    for connection in face_connections:
        start, end = int(connection.start), int(connection.end)
        if start >= MAX_FACE_POINTS or end >= MAX_FACE_POINTS:
            continue
        first, second = face_points[start], face_points[end]
        if first[3] <= 0 or second[3] <= 0:
            continue
        p1 = (round(float(first[0]) * width), round(float(first[1]) * height))
        p2 = (round(float(second[0]) * width), round(float(second[1]) * height))
        cv2.line(out, p1, p2, (210, 170, 80), 1, cv2.LINE_AA)

    for start, end in POSE_EDGES:
        if pose_points[start, 3] < 0.35 or pose_points[end, 3] < 0.35:
            continue
        p1 = tuple(np.round(pose_points[start, :2] * (width, height)).astype(int))
        p2 = tuple(np.round(pose_points[end, :2] * (width, height)).astype(int))
        cv2.line(out, p1, p2, (50, 205, 235), 2, cv2.LINE_AA)
    for index in POSE_TRACK_POINTS:
        point = pose_points[index]
        if point[3] < 0.35:
            continue
        center = tuple(np.round(point[:2] * (width, height)).astype(int))
        cv2.circle(out, center, 4, (30, 230, 250), -1, cv2.LINE_AA)
        cv2.putText(out, f"P{index}", (center[0] + 4, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 45, 55), 1, cv2.LINE_AA)

    for slot in range(2):
        if not hand_present[slot]:
            continue
        for start, end in HAND_EDGES:
            p1 = tuple(np.round(hands[slot, start, :2] * (width, height)).astype(int))
            p2 = tuple(np.round(hands[slot, end, :2] * (width, height)).astype(int))
            cv2.line(out, p1, p2, (190, 80, 230), 2, cv2.LINE_AA)
        for point in hands[slot]:
            center = tuple(np.round(point[:2] * (width, height)).astype(int))
            cv2.circle(out, center, 3, (235, 190, 90), -1, cv2.LINE_AA)
        normal = palm_directions[slot]
        wrist = hands[slot, 0, :2]
        palm_center = (hands[slot, 5, :2] + hands[slot, 17, :2]) / 2
        origin = tuple(np.round((0.5 * wrist + 0.5 * palm_center) * (width, height)).astype(int))
        arrow = np.asarray((normal[0], -normal[1]), dtype=np.float32)
        arrow_norm = float(np.linalg.norm(arrow))
        if arrow_norm > 1e-5:
            end = (origin[0] + int(arrow[0] / arrow_norm * 36),
                   origin[1] + int(arrow[1] / arrow_norm * 36))
            cv2.arrowedLine(out, origin, end, (50, 80, 250), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(out, f"palm n=({normal[0]:+.2f},{normal[1]:+.2f},{normal[2]:+.2f})",
                    (origin[0] + 5, origin[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (25, 45, 95), 1, cv2.LINE_AA)
    return out


def _preview(path: Path, records: list[dict[str, Any]], preview_count: int) -> None:
    if preview_count <= 0 or not records:
        return
    columns = 4
    cell_width, cell_height = 360, 300
    page_size = 24
    path.parent.mkdir(parents=True, exist_ok=True)
    for page_number, start in enumerate(range(0, min(len(records), preview_count), page_size), start=1):
        page = records[start : start + page_size]
        rows = math.ceil(len(page) / columns)
        sheet = np.full((rows * cell_height, columns * cell_width, 3), (248, 246, 239), dtype=np.uint8)
        for offset, record in enumerate(page):
            image = record["annotated"]
            scale = min((cell_width - 12) / image.shape[1], (cell_height - 42) / image.shape[0])
            resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                         max(1, round(image.shape[0] * scale))),
                                 interpolation=cv2.INTER_AREA)
            row, col = divmod(offset, columns)
            x = col * cell_width + (cell_width - resized.shape[1]) // 2
            y = row * cell_height + 6
            sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            text = f'{record["label"]} | pose:{record["pose_detected"]} face:{record["face_detected"]} hands:{int(record["hand_present"].sum())}'
            cv2.putText(sheet, text[:48], (col * cell_width + 6, row * cell_height + cell_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 48, 58), 1, cv2.LINE_AA)
        out = path.with_name(f"{path.stem}_{page_number:02d}{path.suffix or '.jpg'}")
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def extract(image_root: Path, output_path: Path, report_dir: Path, limit: int | None,
            preview_count: int) -> dict[str, Any]:
    files = sorted(path for path in image_root.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        files = files[: max(0, limit)]
    if not files:
        raise FileNotFoundError(f"No supported images found in {image_root}")
    files_by_label: dict[str, list[Path]] = defaultdict(list)
    for image_path in files:
        files_by_label[image_path.parent.name].append(image_path)
    preview_paths: set[Path] = set()
    for rank in range(max((len(items) for items in files_by_label.values()), default=0)):
        for label in sorted(files_by_label):
            if rank < len(files_by_label[label]) and len(preview_paths) < max(0, preview_count):
                preview_paths.add(files_by_label[label][rank])

    pose_model, hand_model, face_model = _detectors()
    face_connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    frame_records: list[dict[str, Any]] = []
    preview_records: list[dict[str, Any]] = []
    class_counts: dict[str, Counter] = defaultdict(Counter)
    try:
        for image_path in files:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                class_counts[image_path.parent.name]["unreadable"] += 1
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            pose_result = pose_model.detect(mp_image)
            hand_result = hand_model.detect(mp_image)

            pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else []
            pose_world_landmarks = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else []
            pose_points = _pose(pose_landmarks)
            pose_world = _pose(pose_world_landmarks)
            face_points, face_crop_used = _detect_face(face_model, image_rgb, pose_landmarks)
            hands = np.zeros((2, 21, 3), dtype=np.float32)
            hand_world = np.zeros((2, 21, 3), dtype=np.float32)
            hand_present = np.zeros(2, dtype=bool)
            hand_scores = np.zeros(2, dtype=np.float32)
            handedness = np.full(2, "", dtype="U8")
            palm_directions = np.zeros((2, 3), dtype=np.float32)
            image_hands = hand_result.hand_landmarks or []
            world_hands = hand_result.hand_world_landmarks or []
            categories = hand_result.handedness or []
            for index, detected in enumerate(image_hands[:2]):
                hands[index] = _xyz(detected)
                side = categories[index][0].category_name if index < len(categories) and categories[index] else ""
                if index < len(world_hands):
                    hand_world[index] = _xyz(world_hands[index])
                    palm_directions[index] = palm_normal(hand_world[index], side)
                hand_present[index] = True
                if index < len(categories) and categories[index]:
                    handedness[index] = side
                    hand_scores[index] = categories[index][0].score

            label = image_path.parent.name
            relative_path = str(image_path.relative_to(image_root))
            pose_detected = bool(pose_landmarks)
            face_detected = bool(np.any(face_points[:, 3]))
            face_regions_detected = bool(pose_detected and all(pose_points[index, 3] >= 0.2
                                                               for index in (2, 5, 7, 8, 9, 10)))
            torso_detected = bool(pose_detected and min(pose_points[11, 3], pose_points[12, 3]) >= 0.35)
            hand_count = int(hand_present.sum())
            complete = (torso_detected and face_regions_detected and hand_count > 0
                        and bool(np.linalg.norm(palm_directions, axis=1).max() > 0))
            class_counts[label]["images"] += 1
            class_counts[label]["pose"] += int(pose_detected)
            class_counts[label]["face_mesh"] += int(face_detected)
            class_counts[label]["face_regions"] += int(face_regions_detected)
            class_counts[label]["torso"] += int(torso_detected)
            class_counts[label]["hands"] += int(hand_count > 0)
            class_counts[label]["complete"] += int(complete)
            record = {
                "label": label,
                "relative_path": relative_path,
                "image_width": int(image_bgr.shape[1]),
                "image_height": int(image_bgr.shape[0]),
                "pose": pose_points,
                "pose_world": pose_world,
                "face": face_points,
                "hands": hands,
                "hand_world": hand_world,
                "hand_present": hand_present,
                "hand_scores": hand_scores,
                "handedness": handedness,
                "palm_directions": palm_directions,
                "pose_detected": pose_detected,
                "face_detected": face_detected,
                "face_regions_detected": face_regions_detected,
                "torso_detected": torso_detected,
                "face_crop_used": face_crop_used,
                "hand_count": hand_count,
                "complete": complete,
            }
            frame_records.append(record)
            if image_path in preview_paths:
                annotated = _annotate(image_bgr, pose_points, face_points, hands,
                                      hand_present, palm_directions, face_connections)
                preview_records.append({**record, "annotated": annotated})
    finally:
        pose_model.close()
        hand_model.close()
        face_model.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        labels=np.asarray([record["label"] for record in frame_records], dtype=np.str_),
        relative_paths=np.asarray([record["relative_path"] for record in frame_records], dtype=np.str_),
        image_sizes=np.asarray([[record["image_width"], record["image_height"]] for record in frame_records], dtype=np.int32),
        pose=np.stack([record["pose"] for record in frame_records]),
        pose_world=np.stack([record["pose_world"] for record in frame_records]),
        face=np.stack([record["face"] for record in frame_records]),
        face_present=np.asarray([record["face_detected"] for record in frame_records], dtype=bool),
        face_regions_present=np.asarray([record["face_regions_detected"] for record in frame_records], dtype=bool),
        torso_present=np.asarray([record["torso_detected"] for record in frame_records], dtype=bool),
        hands=np.stack([record["hands"] for record in frame_records]),
        hand_world=np.stack([record["hand_world"] for record in frame_records]),
        hand_present=np.stack([record["hand_present"] for record in frame_records]),
        hand_scores=np.stack([record["hand_scores"] for record in frame_records]),
        handedness=np.stack([record["handedness"] for record in frame_records]),
        palm_directions=np.stack([record["palm_directions"] for record in frame_records]),
        meta=np.asarray(json.dumps({
            "schema": "mudra_static_tracking_v1",
            "image_root": str(image_root),
            "coordinate_spaces": {
                "pose": "normalized image x,y,z plus visibility",
                "pose_world": "MediaPipe world x,y,z plus visibility",
                "face": "normalized image x,y,z; 478-point face mesh when detected",
                "hands": "normalized image x,y,z; up to two hands x 21 landmarks",
                "hand_world": "MediaPipe world x,y,z; up to two hands x 21 landmarks",
                "palm_directions": "unit normal from wrist-index-mcp-pinky-mcp palm plane in hand-world coordinates, polarity canonicalized by detector hand side",
            },
            "torso_and_ears_eye_mouth_points": "pose indices 11-16, 23-24 and 2,5,7-10",
            "face_mesh": "MediaPipe FaceLandmarker contours include detailed eyes and lips; ears use PoseLandmarker points 7 and 8",
        })),
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "mudra_static_tracking_v1",
        "image_root": str(image_root),
        "images_tracked": len(frame_records),
        "pose_detected": sum(record["pose_detected"] for record in frame_records),
        "face_detected": sum(record["face_detected"] for record in frame_records),
        "pose_face_regions_detected": sum(record["face_regions_detected"] for record in frame_records),
        "torso_detected": sum(record["torso_detected"] for record in frame_records),
        "hands_detected": sum(record["hand_count"] > 0 for record in frame_records),
        "all_requested_regions_detected": sum(record["complete"] for record in frame_records),
        "classes": {
            label: {
                "images": counts["images"],
                "pose": counts["pose"],
                "face_mesh": counts["face_mesh"],
                "face_regions": counts["face_regions"],
                "torso": counts["torso"],
                "hands": counts["hands"],
                "complete": counts["complete"],
                "can_supply_4_train_1_test": counts["complete"] >= 5,
            }
            for label, counts in sorted(class_counts.items())
        },
        "limits": [
            "This step extracts and inspects landmarks only; it does not train a model.",
            "FaceLandmarker supplies detailed eyes and lip contours; PoseLandmarker supplies ear, eye, mouth, torso, and arm points.",
            "Palm direction is a 3D normal vector computed from hand world landmarks; its polarity is side-normalized, not a camera-facing probability.",
            "The 4-train/1-test rule is possible only for classes with at least five usable, non-duplicate source images.",
        ],
    }
    (report_dir / "tracking_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (report_dir / "tracking_by_image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("label", "relative_path", "pose_detected", "torso_detected",
                                                    "face_regions_detected", "face_mesh_detected", "face_crop_used",
                                                    "hand_count", "palm_direction_count", "complete"))
        writer.writeheader()
        for record in frame_records:
            writer.writerow({
                "label": record["label"],
                "relative_path": record["relative_path"],
                "pose_detected": int(record["pose_detected"]),
                "torso_detected": int(record["torso_detected"]),
                "face_regions_detected": int(record["face_regions_detected"]),
                "face_mesh_detected": int(record["face_detected"]),
                "face_crop_used": int(record["face_crop_used"]),
                "hand_count": record["hand_count"],
                "palm_direction_count": int(np.sum(np.linalg.norm(record["palm_directions"], axis=1) > 0)),
                "complete": int(record["complete"]),
            })
    _preview(report_dir / "tracking_preview.jpg", preview_records, preview_count)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_ROOT, help="image root with one label folder per sign")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="ignored local NPZ output for raw landmarks")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="ignored local CSV/JSON/preview output")
    parser.add_argument("--limit", type=int, default=None, help="optional first-N image limit for a quick smoke run")
    parser.add_argument("--preview-count", type=int, default=24, help="number of annotated sample images to put in preview sheets")
    args = parser.parse_args()
    report = extract(args.images, args.output, args.report_dir, args.limit, args.preview_count)
    print(json.dumps({
        "images_tracked": report["images_tracked"],
        "pose_detected": report["pose_detected"],
        "face_detected": report["face_detected"],
        "pose_face_regions_detected": report["pose_face_regions_detected"],
        "torso_detected": report["torso_detected"],
        "hands_detected": report["hands_detected"],
        "all_requested_regions_detected": report["all_requested_regions_detected"],
        "classes_with_5_usable_images": sum(item["can_supply_4_train_1_test"] for item in report["classes"].values()),
        "landmarks": str(args.output),
        "summary": str(args.report_dir / "tracking_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
