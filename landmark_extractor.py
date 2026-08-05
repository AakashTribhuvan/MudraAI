import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "v1" / "models"
DEFAULT_INPUT_PATH = Path("D:/Coding/ISL/training_data/Sample Videos").expanduser().resolve()

HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
POSITION_FEATURE_SIZE = 225
FEATURE_SIZE = POSITION_FEATURE_SIZE * 2
MAX_MISSING_HAND_FRAMES = 3


class LandmarkSmoother:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.previous = None

    def update(self, current):
        current = np.array(current, dtype=np.float32)
        if self.previous is None:
            self.previous = current
            return current

        smoothed = self.alpha * current + (1 - self.alpha) * self.previous
        self.previous = smoothed
        return smoothed


def initialize_landmarkers():
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    return hand_landmarker, pose_landmarker


def get_xyz(landmark):
    if hasattr(landmark, "x") and hasattr(landmark, "y") and hasattr(landmark, "z"):
        return landmark.x, landmark.y, landmark.z
    return landmark[0], landmark[1], landmark[2]


def get_pixel_coordinates(landmark, width, height):
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        x = landmark.x
        y = landmark.y
    else:
        x = landmark[0]
        y = landmark[1]

    return int(x * width), int(y * height)


def draw_connection(frame, landmark_a, landmark_b, width, height):
    point_a = get_pixel_coordinates(landmark_a, width, height)
    point_b = get_pixel_coordinates(landmark_b, width, height)
    cv2.line(frame, point_a, point_b, (255, 0, 0), 3)


def draw_landmark(frame, landmark, width, height):
    point = get_pixel_coordinates(landmark, width, height)
    cv2.circle(frame, point, 8, (0, 255, 0), -1)


def build_hand_vector(raw_landmarks, previous_vector=None):
    vector = np.zeros(63, dtype=np.float32)
    if raw_landmarks is None:
        if previous_vector is not None:
            return previous_vector.astype(np.float32)
        return vector

    for index in range(21):
        if index < len(raw_landmarks):
            x, y, z = get_xyz(raw_landmarks[index])
        else:
            if previous_vector is not None:
                x, y, z = previous_vector[index * 3:index * 3 + 3]
            else:
                x, y, z = 0.0, 0.0, 0.0
        vector[index * 3:index * 3 + 3] = [x, y, z]

    return vector


def build_pose_vector(raw_pose, previous_vector=None):
    vector = np.zeros(99, dtype=np.float32)
    if raw_pose is None:
        if previous_vector is not None:
            return previous_vector.astype(np.float32)
        return vector

    for index in range(33):
        if index < len(raw_pose):
            x, y, z = get_xyz(raw_pose[index])
        else:
            if previous_vector is not None:
                x, y, z = previous_vector[index * 3:index * 3 + 3]
            else:
                x, y, z = 0.0, 0.0, 0.0
        vector[index * 3:index * 3 + 3] = [x, y, z]

    return vector


def build_feature_vector(pose_vector, left_hand_vector, right_hand_vector):
    return np.concatenate(
        [
            pose_vector.astype(np.float32),
            left_hand_vector.astype(np.float32),
            right_hand_vector.astype(np.float32),
        ]
    ).astype(np.float32)


def normalize_landmarks(landmarks, origin, scale):
    if landmarks is None:
        return None

    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)
    return ((points - origin) / scale).astype(np.float32)


def get_body_normalization(pose_landmarks, previous_origin=None, previous_scale=1.0):
    if pose_landmarks is None or len(pose_landmarks) <= RIGHT_SHOULDER:
        origin = previous_origin if previous_origin is not None else np.zeros(3, dtype=np.float32)
        return origin, previous_scale

    left_shoulder = np.asarray(pose_landmarks[LEFT_SHOULDER], dtype=np.float32)
    right_shoulder = np.asarray(pose_landmarks[RIGHT_SHOULDER], dtype=np.float32)
    origin = (left_shoulder + right_shoulder) / 2.0
    scale = float(np.linalg.norm(left_shoulder - right_shoulder))
    return origin, scale if scale >= 1e-6 else previous_scale


def get_hand_label(hand_result, index):
    handedness = getattr(hand_result, "handedness", None)
    if handedness and index < len(handedness):
        first = handedness[index][0] if len(handedness[index]) > 0 else None
        label = getattr(first, "label", None)
        if label is not None:
            return label.lower()
    return None


def extract_sequence_from_video(
    video_path,
    hand_landmarker,
    pose_landmarker,
    show_video=True,
    max_frames=None,
    skip_frames=1,
    output_dir=None,
    timestamp_offset_ms=0,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval_ms = int(1000 / fps) if fps > 0 else 33

    pose_smoother = LandmarkSmoother(alpha=0.7)
    hand_smoothers = {
        "left": LandmarkSmoother(alpha=0.7),
        "right": LandmarkSmoother(alpha=0.7),
    }

    prev_pose_vector = None
    prev_left_hand_vector = None
    prev_right_hand_vector = None
    previous_origin = np.zeros(3, dtype=np.float32)
    previous_scale = 1.0
    missing_hand_frames = {"left": 0, "right": 0}
    previous_feature_vector = None

    sequence_frames = []
    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_counter % skip_frames != 0:
            frame_counter += 1
            continue

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = timestamp_offset_ms + int(frame_counter * frame_interval_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        if pose_result.pose_landmarks:
            raw_pose = pose_result.pose_landmarks[0]
            current_pose = []
            for landmark in raw_pose:
                current_pose.append([landmark.x, landmark.y, landmark.z])
            smoothed_pose = pose_smoother.update(current_pose)

            previous_origin, previous_scale = get_body_normalization(
                smoothed_pose, previous_origin, previous_scale
            )
            normalized_pose = normalize_landmarks(
                smoothed_pose, previous_origin, previous_scale
            )
            prev_pose_vector = build_pose_vector(normalized_pose, prev_pose_vector)

            left_shoulder = smoothed_pose[LEFT_SHOULDER]
            left_elbow = smoothed_pose[LEFT_ELBOW]
            left_wrist = smoothed_pose[LEFT_WRIST]
            right_shoulder = smoothed_pose[RIGHT_SHOULDER]
            right_elbow = smoothed_pose[RIGHT_ELBOW]
            right_wrist = smoothed_pose[RIGHT_WRIST]

            draw_connection(frame, left_shoulder, left_elbow, width, height)
            draw_connection(frame, left_elbow, left_wrist, width, height)
            draw_connection(frame, right_shoulder, right_elbow, width, height)
            draw_connection(frame, right_elbow, right_wrist, width, height)

            for landmark in [left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist]:
                draw_landmark(frame, landmark, width, height)

        left_hand_vector = prev_left_hand_vector if prev_left_hand_vector is not None else np.zeros(63, dtype=np.float32)
        right_hand_vector = prev_right_hand_vector if prev_right_hand_vector is not None else np.zeros(63, dtype=np.float32)

        detected_sides = set()
        for hand_idx, raw_hand_landmarks in enumerate(hand_result.hand_landmarks):
            current_hand = []
            for landmark in raw_hand_landmarks:
                current_hand.append([landmark.x, landmark.y, landmark.z])

            hand_side = get_hand_label(hand_result, hand_idx)
            if hand_side is None:
                hand_side = "left" if hand_idx == 0 else "right"
            if hand_side not in hand_smoothers:
                continue

            detected_sides.add(hand_side)
            smoothed_hand = hand_smoothers[hand_side].update(current_hand)
            normalized_hand = normalize_landmarks(
                smoothed_hand, previous_origin, previous_scale
            )

            if hand_side == "left":
                left_hand_vector = build_hand_vector(normalized_hand, prev_left_hand_vector)
            else:
                right_hand_vector = build_hand_vector(normalized_hand, prev_right_hand_vector)

            for landmark in smoothed_hand:
                draw_landmark(frame, landmark, width, height)

            hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17),
            ]
            for start_index, end_index in hand_connections:
                draw_connection(frame, smoothed_hand[start_index], smoothed_hand[end_index], width, height)

        for hand_side in hand_smoothers:
            if hand_side in detected_sides:
                missing_hand_frames[hand_side] = 0
                continue

            missing_hand_frames[hand_side] += 1
            if missing_hand_frames[hand_side] > MAX_MISSING_HAND_FRAMES:
                if hand_side == "left":
                    left_hand_vector = np.zeros(63, dtype=np.float32)
                else:
                    right_hand_vector = np.zeros(63, dtype=np.float32)

        prev_left_hand_vector = left_hand_vector
        prev_right_hand_vector = right_hand_vector

        feature_vector = build_feature_vector(
            prev_pose_vector if prev_pose_vector is not None else np.zeros(99, dtype=np.float32),
            prev_left_hand_vector if prev_left_hand_vector is not None else np.zeros(63, dtype=np.float32),
            prev_right_hand_vector if prev_right_hand_vector is not None else np.zeros(63, dtype=np.float32),
        )
        velocity_vector = (
            np.zeros(POSITION_FEATURE_SIZE, dtype=np.float32)
            if previous_feature_vector is None
            else feature_vector - previous_feature_vector
        )
        feature_vector = np.concatenate([feature_vector, velocity_vector]).astype(np.float32)
        sequence_frames.append(feature_vector)
        previous_feature_vector = feature_vector[:POSITION_FEATURE_SIZE].copy()

        if show_video:
            cv2.putText(frame, f"Frame {frame_counter}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Features: {feature_vector.shape[0]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow(video_path.name, frame)
            if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                break

        frame_counter += 1
        if max_frames is not None and frame_counter >= max_frames:
            break

    cap.release()
    if show_video:
        cv2.destroyAllWindows()

    if not sequence_frames:
        return np.empty((0, FEATURE_SIZE), dtype=np.float32)

    return np.stack(sequence_frames, axis=0).astype(np.float32)


def discover_videos(input_path):
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() != ".mp4":
            raise ValueError("The input file must be an .mp4 video")
        return [path]

    return sorted(path.rglob("*.mp4"))


def normalize_sequence(sequence):
    return sequence.astype(np.float32, copy=True)


def get_video_duration_ms(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int((frame_count / fps) * 1000) if fps > 0 else 0


def save_sequence(sequence, video_path, output_dir, input_root):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    relative_path = video_path.relative_to(input_root) if input_root in video_path.parents else video_path.name
    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    normalized_sequence = normalize_sequence(sequence)

    stem = video_path.stem
    save_path = output_dir / f"{stem}.npy"
    np.save(save_path, normalized_sequence)

    metadata = {
        "source_video": str(video_path),
        "relative_path": str(relative_path),
        "shape": list(normalized_sequence.shape),
        "frame_count": int(normalized_sequence.shape[0]),
        "feature_size": int(normalized_sequence.shape[1]) if normalized_sequence.ndim == 2 else 0,
    }
    metadata_path = save_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved sequence to: {save_path}")
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Extract pose and hand landmarks from MP4 videos")
    parser.add_argument("input_path", nargs="?", default=str(DEFAULT_INPUT_PATH), help="Path to a folder or a single .mp4 file")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "training_data" / "extracted"), help="Where to save the extracted sequences")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames per video")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame for speed")
    parser.add_argument("--no-display", action="store_true", help="Disable video display while processing")
    parser.add_argument("--label", default=None, help="Optional label to associate with the extracted sequences")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.skip_frames < 1:
        raise ValueError("--skip-frames must be at least 1")
    input_path = Path(args.input_path).expanduser().resolve()
    videos = discover_videos(input_path)

    if not videos:
        print("No MP4 files found in the provided path.")
        return

    hand_landmarker, pose_landmarker = initialize_landmarkers()

    timestamp_offset_ms = 0
    
    try:
        for video_path in videos:

            sequence = extract_sequence_from_video(
                video_path,
                hand_landmarker,
                pose_landmarker,
                show_video=not args.no_display,
                max_frames=args.max_frames,
                skip_frames=args.skip_frames,
                timestamp_offset_ms=timestamp_offset_ms,
            )

            timestamp_offset_ms += get_video_duration_ms(video_path) + 1000
            if sequence.size:
                save_sequence(sequence, video_path, args.output_dir, input_path)
                if args.label is not None:
                    label_file = Path(args.output_dir) / "labels.json"
                    labels = {}
                    if label_file.exists():
                        labels = json.loads(label_file.read_text())
                    labels[video_path.stem] = args.label
                    label_file.write_text(json.dumps(labels, indent=2))
    finally:
        hand_landmarker.close()
        pose_landmarker.close()


if __name__ == "__main__":
    main()
