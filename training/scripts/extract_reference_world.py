"""Re-extract each word's "How to sign it" example with MediaPipe world landmarks (3D, metres).

The avatar (app/frontend/src/avatar/landmarkRetarget.js) aims its bones along the signer's limbs; image
landmarks give only a rough depth, world landmarks a real one. For every word whose example comes from a
video on this machine (team words, INCLUDE videos), the same video is run through HandLandmarker +
PoseLandmarker again and both image and world landmarks are saved:

    data/processed/landmarks/reference_world/<SIGN>.npz   (local, git-ignored like all landmark data)
        hands [T,2,21,3], hand_present [T,2], pose [T,33,4], pose_present [T]      image coordinates
        hands_world [T,2,21,3], pose_world [T,33,3]                                  metres
        timestamps_ms [T], fps, width, height, source_path

Hands are put in signer-left / signer-right slots exactly like the training data (assign_hands).

    python training/scripts/extract_reference_world.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from mudra_ml.config import project_path
from mudra_ml.preprocessing.frame_sampler import resample_step
from mudra_ml.preprocessing.sequence import assign_hands

sys.path.insert(0, str(project_path("app/backend")))
from mudra_api.references import ReferenceClips
from mudra_api.settings import Settings

ASSETS = project_path("training/assets/mediapipe")
VIDEO_SOURCES = ("islwords", "include_video")
MAX_FPS = 30.0


def landmarkers():
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ASSETS / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.5))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ASSETS / "pose_landmarker_full.task")),
        running_mode=vision.RunningMode.VIDEO, min_pose_detection_confidence=0.5))
    return hand, pose


def extract(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = resample_step(fps, MAX_FPS)
    hand_lm, pose_lm = landmarkers()
    rows = {k: [] for k in ("hands", "hand_present", "pose", "pose_present", "hands_world", "pose_world", "timestamps_ms")}
    idx, last_ts = 0, -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ts = max(round(idx * 1000.0 / fps), last_ts + 1)
                last_ts = ts
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                pr = pose_lm.detect_for_video(image, ts)
                pose = np.zeros((33, 4), np.float32)
                pose_w = np.zeros((33, 3), np.float32)
                pose_ok = bool(pr.pose_landmarks)
                if pose_ok:
                    pose[:] = [(p.x, p.y, p.z, p.visibility or 0.0) for p in pr.pose_landmarks[0]]
                    pose_w[:] = [(p.x, p.y, p.z) for p in pr.pose_world_landmarks[0]]
                hr = hand_lm.detect_for_video(image, ts)
                detected = [np.array([(p.x, p.y, p.z) for p in h], np.float32) for h in hr.hand_landmarks]
                detected_w = [np.array([(p.x, p.y, p.z) for p in h], np.float32) for h in hr.hand_world_landmarks]
                labels = [c[0].category_name for c in hr.handedness]
                scores = [float(c[0].score) for c in hr.handedness]
                hands, present, _ = assign_hands(detected, labels, scores, pose if pose_ok else None, False)
                hands_w = np.zeros((2, 21, 3), np.float32)
                for slot in range(2):  # the world landmarks of whichever detection went into this slot
                    if present[slot]:
                        k = next(i for i, d in enumerate(detected) if np.allclose(d, hands[slot]))
                        hands_w[slot] = detected_w[k]
                for key, value in (("hands", hands), ("hand_present", present), ("pose", pose), ("pose_present", pose_ok),
                                   ("hands_world", hands_w), ("pose_world", pose_w), ("timestamps_ms", ts)):
                    rows[key].append(value)
            idx += 1
    finally:
        hand_lm.close()
        pose_lm.close()
        cap.release()
    out = {k: np.asarray(v) for k, v in rows.items()}
    out.update(fps=np.float32(fps / step), width=np.int32(width), height=np.int32(height), source_path=np.array(str(path)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/processed/landmarks/reference_world")
    args = ap.parse_args()
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = ReferenceClips(Settings().reference_split)
    todo = [(sign, row) for sign, (_, row) in sorted(refs.best.items()) if row.source in VIDEO_SOURCES and Path(row.raw_path).exists()]
    if args.limit:
        todo = todo[: args.limit]
    t0 = time.time()
    for i, (sign, row) in enumerate(todo, 1):
        target = out_dir / f"{sign}.npz"
        if target.exists():
            continue
        data = extract(Path(row.raw_path))
        np.savez_compressed(target, **data)
        hands = float(data["hand_present"].any(axis=1).mean()) if len(data["hand_present"]) else 0.0
        print(f"[{i}/{len(todo)}] {sign}: {len(data['timestamps_ms'])} frames, hands in {hands:.0%} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {len(todo)} words -> {out_dir}")


if __name__ == "__main__":
    main()
