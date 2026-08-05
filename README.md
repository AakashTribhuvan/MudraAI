# ISL Hand and Pose Tracking for Dynamic Sign Recognition

This project collects pose and hand landmarks for Indian Sign Language (ISL) using MediaPipe and stores training-ready feature sequences for dynamic gesture recognition.

## What the project does

The workspace contains two main modes:

- `v1/hand_tracker.py`: real-time webcam capture, landmark visualization, smoothing, sequence buffering, and `.npy` saving.
- `landmark_extractor.py`: batch video processing for `.mp4` files, per-frame pose + hands extraction, normalization, velocity features, and metadata output.

## Current pipeline

1. Open a webcam stream or select a folder of `.mp4` videos.
2. Detect pose landmarks using MediaPipe Pose.
3. Detect hand landmarks using MediaPipe Hand Landmarker.
4. Smooth landmark positions over time.
5. Normalize pose and hand positions relative to the shoulders.
6. Fill missing detections using the last known landmark values or zero vectors after a timeout.
7. Build a feature vector per frame.
8. Append velocity features to capture motion between frames.
9. Save extracted sequences as NumPy arrays and write metadata JSON.

## Feature vector structure

The extractor builds a per-frame input vector of size `225` and then appends a velocity vector of size `225` for a total feature size of `450`.

- Pose landmarks: 33 × 3 = 99 values
- Left-hand landmarks: 21 × 3 = 63 values
- Right-hand landmarks: 21 × 3 = 63 values
- Velocity features: 225 values

Total per frame:

- `450` values

## Key features implemented

### Real-time tracking (`v1/hand_tracker.py`)

- Live pose and hand tracking from webcam
- Landmark smoothing for pose and hands
- Drawing of arm connections and hand skeletons
- Sequence buffering with frame count display
- Save sequence buffer as `.npy` when pressing `s`
- Clear buffer with `r`
- Exit with `Esc`

### Batch video extraction (`landmark_extractor.py`)

- Supports input folder or single `.mp4` file
- Uses MediaPipe to detect pose and hand landmarks from video frames
- Normalizes landmarks based on shoulder distance
- Tracks missing hand detection for a short tolerance window
- Builds left/right hand separation in the feature vector
- Appends velocity features to represent motion
- Saves sequence arrays to `training_data/extracted`
- Writes per-video metadata JSON files
- Optional `--label` support to store labels in `labels.json`

## Controls and usage

### Run real-time tracker

```bash
python v1/hand_tracker.py
```

### Run video landmark extraction

```bash
python landmark_extractor.py <input_path> [--output-dir OUTPUT_DIR] [--max-frames N] [--skip-frames N] [--no-display] [--label LABEL]
```

Example:

```bash
python landmark_extractor.py training_data/Sample\ Videos --output-dir training_data/extracted --skip-frames 2 --label hello
```

## Output format

### `v1/hand_tracker.py`

Saved sequences are `.npy` arrays with shape:

```python
(T, 225)
```

where `T` = number of frames collected.

### `landmark_extractor.py`

Saved sequences are `.npy` arrays with shape:

```python
(T, 450)
```

where `T` = number of frames processed, and `450` includes the original landmark features plus frame-to-frame velocity.

Each extracted sequence also produces a `.json` metadata file containing:

- `source_video`
- `relative_path`
- `shape`
- `frame_count`
- `feature_size`

## Project files

- `landmark_extractor.py`: video-based landmark extraction and sequence saving
- `v1/hand_tracker.py`: live webcam tracker and real-time sequence recording
- `v1/models/`: MediaPipe `.task` models for pose and hand tracking
- `training_data/`: saved sequences and sample video inputs
- `training_data/extracted/labels/`: optional label metadata path

## Requirements

The project depends on:

- Python 3
- OpenCV
- NumPy
- MediaPipe

Install with pip if needed:

```bash
pip install opencv-python numpy mediapipe
```

## Notes

- `landmark_extractor.py` now supports batch processing of `.mp4` videos and produces normalized, velocity-augmented feature vectors for sign recognition training.
- The live tracker can still record gesture sequences interactively, while the extractor is better for large annotated video datasets.
- Make sure the `v1/models/` folder contains `hand_landmarker.task` and `pose_landmarker_lite.task` before running either script.
