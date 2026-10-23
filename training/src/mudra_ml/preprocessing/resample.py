"""Inference-time clean-up of a live sign before the model sees it (no retraining involved).

Live cameras often deliver 5-12 frames per second and lose a hand for a frame or two in dim light, while the
model learned mostly from 25 fps clips. The model samples ``seq_len`` frames from a sign and also reads their
velocities, so a slow camera shows it repeated frames with zero velocity followed by jumps, and a lost hand
shows it a hand that vanishes and reappears. Both are fixed here by linear interpolation in time:

- gap filling (``sequence.fill_hand_gaps``): a hand missing for at most ``max_gap_s`` between two frames
  where it is seen is filled in;
- ``resample``: a sign recorded below ``below_fps`` is put on an even ``fps`` time grid (with gap filling).

Hands are never invented at the start or end of a sign, or across longer gaps (a hand really out of view).
"""

from __future__ import annotations

import numpy as np

from mudra_ml.preprocessing.sequence import LandmarkSequence, fill_hand_gaps

DEFAULT_FPS = 25.0


def frame_times(seq: LandmarkSequence) -> np.ndarray:
    """Frame times in seconds from the start (the clip's own fps, else 25 fps, when timestamps are missing)."""
    t = np.asarray(seq.timestamps_ms, np.float64) / 1000
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        fps = float(seq.meta.get("fps") or 0) or DEFAULT_FPS
        t = np.arange(seq.num_frames) / fps
    return t - t[0]


def _interp_hand(t: np.ndarray, grid: np.ndarray, pts: np.ndarray, seen: np.ndarray, max_gap_s: float,
                 tol: float) -> tuple[np.ndarray, np.ndarray]:
    """One hand's landmarks [n, 21, 3] at the grid times; present where it is seen (within tol) or inside a short gap."""
    out = np.zeros((len(grid),) + pts.shape[1:], np.float32)
    present = np.zeros(len(grid), bool)
    idx = np.flatnonzero(seen)
    if len(idx) == 0:
        return out, present
    ts = t[idx]
    j = np.searchsorted(ts, grid)  # first seen frame at or after each grid time
    for g, (time, k) in enumerate(zip(grid, j)):
        prev = k - 1 if k == len(ts) or ts[k] > time else k
        nxt = k if k < len(ts) else None
        if prev >= 0 and nxt is not None and ts[nxt] - ts[prev] <= max_gap_s:
            a = (time - ts[prev]) / (ts[nxt] - ts[prev]) if ts[nxt] > ts[prev] else 0.0
            out[g], present[g] = (1 - a) * pts[idx[prev]] + a * pts[idx[nxt]], True
        elif prev >= 0 and abs(ts[prev] - time) <= tol:  # edges and long gaps: only a frame this close counts
            out[g], present[g] = pts[idx[prev]], True
        elif nxt is not None and abs(ts[nxt] - time) <= tol:
            out[g], present[g] = pts[idx[nxt]], True
    return out, present


def resample(seq: LandmarkSequence, fps: float, max_gap_s: float) -> LandmarkSequence:
    """The sign on an even ``fps`` time grid; hands interpolated across gaps up to ``max_gap_s``."""
    t = frame_times(seq)
    if seq.num_frames < 2 or t[-1] <= 0:
        return seq
    grid = np.arange(0.0, t[-1] + 1e-9, 1.0 / fps)
    src_dt = float(np.median(np.diff(t)))
    tol = min(0.5 / fps, 0.5 * src_dt) + 1e-6  # a grid time this close to a frame takes that frame
    gap = max(max_gap_s, 1.05 * src_dt)  # neighbouring frames of a slow camera are never a "gap"
    hands = np.zeros((len(grid),) + seq.hands.shape[1:], np.float32)
    present = np.zeros((len(grid), seq.hands.shape[1]), bool)
    for h in range(seq.hands.shape[1]):
        hands[:, h], present[:, h] = _interp_hand(t, grid, seq.hands[:, h], seq.hand_present[:, h], gap, tol)
    pose, pose_present = _interp_hand(t, grid, seq.pose, seq.pose_present, gap, tol)
    if not pose_present.any() and seq.pose_present.any():
        pose_present[:] = False
    return LandmarkSequence(hands, present, present.astype(np.float32), pose.astype(np.float32), pose_present,
                            (seq.timestamps_ms[0] + grid * 1000).astype(np.int64), {**seq.meta, "fps": fps})


def clean(seq: LandmarkSequence, below_fps: float, fps: float, max_gap_s: float) -> LandmarkSequence:
    """``resample`` when the sign was recorded below ``below_fps``, else only ``fill_hand_gaps``."""
    t = frame_times(seq)
    rate = (seq.num_frames - 1) / t[-1] if seq.num_frames > 1 and t[-1] > 0 else fps
    if below_fps > 0 and rate < below_fps:
        return resample(seq, fps, max_gap_s)
    return fill_hand_gaps(seq, max_gap_s) if max_gap_s > 0 else seq
