import numpy as np

from mudra_ml.preprocessing.resample import clean, resample
from mudra_ml.preprocessing.sequence import LandmarkSequence


def _seq(times_s, xs, present=None):
    n = len(times_s)
    hands = np.zeros((n, 2, 21, 3), np.float32)
    hands[:, 1, :, 0] = np.asarray(xs, np.float32)[:, None]
    pres = np.zeros((n, 2), bool)
    pres[:, 1] = True if present is None else present
    pose = np.zeros((n, 33, 4), np.float32)
    pose[:, 11], pose[:, 12] = [0.6, 0.5, 0, 1], [0.4, 0.5, 0, 1]
    return LandmarkSequence(hands, pres, pres.astype(np.float32), pose, np.ones(n, bool),
                            (np.asarray(times_s) * 1000).astype(np.int64), {"width": 640, "height": 480})


def test_resample_puts_a_slow_camera_on_an_even_grid_with_interpolated_motion():
    t = np.arange(0, 1.01, 1 / 6)  # a 6 fps camera
    out = resample(_seq(t, t), 25.0, 0.3)  # the hand moves linearly: x = t
    assert out.num_frames == 26 and out.hand_present[:, 1].all()
    grid = out.timestamps_ms / 1000 - out.timestamps_ms[0] / 1000
    assert np.allclose(out.hands[:, 1, 0, 0], grid, atol=2e-3)  # in between frames follow the motion
    assert out.meta["fps"] == 25.0 and out.pose_present.all()


def test_resample_never_invents_a_hand_across_a_long_gap_or_before_it_appears():
    t = np.arange(0, 2.01, 0.1)  # 10 fps
    seen = (t < 0.5) | (t > 1.5)  # out of view for a second
    seen[:2] = False  # and not there at the start
    out = resample(_seq(t, np.ones_like(t), seen), 25.0, 0.3)
    grid = out.timestamps_ms / 1000
    assert not out.hand_present[(grid > 0.6) & (grid < 1.4), 1].any()
    assert not out.hand_present[grid < 0.15, 1].any()


def test_clean_fills_short_gaps_and_resamples_only_slow_clips():
    t = np.arange(0, 1.0, 1 / 25)
    seen = np.ones(len(t), bool)
    seen[10:13] = False  # lost for 3 frames (0.12 s)
    fast = clean(_seq(t, t, seen), 18.0, 25.0, 0.3)
    assert fast.num_frames == len(t) and fast.hand_present[:, 1].all()  # same frames, gap filled
    slow = clean(_seq(np.arange(0, 1.0, 1 / 8), np.zeros(8)), 18.0, 25.0, 0.3)
    assert slow.num_frames > 8  # below 18 fps: resampled
