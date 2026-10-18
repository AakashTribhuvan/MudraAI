"""Continuous-stream benchmark: how the app behaves on a live camera.

Held-out clips are joined into streams (smooth 0.3-1.0 s transitions, plus 2-4 s idle
stretches every few signs), with known sign intervals. Two systems are scored:

  baseline  current app logic: 4 s rolling window every 0.25 s, commit after two equal
            confident predictions, reset after 0.8 s without hands
  segment   SignSegmenter (resting -> signing -> ended) + one whole-sign classification

Metrics per source: correct / wrong / missed per true sign, extra words per sign,
false triggers per idle minute, delay from sign end to word.

    python training/scripts/eval_continuous.py --split .../val.csv --tune      # choose segmenter settings
    python training/scripts/eval_continuous.py --split .../test.csv            # report with saved settings
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mudra_ml import schema
from mudra_ml.config import project_path
from mudra_ml.data.manifest import read_manifest
from mudra_ml.datasets.camera import CloseupParams, DeskParams, closeup_view, desk_view
from mudra_ml.datasets.distractors import synth_fidget
from mudra_ml.preprocessing.sequence import LandmarkSequence, load_sequence
from mudra_ml.streaming import Frame, ModelBundle, SegmenterConfig, StreamingRecognizer

CFG_PATH = project_path("training/configs/streaming.json")


def rest_in_view_frame(s, rng):
    """Hands resting still in front of the chest (in view), as people do at a desk between words.

    Hand shapes come from the clip's last detected hands; wrists sit 0.75 shoulder widths below each
    shoulder, 30% towards the middle. Returns (hands, present, pose, pose_present) or None.
    """
    w, h = int(s.meta.get("width", 0)) or 1, int(s.meta.get("height", 0)) or 1
    if not s.pose_present.any():
        return None
    pose = s.pose[np.flatnonzero(s.pose_present)[-1]].copy()
    shoulders = {0: pose[schema.POSE_LEFT_SHOULDER, :2], 1: pose[schema.POSE_RIGHT_SHOULDER, :2]}
    sw = float(np.linalg.norm((shoulders[0] - shoulders[1]) * [w, h]))
    centre = (shoulders[0] + shoulders[1]) / 2
    hands = np.zeros((2, 21, 3), np.float32)
    present = np.zeros(2, bool)
    for hand, (wrist_idx, elbow_idx) in ((0, (schema.POSE_LEFT_WRIST, schema.POSE_LEFT_ELBOW)),
                                         (1, (schema.POSE_RIGHT_WRIST, schema.POSE_RIGHT_ELBOW))):
        seen = np.flatnonzero(s.hand_present[:, hand])
        if not len(seen):
            continue
        shape = s.hands[seen[-1], hand] - s.hands[seen[-1], hand, schema.WRIST]
        target = shoulders[hand] + (centre - shoulders[hand]) * 0.3 + [0, 0.75 * sw / h]
        target = target + rng.normal(0, 0.01, 2)
        if not (0 <= target[0] <= 1 and 0 <= target[1] <= 1):
            continue
        hands[hand] = shape
        hands[hand, :, :2] += target
        present[hand] = True
        pose[wrist_idx, :2] = target
        pose[elbow_idx, :2] = (target + shoulders[hand]) / 2 + [0, 0.3 * sw / h]
    if not present.any():
        return None
    return hands, present, pose, True


def build_streams(rows, classes, fps, signs_per_stream=8, seed=0, fidgets=True, closeup=False, rest_in_view=False, desk=False):
    """closeup: every stream is seen through one simulated close-up webcam (datasets/camera.py).
    rest_in_view: between signs the hands rest still in view instead of dropping out of the frame.
    desk: close-up, and the hands rest in view between signs (camera.desk_view; the live failure of 2026-10-07/12)."""
    rng = np.random.default_rng(seed)
    rows = [r for r in rows if r.label in classes]
    order = rng.permutation(len(rows))
    streams = []
    for k in range(0, len(order), signs_per_stream):
        hands, present, pose, pose_p, truth, idle = [], [], [], [], [], []
        prev, prev_seq = None, None
        cam = CloseupParams.sample(rng) if closeup or desk else None
        desk_rest = (DeskParams.sample(rng), {}) if desk else None
        for j, idx in enumerate(order[k : k + signs_per_stream]):
            r = rows[idx]
            s = load_sequence(Path(r.landmarks_path))
            if cam is not None:
                s = closeup_view(s, cam)
                if s is None:
                    continue
            if desk_rest is not None:
                s = desk_view(s, desk_rest[0], rng, desk_rest[1])
            if prev is not None:  # smooth transition: interpolate last frame of previous clip -> first frame of this one
                n = int(rng.uniform(0.3, 1.0) * fps)
                if rest_in_view and j % 3 != 0:  # short pause with the hands held still in view
                    rv = rest_in_view_frame(prev_seq, rng)
                    if rv is not None:
                        prev = (rv[0], rv[1], rv[2], rv[3])
                        for _ in range(int(rng.uniform(0.6, 1.5) * fps)):
                            jit = rng.normal(0, 0.004, prev[0].shape).astype(np.float32)
                            hands.append(prev[0] + jit * prev[1][:, None, None]); present.append(prev[1]); pose.append(prev[2]); pose_p.append(prev[3])
                if j % 3 == 0:  # idle stretch: rest, often with a fidget (face touch / hair), then transition
                    hold = int(rng.uniform(2.0, 4.0) * fps)
                    rv = rest_in_view_frame(prev_seq, rng) if rest_in_view else None
                    if rv is not None:
                        prev = (rv[0], rv[1], rv[2], rv[3])
                    i0 = len(hands)
                    for _ in range(hold):
                        jit = rng.normal(0, 0.004 if rest_in_view else 0.002, prev[0].shape).astype(np.float32)
                        hands.append(prev[0] + jit * prev[1][:, None, None]); present.append(prev[1]); pose.append(prev[2]); pose_p.append(prev[3])
                    if fidgets and rng.random() < 0.7:
                        f = synth_fidget(prev_seq, rng, length=int(rng.uniform(0.8, 1.6) * fps))
                        k = i0 + int(rng.integers(0, max(1, hold - len(f.hands))))
                        for m in range(min(len(f.hands), len(hands) - k)):
                            hands[k + m], present[k + m], pose[k + m] = f.hands[m], f.hand_present[m], f.pose[m]
                    idle.append((i0, len(hands)))
                for a in np.linspace(0, 1, n + 2)[1:-1]:
                    both = prev[1] & s.hand_present[0]
                    h = ((1 - a) * prev[0] + a * s.hands[0]) * both[:, None, None]
                    hands.append(h.astype(np.float32)); present.append(both)
                    pose.append(((1 - a) * prev[2] + a * s.pose[0]).astype(np.float32)); pose_p.append(prev[3] and s.pose_present[0])
            start = len(hands)
            hands += list(s.hands); present += list(s.hand_present); pose += list(s.pose); pose_p += list(s.pose_present)
            truth.append((start / fps, len(hands) / fps, r.label))
            prev = (s.hands[-1], s.hand_present[-1], s.pose[-1], bool(s.pose_present[-1]))
            prev_seq = s
            meta = s.meta
        for _ in range(int(1.5 * fps)):  # rest after the last sign so the segmenter can close it
            jit = rng.normal(0, 0.002, prev[0].shape).astype(np.float32)
            hands.append(prev[0] + jit * prev[1][:, None, None]); present.append(prev[1]); pose.append(prev[2]); pose_p.append(prev[3])
        streams.append({"hands": np.stack(hands), "present": np.stack(present), "pose": np.stack(pose),
                        "pose_present": np.array(pose_p), "truth": truth, "idle": [(a / fps, b / fps) for a, b in idle],
                        "width": int(meta["width"]), "height": int(meta["height"])})
    return streams


def subsample(st, src_fps, cap):
    """The stream as a slower camera would deliver it (frame rate capped at ``cap``)."""
    if cap is None or cap >= src_fps:
        return st, src_fps
    idx = np.unique(np.round(np.arange(0, len(st["hands"]), src_fps / cap)).astype(int))
    idx = idx[idx < len(st["hands"])]
    out = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(st["hands"]) else v) for k, v in st.items()}
    return out, src_fps * len(idx) / len(st["hands"])


REASONS: defaultdict = defaultdict(int)  # why segments ended (rest / still / max_length), for the report
AUTO_ADD = None  # set by --auto-add: also commit an unsure sign's best guess at this confidence (the app's policy)


def run_segmenter(bundle, cfg, st, fps):
    rec = StreamingRecognizer(bundle, SegmenterConfig(**{**cfg.to_dict(), "provisional_s": 1e9}), st["width"], st["height"])
    words = []
    for i in range(len(st["hands"])):
        ev = rec.push(Frame(i / fps, st["hands"][i], st["present"][i], st["pose"][i], bool(st["pose_present"][i])))
        if not ev or ev.get("event") != "sign_ended":
            continue
        REASONS[ev.get("stats", {}).get("reason", "none")] += 1
        res = ev["result"]
        if res["status"] == "ok":
            words.append((ev["t0"], ev["t1"], i / fps, res["sign"]))
        elif (AUTO_ADD is not None and res["status"] in ("uncertain", "no_sign") and res["top_k"]
              and res["top_k"][0]["confidence"] >= AUTO_ADD and ev["stats"]["hand_rate"] >= 0.5):
            words.append((ev["t0"], ev["t1"], i / fps, res["top_k"][0]["sign"]))
    return words


def run_baseline(bundle, st, fps, window_s=4.0, stride_s=0.25):
    n, win, stride = len(st["hands"]), round(window_s * fps), round(stride_s * fps)
    ends = list(range(12, n + 1, stride))
    segs, kept = [], []
    for e in ends:
        a = max(0, e - win)
        if st["present"][a:e].any(axis=1).mean() >= 0.4:
            segs.append(LandmarkSequence(st["hands"][a:e], st["present"][a:e], st["present"][a:e].astype(np.float32),
                                         st["pose"][a:e], st["pose_present"][a:e], np.arange(e - a), {}))
            kept.append(e)
    probs = bundle.probs(segs, st["width"], st["height"]) if segs else []
    words, prev, last_commit, last_hand = [], None, None, 0
    kept_set = dict(zip(kept, range(len(kept))))
    for e in ends:
        if st["present"][e - 1].any():
            last_hand = e
        elif (e - last_hand) / fps > 0.8:
            last_commit, prev = None, None
        if e not in kept_set:
            continue
        d = bundle.decide(probs[kept_set[e]])
        if d["status"] == "ok":
            if d["sign"] == prev and d["sign"] != last_commit:
                words.append((max(0, e - win) / fps, e / fps, e / fps, d["sign"]))
                last_commit = d["sign"]
            prev = d["sign"]
        else:
            prev = None
    return words


def score(streams, outputs):
    c = defaultdict(float)
    delays = []
    for st, words in zip(streams, outputs):
        matched = set()
        for t0, t1, t_commit, sign in words:
            overlaps = [(min(t1, e) - max(t0, s), k) for k, (s, e, _) in enumerate(st["truth"])]
            ov, k = max(overlaps)
            if ov <= 0 or k in matched:
                in_idle = any(a <= t_commit <= b for a, b in st["idle"])
                c["idle_false" if in_idle else "extra"] += 1
                continue
            matched.add(k)
            c["correct" if sign == st["truth"][k][2] else "wrong"] += 1
            delays.append(t_commit - st["truth"][k][1])
        c["signs"] += len(st["truth"])
        c["idle_min"] += sum(b - a for a, b in st["idle"]) / 60
    n = c["signs"]
    return {"signs": int(n), "correct": c["correct"] / n, "wrong": c["wrong"] / n,
            "missed": (n - c["correct"] - c["wrong"]) / n, "extra_per_sign": c["extra"] / n,
            "idle_false_per_min": c["idle_false"] / max(c["idle_min"], 1e-9),
            "median_delay_s": float(np.median(delays)) if delays else None}


def objective(m):
    return m["correct"] - 3 * m["wrong"] - 3 * m["extra_per_sign"] - 0.5 * m["idle_false_per_min"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="app/backend/models/isl_mudra_combined_v9_bilstm")
    ap.add_argument("--split", required=True)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-threshold-only", action="store_true", help="keep saved segmenter settings, tune only the threshold")
    ap.add_argument("--default-fps", type=float, default=25.0)
    ap.add_argument("--config-out", default=None, help="where tuned settings are saved (default training/configs/streaming.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--closeup", action="store_true", help="view every stream through a simulated close-up webcam")
    ap.add_argument("--scenarios", default=None,
                    help="comma list of normal, closeup, closeup_rest (close-up + hands resting in view), "
                         "desk (close-up + hands never leave the frame); "
                         "tuning optimises the mean over all of them")
    ap.add_argument("--fps-cap", type=float, default=None, help="simulate a camera delivering at most this many frames/s")
    ap.add_argument("--grid", default=None, help='JSON segmenter grid for --tune, e.g. {"still_s": [0.5, 0.8]}')
    ap.add_argument("--set", default=None, help='JSON settings applied on top of the saved ones, e.g. {"still_s": 0.6}')
    ap.add_argument("--no-baseline", action="store_true", help="skip the old rolling-window baseline")
    ap.add_argument("--auto-add", type=float, default=None, help="score the app's auto-add of unsure signs at this confidence")
    args = ap.parse_args()
    global AUTO_ADD
    AUTO_ADD = args.auto_add
    if (args.tune or args.tune_threshold_only) and "test" in Path(args.split).stem.lower():
        # settings chosen here are written to the live config (and used for training crops): never from test
        ap.error("--tune on a test split is not allowed; tune on the validation split")
    scenarios = (args.scenarios.split(",") if args.scenarios else ["closeup" if args.closeup else "normal"])

    bundle = ModelBundle(project_path(args.model))
    rows = read_manifest(project_path(args.split))
    by_src = defaultdict(list)
    for r in rows:
        by_src[r.source].append(r)
    data = {}
    for src, rs in by_src.items():
        fps = float(load_sequence(Path(rs[0].landmarks_path)).meta.get("fps") or 0) or args.default_fps
        for sc in scenarios:
            sts = build_streams(rs, set(bundle.classes), fps, closeup=sc.startswith("closeup"),
                                rest_in_view=sc == "closeup_rest", desk=sc == "desk")
            capped = [subsample(st, fps, args.fps_cap) for st in sts]
            name = src if len(scenarios) == 1 else f"{src}/{sc}"
            data[name] = (capped[0][1] if capped else fps, [c[0] for c in capped])

    cfg_path = project_path(args.config_out) if args.config_out else CFG_PATH
    cfg = SegmenterConfig.load(cfg_path if cfg_path.exists() else CFG_PATH)
    if args.set:
        cfg = SegmenterConfig(**{**cfg.to_dict(), **json.loads(args.set)})
    if args.tune_threshold_only:
        args.tune = True
    if args.tune:
        grid = ({} if args.tune_threshold_only else json.loads(args.grid) if args.grid else
                {"raise_y": [0.5, 0.75, 1.0], "speed": [2.0, 3.0, 4.0], "on_s": [0.05, 0.1], "off_s": [0.3, 0.4, 0.5]})
        best = (None, cfg) if not grid else None
        for vals in (itertools.product(*grid.values()) if grid else []):
            trial = SegmenterConfig(**{**cfg.to_dict(), **dict(zip(grid, vals))})
            ms = [score(sts, [run_segmenter(bundle, trial, st, fps) for st in sts]) for fps, sts in data.values()]
            val = float(np.mean([objective(m) for m in ms]))
            if best is None or val > best[0]:
                best = (val, trial)
        cfg = best[1]
        # Stage 2: whole-sign commit threshold (window-tuned card threshold is too strict for full signs)
        best_t = None
        for th, mg, tta in itertools.product((0.4, 0.5, 0.6, 0.7), (0.0,), (1, 3)):
            trial = SegmenterConfig(**{**cfg.to_dict(), "commit_threshold": th, "margin": mg, "tta": tta})
            ms = [score(sts, [run_segmenter(bundle, trial, st, fps) for st in sts]) for fps, sts in data.values()]
            val = float(np.mean([objective(m) for m in ms]))
            if best_t is None or val > best_t[0]:
                best_t = (val, trial)
        cfg = best_t[1]
        print("decision chosen:", {"threshold": cfg.commit_threshold, "margin": cfg.margin, "tta": cfg.tta})
        cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        print("chosen on this split:", {k: getattr(cfg, k) for k in grid}, f"-> {cfg_path.name}")

    report = {"model": bundle.name, "segmenter": cfg.to_dict(), "results": {}}
    for src, (fps, sts) in data.items():
        for name, run in (("baseline", lambda st, fps=fps: run_baseline(bundle, st, fps)),
                          ("segment", lambda st, fps=fps: run_segmenter(bundle, cfg, st, fps))):
            if name == "baseline" and args.no_baseline:
                continue
            REASONS.clear()
            m = score(sts, [run(st) for st in sts])
            if name == "segment":
                total = max(sum(REASONS.values()), 1)
                m["end_reasons"] = {k: round(v / total, 3) for k, v in sorted(REASONS.items())}
            report["results"][f"{src}/{name}"] = m
            d = f"{m['median_delay_s']:.2f}s" if m["median_delay_s"] is not None else "-"
            print(f"{src:22s} {name:8s} signs {m['signs']:4d} | correct {m['correct']:6.1%} wrong {m['wrong']:5.1%} "
                  f"missed {m['missed']:5.1%} | extra/sign {m['extra_per_sign']:.3f} idle false/min {m['idle_false_per_min']:.2f} | delay {d}"
                  + (f" | ends {m['end_reasons']}" if "end_reasons" in m else ""))
    if args.out:
        Path(project_path(args.out)).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
