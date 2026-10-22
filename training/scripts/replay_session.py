"""Replay a saved live session (MUDRA_SESSION_DIR) through the live recognizer, offline.

The API saves each /stream session as landmarks only when MUDRA_SESSION_DIR is set. Replaying
shows exactly what the segmenter and model did with real camera input, and lets settings or a
new model be tried on it before they are deployed:

    python training/scripts/replay_session.py experiments/sessions/session_20261012-122400.npz
    python training/scripts/replay_session.py experiments/sessions/*.npz --set '{"still_s": 0.8}'
    python training/scripts/replay_session.py SESSION --model experiments/exports/<model>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mudra_ml.config import project_path
from mudra_ml.preprocessing.sequence import load_sequence
from mudra_ml.streaming import Frame, ModelBundle, SegmenterConfig, StreamingRecognizer


def replay(bundle: ModelBundle, cfg: SegmenterConfig, path: Path, quiet: bool = False) -> Counter:
    seq = load_sequence(path)
    w, h = int(seq.meta.get("width", 640)), int(seq.meta.get("height", 480))
    rec = StreamingRecognizer(bundle, SegmenterConfig(**{**cfg.to_dict(), "provisional_s": 1e9}), w, h)
    counts: Counter = Counter()
    t = seq.timestamps_ms / 1000
    print(f"{path.name}: {seq.num_frames} frames, {t[-1]:.1f} s, {(seq.num_frames - 1) / max(t[-1], 1e-6):.1f} fps,"
          f" hands {seq.hand_present.any(axis=1).mean():.0%}, client {seq.meta.get('client', {})}")
    for i in range(seq.num_frames):
        ev = rec.push(Frame(float(t[i]), seq.hands[i], seq.hand_present[i], seq.pose[i], bool(seq.pose_present[i])))
        if not ev or ev["event"] not in ("sign_ended", "too_short"):
            continue
        st = ev.get("stats", {})
        counts[ev["event"] if ev["event"] == "too_short" else ev["result"]["status"]] += 1
        counts[f"end:{st.get('reason')}"] += 1
        if quiet:
            continue
        if ev["event"] == "too_short":
            print(f"  {t[i]:7.2f}s too_short {st.get('duration_s')}s")
            continue
        r = ev["result"]
        top = ", ".join(f"{c['sign']} {c['confidence']:.2f}" for c in r["top_k"][:3])
        print(f"  {ev['t0']:7.2f}-{ev['t1']:6.2f}s {st.get('reason'):10s} {r['status']:9s} {r.get('sign') or '-':16s}"
              f" bg {r.get('background', 0):.2f} | {top}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--model", default="app/backend/models/isl_mudra_combined_v15b_bilstm")
    ap.add_argument("--config", default="training/configs/streaming.json")
    ap.add_argument("--set", default=None, help="JSON settings on top of --config")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    args = ap.parse_args()
    bundle = ModelBundle(project_path(args.model))
    cfg = SegmenterConfig.load(project_path(args.config))
    if args.set:
        cfg = SegmenterConfig(**{**cfg.to_dict(), **json.loads(args.set)})
    total: Counter = Counter()
    for s in args.sessions:
        total += replay(bundle, cfg, Path(s), args.quiet)
    print("total:", dict(sorted(total.items())))


if __name__ == "__main__":
    main()
