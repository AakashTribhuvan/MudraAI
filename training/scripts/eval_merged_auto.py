"""Check the merged and auto modes on OUR held-out word clips (offline; live accuracy is a separate check).

Merged: each whole clip is classified by our word model at the live commit threshold; when it is uncertain the
merged fallback may answer with a teammate word. Every test clip is a word our model knows, so any fallback
answer is wrong: this counts the errors the merge adds.
Auto: each clip is streamed frame by frame through AutoRecognizer (segmenter + merged words + letters). Counts
the words found and the letters fired during word signs (all false: the clips are words, not spelling).

    python training/scripts/eval_merged_auto.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter

from mudra_ml.config import project_path
from mudra_ml.letters import LetterModel
from mudra_ml.preprocessing.sequence import load_sequence
from mudra_ml.streaming import Frame, ModelBundle, SegmenterConfig, StreamingRecognizer

sys.path.insert(0, str(project_path("app/backend")))
from mudra_api.app import create_app  # noqa: F401  (same import path as the server)
from mudra_api.auto_mode import AutoRecognizer
from mudra_api.extra_models import ExtraModels, MergedFallback
from mudra_api.settings import Settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="data/metadata/splits/combined_words_v4_app/test.csv")
    args = ap.parse_args()
    st = Settings(device="cpu")
    bundle = ModelBundle(st.model_dir)
    cfg = SegmenterConfig.load(project_path("training/configs/streaming.json"))
    threshold = cfg.commit_threshold if cfg.commit_threshold is not None else bundle.threshold
    extra = ExtraModels(st.common_word_video_model_dir, st.isl500_video_model, st.signline_intro_video_model)
    merged = MergedFallback(extra.sources(), set(bundle.classes))
    letters = LetterModel(st.letters_model_dir)
    rows = list(csv.DictReader(open(project_path(args.split), encoding="utf-8")))

    m = Counter()
    a = Counter()
    for r in rows:
        seq = load_sequence(project_path(r["landmarks_path"]))
        w, h = int(seq.meta.get("width", 0)), int(seq.meta.get("height", 0))
        d = bundle.decide(bundle.probs([seq], w, h)[0], threshold=threshold)
        m["clips"] += 1
        m["v9_correct"] += d["status"] == "ok" and d["sign"] == r["label"]
        if d["status"] == "uncertain":
            m["v9_uncertain"] += 1
            got = merged(seq, w, h, d)
            if got:
                m["fallback_answers_all_wrong"] += 1
        # streamed like live: 0.5 s without hands before and 1.3 s after, so the segmenter sees a start and an end
        def stream(rec):
            found, n_letters = [], 0
            first, last = 0, seq.num_frames - 1
            t0 = float(seq.timestamps_ms[0]) / 1000
            blank = lambda i, t: Frame(t=t, hands=seq.hands[i] * 0, present=seq.hand_present[i] & False,
                                       pose=seq.pose[i], pose_present=bool(seq.pose_present[i]))
            frames = [blank(first, t0 - 0.5 + k / 30) for k in range(15)]
            frames += [Frame(t=float(seq.timestamps_ms[i]) / 1000, hands=seq.hands[i], present=seq.hand_present[i],
                             pose=seq.pose[i], pose_present=bool(seq.pose_present[i])) for i in range(seq.num_frames)]
            t1 = float(seq.timestamps_ms[last]) / 1000
            frames += [blank(last, t1 + k / 30) for k in range(1, 40)]
            for f in frames:
                out = rec.push(f)
                for ev in out if isinstance(out, list) else [out] if out else []:
                    if ev["event"] == "letter":
                        n_letters += 1
                    elif ev["event"] == "sign_ended" and ev["result"].get("status") == "ok":
                        found.append(ev["result"]["sign"])
            return found, n_letters

        plain, _ = stream(StreamingRecognizer(bundle, cfg, w, h))
        a["v9_streamed_found"] += r["label"] in plain
        words, n_letters = stream(AutoRecognizer(StreamingRecognizer(bundle, cfg, w, h, None, fallback=merged), letters, w, h))
        a["clips"] += 1
        a["word_found"] += r["label"] in words
        a["other_words"] += sum(x != r["label"] for x in words)
        a["false_letters"] += n_letters
        a["clips_with_false_letter"] += n_letters > 0
    report = {"split": args.split, "merged": dict(m), "auto": dict(a), "merged_words": merged.routing,
              "excluded_words": merged.excluded}
    out = project_path("experiments/reports/merged_auto_on_our_test.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in ("merged", "auto")}, indent=1))
    print("merged words by model:", {mid: sorted(w for w, x in merged.routing.items() if x == mid) for mid in set(merged.routing.values())})
    print("excluded (model below the accuracy bar):", merged.excluded)


if __name__ == "__main__":
    main()
