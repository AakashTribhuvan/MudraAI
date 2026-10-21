"""Score the teammate's packaged models (MitIndia 8105c5e) against our word model on OUR held-out test clips.

Models: common_word_video (17-word BiLSTM), isl500_video (14-word SVM), signline_intro_video (26-word SVM),
static_top10 (10-word per-frame SVM). For each, the test clips of words it shares with our vocabulary are
classified by it and by our word model (at the live commit threshold); on every other test clip any answer
it gives is a false alarm. Results go to <model dir>/evaluation_on_our_test.json (shown by GET /models) and
experiments/reports/teammate_models_on_our_test.json.

    python training/scripts/compare_teammate_models.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from mudra_ml.config import project_path
from mudra_ml.preprocessing.sequence import load_sequence
from mudra_ml.streaming import Frame, ModelBundle, SegmenterConfig

sys.path.insert(0, str(project_path("app/backend")))
from mudra_api.isl500_video import ISL500WordModel, _palm_directions, temporal_features
from mudra_api.static_top10 import StaticTop10Model, StaticTop10StreamingRecognizer

MODELS = project_path("app/backend/models")
# their glosses -> ours (assumed equivalent, not ISL-reviewed)
TO_OURS = {"THANKYOU": "THANK_YOU", "THANK": "THANK_YOU", "I_ME_MINE_MY": "I"}


def ours(label: str) -> str:
    return TO_OURS.get(label, label)


def common_word(path: Path):
    bundle = ModelBundle(path)

    def predict(seq, w, h):
        d = bundle.decide(bundle.probs([seq], w, h)[0])
        return d["sign"] if d["status"] == "ok" else None
    return [c for c in bundle.classes if not c.startswith("_")], predict


def video_svm(path: Path):
    model = ISL500WordModel(path)

    def predict(seq, w, h):
        try:
            vec, _ = temporal_features(seq, w, h, model.descriptor)
        except ValueError:
            return None
        r = model.predict(vec)
        return r["sign"] if r["status"] == "ok" else None
    return model.classes, predict


def static_top10(path: Path):
    model = StaticTop10Model(path)

    def predict(seq, w, h):  # the live rule: a label held for 0.45 s with 3 votes; first answer counts
        rec = StaticTop10StreamingRecognizer(model, w, h)
        for i in range(seq.num_frames):
            f = Frame(t=float(seq.timestamps_ms[i]) / 1000, hands=seq.hands[i], present=seq.hand_present[i],
                      pose=seq.pose[i], pose_present=bool(seq.pose_present[i]))
            out = rec.push(f, _palm_directions(f.hands, f.present))
            if out:
                return out["result"]["sign"]
        return None
    return model.classes, predict


LOADERS = {
    "common_word_video": ("common_word_video_baseline", common_word),
    "isl500_video": ("isl500_video_pilot/model.joblib", video_svm),
    "signline_intro_video": ("signline_intro_video/model.joblib", video_svm),
    "static_top10": ("static_top10_v2/model.joblib", static_top10),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="data/metadata/splits/combined_words_v4_app/test.csv")
    ap.add_argument("--word-model", default="app/backend/models/isl_mudra_combined_v13b_bilstm")
    args = ap.parse_args()

    bundle = ModelBundle(project_path(args.word_model))
    seg = SegmenterConfig.load(project_path("training/configs/streaming.json"))
    threshold = seg.commit_threshold if seg.commit_threshold is not None else bundle.threshold
    with open(project_path(args.split), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clips = []
    for r in rows:
        seq = load_sequence(project_path(r["landmarks_path"]))
        clips.append((r["label"], seq, int(seq.meta.get("width", 0)), int(seq.meta.get("height", 0))))
    v9_cache: dict[int, str | None] = {}

    def v9(i):
        if i not in v9_cache:
            label, seq, w, h = clips[i]
            d = bundle.decide(bundle.probs([seq], w, h)[0], threshold=threshold)
            v9_cache[i] = d["sign"] if d["status"] == "ok" else None
        return v9_cache[i]

    report = {}
    for model_id, (rel, loader) in LOADERS.items():
        classes, predict = loader(MODELS / rel)
        shared = sorted({ours(c) for c in classes} & set(bundle.classes))
        per_word: dict[str, Counter] = defaultdict(Counter)
        false_alarms: Counter = Counter()
        others = 0
        for i, (label, seq, w, h) in enumerate(clips):
            got = predict(seq, w, h)
            got = ours(got) if got else None
            if label in shared:
                c = per_word[label]
                c["clips"] += 1
                c["model_correct"] += got == label
                c["model_wrong"] += got is not None and got != label
                c["word_model_correct"] += v9(i) == label
            else:
                others += 1
                false_alarms[got] += got is not None
        total = sum(per_word.values(), Counter())
        fa = sum(false_alarms.values())
        result = {
            "date": datetime.now().astimezone().date().isoformat(), "split": args.split, "word_model": bundle.name,
            "model": model_id, "vocabulary": len(classes), "shared_words": shared, "label_mapping": TO_OURS,
            "shared_clips": total["clips"], "model_correct": total["model_correct"],
            "model_wrong": total["model_wrong"], "word_model_correct": total["word_model_correct"],
            "false_alarms": fa, "other_clips": others, "false_alarm_rate": round(fa / max(others, 1), 4),
            "per_word": {k: dict(v) for k, v in sorted(per_word.items())},
            "not_in_our_vocabulary": sorted(c for c in classes if ours(c) not in bundle.classes),
            "note": "Our test clips come from the same signers as our word model's training data; the teammate's "
                    "models were trained on other people's videos, so this comparison favours our word model.",
        }
        report[model_id] = result
        out = MODELS / rel
        out = (out if out.is_dir() else out.parent) / "evaluation_on_our_test.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{model_id}: shared {len(shared)} words / {total['clips']} clips -> model {total['model_correct']} "
              f"(wrong {total['model_wrong']}), {bundle.name} {total['word_model_correct']}; false alarms "
              f"{fa}/{others} ({fa / max(others, 1):.1%})", flush=True)
    path = project_path("experiments/reports/teammate_models_on_our_test.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
