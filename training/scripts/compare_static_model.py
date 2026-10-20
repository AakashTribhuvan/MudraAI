"""Compare the deployed word model with the teammate's static-pose model on OUR held-out test clips.

For the static model's live labels that are also in our vocabulary (I, NICE, THANK -> THANK_YOU, YOU),
both models classify the same test clips (the word model at the live commit threshold; the static model
with its own whole-clip rule). On every other test clip, any static answer is a false alarm (it matters
when several models run at once). The result is written into the static model's card as
``evaluation_on_our_test``, which GET /models shows.

    python training/scripts/compare_static_model.py --split data/metadata/splits/combined_words_v4_app/test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

from mudra_ml.config import project_path
from mudra_ml.preprocessing.sequence import load_sequence
from mudra_ml.streaming import ModelBundle, SegmenterConfig

sys.path.insert(0, str(project_path("app/backend")))
from mudra_api.static_pose import StaticPoseModel

# the static model's labels that exist in our vocabulary (gloss correspondence assumed, not ISL-reviewed)
TO_OURS = {"I_ME_MINE_MY": "I", "THANK": "THANK_YOU", "NICE": "NICE", "YOU": "YOU"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="data/metadata/splits/combined_words_v4_app/test.csv")
    ap.add_argument("--word-model", default="app/backend/models/isl_mudra_combined_v9_bilstm")
    ap.add_argument("--static-model", default="app/backend/models/mudra_intro_static_pose_medoids_v1.npz")
    args = ap.parse_args()

    bundle = ModelBundle(project_path(args.word_model))
    seg = SegmenterConfig.load(project_path("training/configs/streaming.json"))
    threshold = seg.commit_threshold if seg.commit_threshold is not None else bundle.threshold
    static = StaticPoseModel(project_path(args.static_model))
    with open(project_path(args.split), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    shared = {TO_OURS[s] for s in static.live_labels if s in TO_OURS and TO_OURS[s] in bundle.classes}

    per_word: dict[str, Counter] = defaultdict(Counter)
    false_alarms: Counter = Counter()
    others = 0
    for r in rows:
        seq = load_sequence(project_path(r["landmarks_path"]))
        w, h = int(seq.meta.get("width", 0)), int(seq.meta.get("height", 0))
        st = static.predict_sequence(seq, w, h)
        st_sign = TO_OURS.get(st["sign"], st["sign"]) if st["status"] == "ok" else None
        if r["label"] in shared:
            ours = bundle.decide(bundle.probs([seq], w, h)[0], threshold=threshold)
            c = per_word[r["label"]]
            c["clips"] += 1
            c["word_model_correct"] += ours["status"] == "ok" and ours["sign"] == r["label"]
            c["word_model_wrong"] += ours["status"] == "ok" and ours["sign"] != r["label"]
            c["static_correct"] += st_sign == r["label"]
            c["static_wrong"] += st_sign is not None and st_sign != r["label"]
        else:
            others += 1
            if st_sign is not None:
                false_alarms[st_sign] += 1

    total = sum(per_word.values(), Counter())
    result = {
        "date": datetime.now().astimezone().date().isoformat(), "split": args.split, "word_model": bundle.name,
        "shared_words": sorted(shared), "label_mapping": TO_OURS,
        "shared_clips": total["clips"],
        "word_model_correct": total["word_model_correct"], "static_correct": total["static_correct"],
        "static_false_alarms": sum(false_alarms.values()), "other_clips": others,
        "static_false_alarm_rate": round(sum(false_alarms.values()) / max(others, 1), 4),
        "false_alarms_by_label": dict(false_alarms), "per_word": {k: dict(v) for k, v in sorted(per_word.items())},
        "untestable_live_labels": sorted(s for s in static.live_labels if s not in TO_OURS),
        "note": "Our test clips come from the same signers as the word model's training data; the static model was "
                "built from other people's phone photos. The comparison therefore favours the word model.",
    }
    card_path = project_path(args.static_model).with_suffix(".json")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["evaluation_on_our_test"] = result
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print(f"shared words: word model {total['word_model_correct']}/{total['clips']}, static {total['static_correct']}/{total['clips']}; "
          f"static false alarms {result['static_false_alarms']}/{others} ({result['static_false_alarm_rate']:.1%}) -> {card_path}")


if __name__ == "__main__":
    main()
