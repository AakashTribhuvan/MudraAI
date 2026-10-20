# Teammate models (MitIndia) and how SignLine uses them

Source: AakashTribhuvan/MitIndia, branch `work/static-landmark-tracking` (the only branch with models; `main`
holds an older phone-webcam prototype). Imported 2026-10-19 from commit `8105c5e`; earlier import: the static-pose
model from `e040672`.

## What the branch contains

| Model (in `app/backend/models/`) | Type | Words | Teammate's own test | Used here as |
|---|---|---|---|---|
| `common_word_video_baseline/` | BiLSTM, our model-card format (`model.pt`, weights only) | 17 | 45/51 (88.2%), signer-held-out | `combined` fallback; `common_word_video` |
| `isl500_video_pilot/model.joblib` | calibrated SVM, 12 sampled frames | 14 | 33/41 (80.5%), ISL500 signers 13-15 | `isl500_video` |
| `signline_intro_video/model.joblib` | calibrated SVM | 26 | 48/77 (62.3%) | loaded by nothing (hidden, as in the teammate's app) |
| `static_top10_v2/model.joblib` | SVM per frame | 10 | not reported | loaded by nothing |
| `mudra_intro_static_pose_medoids_v1` | static-pose medoids (imported earlier) | 10 | photo test | `intro_static` |

**Datasets are not in the repository.** Raw videos, landmarks and splits stayed on the teammate's PC; the branch
only has download / preparation scripts (`download_signline_intro_terms.py`, `prepare_signline_intro_data.py`)
for ISL500/ISL-DATA (Hugging Face), whose card limits use to **research and academic purposes**. Nothing was
downloaded. Each model folder has a `SOURCE.md` with the source and this licence note.

## Safety

`.joblib` files are pickles (loading one can run code). They are loaded only through
`app/backend/mudra_api/safe_joblib.py`, which resolves an allow-list of numpy / scikit-learn / joblib-array
classes and refuses everything else (tested with a booby-trapped pickle). All three files contain only
`Pipeline`, `StandardScaler`, `SVC`, `CalibratedClassifierCV` and numpy arrays. `model.pt` is loaded with
`weights_only=True`. Their features use hand landmarks and 15 body pose points (nose, eyes, ears, mouth,
shoulders, elbows, wrists, hips); no face mesh, nothing that identifies a person.

## Measured on OUR test split

`training/scripts/compare_teammate_models.py` (results in each model folder's `evaluation_on_our_test.json`
and `experiments/reports/teammate_models_on_our_test.json`; split `combined_words_v4_app/test.csv`, 1,501 clips):

| Model | Shared words / clips | Teammate model right | Our v9 right | Answers on other words' clips |
|---|---|---|---|---|
| common_word_video | 7 / 41 | 23 (56%) | 38 (93%) | 67.0% |
| isl500_video | 7 / 43 | 4 | 43 | 59.9% |
| signline_intro_video | 14 / 92 | 0 | 89 | 7.0% |
| static_top10 | 4 / 22 | 0 | 22 | 36.6% |

Our test clips come from the signers our model was trained on, so the comparison favours v9; the drop of the
teammate models is the cost of a new camera / signer set (their videos are ISL500, INCLUDE, CISLR).

## Decision (by accuracy): three modes in Communication

The models cannot be fused into one network (BiLSTMs, SVMs, pose medoids), so they are merged as a router:
every word is answered by the most accurate model that knows it (`mudra_api/extra_models.py`, `MergedFallback`).

1. **SignLine words (v9)**, the default: 314 words.
2. **Merged** (`merged`): 326 words. v9 keeps all 314 (best on every shared word above). Words v9 lacks go to
   the most accurate teammate model that has them (their own tests: common words 88.2% > ISL500 80.5% >
   SignLine-26 62.3%): EAT, FOOD, GO, HELP, NO, PLEASE, WATER, WHAT, WHERE, YES from common words; MYSELF, THIS
   from ISL500; SignLine-26 adds nothing new. Models under 50% (static poses 17.8%, static top-10 unreported and
   0 right on ours) give no words: DO, FROM, HOW, NAME, NOT are left out. Another model is asked only when v9 is
   *uncertain* (never on "no sign") and accepted at >= 0.7. On our 1,501 test clips: v9 1,466 right, uncertain on
   17, the router added 0 wrong answers (`training/scripts/eval_merged_auto.py`).
3. **Auto** (`auto`, `mudra_api/auto_mode.py`): Merged words + A-Z letters from one stream. A letter is the Spell
   mode's rule (same confident letter in 3 checks 0.25 s apart, not repeated while held) and is looked for only
   while no word sign is in progress, with a hand raised in front of the body, and not within 1 s after a word; a
   word ending within 2 s of a letter needs >= 0.9 (moving between letters can look like a short word). Letters
   close together (3 s) build one spelled word in the sentence. On our word clips streamed frame by frame: the same
   words found as v9 alone (638 = 638; streaming isolated clips finds fewer than whole-clip classification for
   both), false letters on 3 of 1,501 clips (0.2%). Spelling accuracy itself is not measured (no held-letter videos).
   Phrases are not part of Auto (the phrase model needs a recorded 2-6 s clip; Phrase mode stays separate).

The teammate models alone (`intro_static`, `common_word_video`, `isl500_video`) still work through the API
(`"model"` in the stream start message) but are no longer in the dropdown. Not imported: the teammate's segmenter
re-tuning (unmeasured live) and the test-video viewers (need their videos).

## Next

1. Camera check of Merged (the 12 added words, false answers in everyday signing) and Auto (spelling).
2. To train the 10 words into our own model: record them on the Teach a sign page, or (with permission and the
   research-only licence) download the ISL500 clips with the teammate's scripts.
