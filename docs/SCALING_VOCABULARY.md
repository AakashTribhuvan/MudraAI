# Scaling the vocabulary: barriers between words (2026-10-18)

Question from the owner: recorded words work well with 6-8 words; with 400-500 words they must not mix
up, made-up answers must not appear, and accuracy must not drop. This document records what was built,
what was measured, and what was and was not switched on.

## What was built

| Part | Where | Live? |
|---|---|---|
| **Margin** between taught words (best must beat the runner-up) | `FewShotConfig.margin` (`mudra_ml/fewshot.py`) | setting kept at 0.0 (chosen on validation) |
| **Per-word boundary**: each taught word's edge from how much its own takes vary; a sign outside it is not that word | `FewShotConfig.radius_k` | **on, `radius_k` 2.0** (`training/configs/fewshot.json`) |
| **Leave out takes that do not match** the word's other takes | `FewShotConfig.outlier_z` | off (validation did not choose it) |
| **Quality report per recorded word**: how well the takes agree, takes left out, nearest other word and "too close", which known word the model reads the takes as | `CustomWordBank.report`, `RecordingStore.quality`, `GET/POST /recordings` → `quality`; shown on *Teach a sign* | **on** |
| **"✓ Added: next word" cue** on Communication (a clear pause between signs) | `CommunicatePage.jsx` | **on** |
| **Push-apart training** (additive-margin cosine head, CosFace) | `ModelConfig.head = "cosine"` (`models/sequence_model.py`) | trained and measured; not deployed (below) |
| **"I don't know this sign" boundary around known words** | `KnownWordBoundaries` (`fewshot.py`), `known_boundary_k` in `streaming.json`, `build_known_boundaries.py` | built, **off** (fails the known-word rule, below) |
| **Retraining with the app's recordings** | `build_app_split.py` → `combined_words_v4_app`; `mudra_words_v9.yaml` | **v9 deployed** |

## The large-vocabulary benchmark

`make_fewshot_split.py` holds 150 of the 314 words out of training entirely (164 known words remain).
`eval_fewshot_scale.py` then "teaches" N = 25 … 150 of them from k = 3 / 5 / 10 recorded clips and measures,
on held-out clips: **recognised**, **mixup** (taken as another taught word), **made-up** (a never-taught sign
gets an answer), **stolen** (a known word's clip taken by a taught word), **known correct**. Settings are
chosen on validation (half the held-out words taught, the rest act as never-taught signs); test is reported.
Two test models with the v8 recipe: normal head (`fewshot_scale_ce`) and push-apart head (`fewshot_scale_cos`).

**Test, normal model, per-word boundary (the live settings) vs the old settings, N = 150, k = 10:**
recognised 59.2% vs 61.9%, **mixup 5.0% vs 7.0%**, **stolen 0.73% vs 1.20%**, known correct 97.2% vs 96.8%
(`experiments/reports/fewshot_scale_ce.json`). Mix-ups grow with N: at k = 10 about 0% (25 words), 1% (50),
2.6-3.7% (100), 5-7% (150). Ten takes per word recognise ~10 points more than three.

**Made-up answers come mostly from the word model itself:** with the old settings 48-60% (normal model) and 41-50% (push-apart) of never-taught
signs got an answer, and most of those were known words (on validation only 8-14% came from taught words).

**Known-word boundary** (`experiments/reports/fewshot_scale_*_known.json`): with all barriers, made-up
answers fall to 10-19% (normal) / 7-14% (push-apart) and mix-ups at 150 words to ~2%, but known words
answered correctly fall from 97% to 79.5% (normal) / 86.3% (push-apart). A rule fixed beforehand allowed
at most a 3-point drop; on validation no strictness passed (`fewshot_tradeoff.py`: even known_k 6 costs
7 points), so it stays off. The better fix is an "other sign" class trained from out-of-vocabulary signs.

**Push-apart vs normal head:** same accuracy on the 164 known words (98.2% test clips); with barriers it
loses fewer known words and makes up fewer answers, but it does not reduce mix-ups between taught words.

## v9: the word model retrained with the app's recordings

`combined_words_v4_app` = `combined_words_v3` + 30 *Teach a sign* takes of 6 words (COMPUTER, FRIEND, HAPPY,
I, NEW, TODAY; all already among the 314; one signer, one session; per word 3 train / 1 val / 1 test).
Two candidates; the push-apart one was worse on validation streams (e.g. desk 63.9% vs 66.7%), so the
normal one was chosen. **v9 vs v8, test streams at 7 fps** (`experiments/reports/streams_test_*.json`):

| Source / framing | v8 correct | v9 correct |
|---|---|---|
| INCLUDE normal / close-up / close-up + rest / desk | 86.6 / 85.0 / 84.4 / 65.4% | 86.6 / 85.5 / 85.3 / 67.2% |
| Team words normal / close-up / close-up + rest / desk | 92.8 / 87.3 / 84.0 / 81.8% | 95.0 / 89.5 / 85.1 / 82.9% |
| INCLUDE videos normal / close-up / close-up + rest / desk | 86.5 / 83.3 / 84.3 / 73.9% | 86.3 / 83.3 / 84.3 / 74.1% |
| The owner's own takes (6 clips) normal / close-up / close-up + rest / desk | 16.7 / 16.7 / 16.7 / 0% | 83.3 / 66.7 / 66.7 / 66.7% |

Wrong answers stay within about ±1 point. Clip test 98.33% (v8 98.06% on the same test set without the 6 app
clips). The owner's takes are new takes from the same session as the training takes: a promising signal,
not a measure for new people. With v9 the owner's six words are far apart (nearest-word similarity 0.12-0.31,
none "too close"; under v8, I and TODAY were 0.83 alike).

## How to grow to 400-500 words

1. Record each word **10-15 times by at least 3 people on different days** (the *Teach a sign* page warns
   about takes that disagree and words that look alike).
2. When words have 5+ takes, run `build_app_split.py` and retrain (`mudra_words_v9.yaml` with the new split);
   deploy only if the test streams are not worse (the rule used for v9). Taught words that stay prototypes
   lose about a quarter of their signs to known words (`taken_as_known` 26-28% in the benchmark); retraining fixes that.
3. Next modelling step against made-up answers: an **"other sign" class** trained from signs outside the
   vocabulary (e.g. cuts from the sentence videos), measured with `eval_fewshot_scale.py` (made-up) and
   `eval_continuous.py` (known words).

Rollback: `MUDRA_MODEL_DIR=app/backend/models/isl_mudra_combined_v8_desk_bilstm`; old custom-word settings:
`radius_k` null in `training/configs/fewshot.json`.
