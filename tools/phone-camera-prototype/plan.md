# MUDRA ISL Recognition: Data-to-Model Plan

**Status:** an exploratory still-image pose baseline has been run; verified labels, source groups,
motion sequences, and a live sign recognizer are not yet established.
**Scope:** decide whether a model improves sign recognition and live use before investing in
large-scale training. This exploratory work is kept under
`Mudra/tools/phone-camera-prototype`; the production app and validated model live at the workspace
root and remain the only default recognition path.

## Current UI and model separation (2026-10-14)

The dashboard top-level navigation now separates **Static Signs** from **Dynamic Signs**. Studio
and still-image Learning live under Static Signs. Its model is
`training_data/models/static_pose_medoids.npz`. Dynamic Signs shows live tracker diagnostics and
explicitly reports **NOT TRAINED**; the separate dynamic checkpoint/data pipeline does not exist.
The link to `video_review.py` still opens the standalone reviewer on port 8766, not an integrated
video player. Prerecorded testing and a reviewed-example learning library remain future work.

The static prototype's selected vocabulary has been expanded to 20 folders by adding `WATER`,
`NOT`, `HURT`, `CRYING`, `THAT`, `REALLY`, `OLD_AGE`, `ANGRY`, `SOME ONE`, and `TELL`. Their
observed folder sizes range from 9 to 21 still images. These are provisional folder labels selected
by image count, not verified ISL glosses or proof that the signs are static. Review each candidate
with an ISL-fluent signer; exclude or move motion-dependent signs to the dynamic workflow. Re-run
`python static_sign_prototype.py prepare` and `python static_sign_prototype.py train` to regenerate
static artifacts after selection changes. The dynamic pipeline must never load or overwrite this
static model.

## Product direction: curated sign catalog first

Treat a reviewed sign catalog as the source of truth, and treat the vision model as a secondary
retrieval aid. Each catalog entry should have a stable ID, reviewed ISL gloss/meaning, variants,
reference media, motion type, signer/source/licensing provenance, and review status. The live
system should compare a tracked action with the approved reference examples, rank candidates, and
return "unknown" when no candidate is sufficiently similar. Do not let an unreviewed folder name
become a dictionary definition or make a model's guess authoritative.

For the current small pilot, keep labeled landmark examples and metadata in versioned NumPy files
plus the CSV provenance/review manifest; a vector database is unnecessary. A nearest-exemplar or
small exemplar-bank matcher is a better first retrieval baseline than collapsing every sign to one
average pose. Static poses can be compared as poses; movement-dependent signs need ordered landmark
sequences and a temporal similarity method (for example, a carefully evaluated DTW baseline).
Sentence translation is a separate future task: knowing isolated word signs does not by itself
recover sign order, grammar, non-manual markers, or sentence meaning.

After dynamic-sign sequences and reviewed sentence glosses are available, evaluate a small language
model or constrained next-gloss predictor as an optional contextual reranker. Keep visual evidence
and unknown/rejection decisions authoritative, report the raw visual candidate separately, and test
for sentence-level gains without allowing context to conceal a weak or incorrect sign match.

ISLRTC's reviewed dictionary videos are a relevant reference for meanings and variants, subject to
the source's reuse terms. The Kaggle dataset named in the project brief must be audited for language,
label meaning, signer/source metadata, and license before inclusion; a static alphabet dataset is
not a substitute for verified ISL word or sentence examples. Do not add either source to training
until those conditions are recorded.

## The goal and the three different problems

The immediate goal is not sentence translation. It is to determine when signing begins and ends,
then recognize the sign or signs inside those boundaries. These tasks need different labels and
must be evaluated separately:

1. **Isolated sign recognition:** a clip contains one sign instance and one verified sign label.
   The output is one class for the clip. This is the smallest useful first classification task.
2. **Continuous sign recognition:** a longer clip contains a sequence of signs. It needs time
   boundaries and usually an ordered gloss sequence (or another explicit sign-level annotation);
   a clip-level sentence label alone does not identify where each sign occurs.
3. **Sign-language translation:** a sequence of signed linguistic content is mapped to a natural
   language sentence. This is not word-for-word sign classification: ISL grammar and natural
   language word order differ. It needs sentence-level parallel data and its own evaluation.

Static and dynamic describe the movement pattern, not separate recognition targets. A sign may have
a relatively stable handshape, a required movement, or a meaningful combination of both. These
properties must be checked from the actual ISL gloss and a signer's full example, not inferred from
its English folder name or one still frame. ISLRTC describes sign languages as using hand movement,
facial expression, and head/body position; its official dictionary is a useful place to inspect
video examples ([official ISLRTC dictionary announcement](https://islrtc.nic.in/video/indian-sign-language-dictionary-launched-on-website/)),
but the English word alone may have multiple meanings and corresponding signs.
For each pilot label, record the intended meaning/gloss and have an ISL-fluent reviewer inspect the
dictionary/example video for required movement and variants. A still pose can be a useful
recognition target only when the chosen sign is actually distinguishable from a pose. A still hand
between signs is not enough evidence to create another instance of the same sign. Boundary labels
must represent the intended sign instance, not just motion/no-motion.

If movement is required to convey the gloss, then the sign is dynamic for recognition and its
ordered movement belongs in the sample. A frame captured during that sign is only one pose within
the movement, not a complete static example. Small incidental movement or camera/landmark jitter
does not by itself make a lexical sign dynamic; the distinction is whether the motion changes or
completes the sign meaning.

## What exists already and what that means

The live phone-webcam prototype currently has a heuristic start/rest boundary detector and a browser
preview, but it does not save camera video/landmarks. An optional static-pose medoid experiment can
compare still-like webcam poses during stable holds; it is not a validated recognizer. The boundary
heuristic still needs evaluation with a signer before it should be treated as a source of training
labels. The offline `static_sign_prototype.py` extracts features from stills; neither path preserves
temporal order or recognizes movement-dependent signs.

### Current `training_data` snapshot

At the time this plan was updated, `training_data/Frames_Word_Level` contained 131
word-named folders and 1,036 `.jpg` files. Seventeen folders were empty. The image counts vary
substantially by folder, and example filenames are numbered images such as `ABUSE (1).jpg`.
No video files were present under that directory. This is a filesystem inventory only: it does not
establish that the images are unique signs, correctly labeled, independent examples, or consecutive
frames from known source videos.

Treat this collection as extracted still frames until its source and extraction method are
confirmed. Several frames from one video are not several independent training examples. A folder
name is a proposed label, not verified ISL ground truth. Do not train a motion classifier from
unordered still images or split near-identical frames across train and test. First determine whether
the original videos, source URLs/IDs, signer/session details, timestamps, and use permissions exist.
Keep this inventory snapshot separate from future verified counts.

The first selected pilot processed `YOU` (110 JPGs), `I_ME_MINE_MY` (97 JPGs), and `WHAT` (31 JPGs).
The automatic Pose/Hand tracking gate records pose visibility and hand size, and
retains rejected samples with reasons. A deterministic split keeps each inferred filename family
together, but family IDs are not confirmed source videos or signers. The exploratory nearest-medoid
test score is not a valid accuracy claim: labels remain unverified; a signer/source may appear across
different filename families; and some validation splits are too small to calibrate unknown rejection.
The manifest, feature archive, review contact sheets, model, confusion matrix, and evaluation report
are under `training_data/`.

The still-image feature schema uses torso-centered/body-scaled shoulder and hip anchors, unit
directions for upper-arm and forearm bones, body-relative hand-wrist locations, and palm-width-scaled
finger shape. Unit directions prevent different measured arm-segment lengths from directly changing
the arm-pose feature; body-relative wrist location remains because signing space is meaningful.
Automatically accepted samples still require human review: detector confidence and hand size do not
prove every finger landmark is correct.

The current schema-v10 pilot uses `I_ME_MINE_MY`, `YOU`, `FOOD`, `HELP`, `WHAT`, `HOW`,
`COLLEGE_SCHOOL`, `NICE`, `THANK`, and `DO`. The five additions have multiple inferred filename
groups as well as image counts, though their labels remain unverified. Each source image has
original and horizontal-mirror extractions grouped together in the
split; only original images are used for validation/test scoring. When hips are not reliably
visible, features estimate torso center using shoulders and a shoulder-width vertical offset.
Pose nose/ears (or eyes when ears are unavailable) provide a face anchor; wrist position is encoded
relative to the nose and face width, with an index-finger direction dot product toward the nose.
This uses the existing Pose Landmarker model, not a dense face mesh.

The previous schema-v9 run accepted 506 augmented rows and scored 51.3% (20/39); it was replaced
when the pilot expanded to ten labels. The schema-v10 run accepted 656 augmented rows (351
originals and 305 mirrored views); 55 mirrored views were rejected because their detected hand
count differed from their original. Its 55-image original test split scored 36.4% (20/55). This is
exploratory evidence only: source groups and labels remain unverified, and the score is not a
new-signer estimate. Several new count-specific profiles have no rejection threshold and are
withheld from Learning.
Schema v10 creates separate medoids and rejection thresholds for each exact detected-hand count.
Live and still-image matching only compares one-hand input to one-hand profiles and two-hand input
to two-hand profiles; zero is no longer a wildcard. `HELP` is trained only from two-hand examples.
Current trained profiles include one- and two-hand variants for `FOOD`, `I_ME_MINE_MY`, `WHAT`, and
`YOU`, and two hands for `HELP`. The two-hand `FOOD` and `WHAT` profiles have no reliable rejection
threshold yet and are therefore withheld from Learning.
Detector handedness is absent from feature extraction, prediction inputs, and the quality gate.
Hands are ordered spatially relative to the torso; body and hand geometry still preserve meaningful
bilateral placement. The Learning tab's gold wrist-position markers are reconstructed from the
selected label/count prototype's torso-relative wrist coordinates and the live shoulder/torso
anchor, not from a neural-network attention map or verified dictionary pose. They are approximate
and require visible shoulders.

The matching path counts raw detections before body-relative feature-slot assignment. The current
manifest has 11 accepted one-hand and 13 two-hand `HELP` originals; only the 13 two-hand examples
train the `HELP` profile. Their folder labels and image quality remain unverified; the data does not
establish that class reliably.
A second hand missed due to occlusion cannot be recovered by this check. Mirroring remains an
unverified augmentation assumption for asymmetric signs.

An optional `test.py --static-demo` path now compares live landmarks against this medoid model while
the activity tracker reports a static hold. It stabilizes a match before displaying it in the web
dashboard and uses p90 distance estimates from inferred filename-group holdouts as explicitly
experimental demo thresholds. This is for end-to-end smoke testing only; it is not live validation
of these labels or recognition quality.

The sibling [MUDRA handover](../Mudra/HANDOVER.md) documents a much more developed reference
pipeline: normalized landmark sequences, a BiLSTM, a background class, grouped data splits, and
synthetic continuous-stream evaluation. It reports strong clip-level results on its existing
test data, but much weaker results on unseen recording sessions and a live-webcam trial in which
none of 11 attempted signs were recognized. These are results reported by that project, not
independently reproduced here. The key lesson is to treat the existing model as a baseline to
reproduce where labels overlap, and to prioritize signer/session-held-out and real-stream behavior
over random-split clip accuracy. Do not assume more training or a new architecture will solve the
live gap.

## Research and reusable references

These are references for evaluation design and candidate approaches, not evidence that their
reported scores will transfer to this camera, signer population, or ISL data.

| Resource | What it is useful for | Limit |
|---|---|---|
| [iSign: A Benchmark for Indian Sign Language Processing (ACL Findings 2024)](https://arxiv.org/abs/2407.05404) and its [project and dataset site](https://exploration-lab.github.io/iSign/) | ISL-specific benchmark with video/pose-to-text and other tasks; useful for understanding continuous ISL data, task definitions, and the difference between phrase translation and isolated word classification. | Its task/data format may not match the supplied videos. The project reports that certified-signer validation covered a sampled subset, so verify labels and applicable terms rather than assuming every pair is ground truth. |
| [Hierarchical Windowed Graph Attention Network and a Large Scale Dataset for Isolated Indian Sign Language Recognition](https://arxiv.org/abs/2407.14224) | A recent ISL-specific isolated recognition reference and candidate comparison point for landmark-based methods. | A published benchmark result is not a result on our held-out signers or phone-camera stream. Dataset and code terms must be checked independently. |
| [SPOTER: Sign Pose-Based Transformer for Word-Level Sign Language Recognition (WACV Workshops 2022)](https://openaccess.thecvf.com/content/WACV2022W/HADCV/html/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.html) and [code](https://github.com/maty-bohacek/spoter) | A pose/landmark sequence approach designed for relatively low compute; useful as a candidate baseline or implementation reference if our data is small. | Its experiments use WLASL and LSA64, not ISL. Check code and dataset licenses separately; do not assume the code license grants rights to its data. |
| [WLASL paper](https://arxiv.org/abs/1910.11006) and [dataset/code repository](https://github.com/dxli94/WLASL) | Word-level video recognition, split metadata, and dataset-loader examples. The repository documents fields such as signer, source, gloss, and clip frame range, which are useful ideas for our own manifest. | WLASL is American Sign Language, not ISL. Its repository requires agreement to its Computational Use of Data Agreement; do not download or use its media without reviewing the terms. Use it as a methodology reference, not ISL training data. |
| [Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation (CVPR 2020)](https://arxiv.org/abs/2003.13830) and [code](https://github.com/neccam/slt) | A reference for continuous sign recognition followed by translation, including the distinction between gloss recognition and text translation. | It targets a larger continuous-data problem. It is not the first model to train for one labeled clip per sign, and its benchmark language/domain is not ISL. |
| [AUTSL](https://arxiv.org/abs/2008.00932) | A large-scale isolated/multimodal benchmark useful for understanding signer-independent evaluation and appearance variation. | It is Turkish Sign Language; results and vocabulary do not transfer to ISL. |

Before using any external data, code, pretrained weights, or derived features, record the applicable
license/terms, attribution requirements, allowed research/commercial use, and redistribution
restrictions. Obtain informed consent and permission for the supplied videos and any new recordings.
Do not upload private recordings to public services or share them outside the approved project.

## Staged plan and decision gates

### Stage 0 — Receive and inventory the video collection

Before writing a trainer, identify:

- Whether each file is an **isolated sign clip**, a continuous signing stream, or a mixture.
- How labels are provided: filename/folder convention, annotation table, per-clip gloss, or
  start/end timestamps and ordered glosses. Verify whether a label is an ISL gloss, an English
  translation, a sentence, or only a topic.
- Signer ID, session/recording-day ID, camera/device, framing, frame rate, resolution, and source
  for each recording, if available.
- Whether the same source video appears in multiple crops, encodings, resolutions, or folders.
- Data origin, consent, license/terms, allowed use, and whether any faces/other people require
  additional privacy handling.
- Readability and completeness: file count, duration, dimensions, frame rate, decode failures,
  label counts, missing annotations, and class imbalance.

Create a read-only inventory/manifest with a stable clip ID, relative path, label(s), signer and
session IDs where known, source-video ID, time range, technical metadata, and a content hash. Keep
the original files unchanged. Sample-check labels with a qualified ISL signer or knowledgeable
annotator; record corrections and uncertainty instead of silently relabeling.

For the existing JPG collection, inventory each image and its parent label folder, calculate a
content hash, identify exact/near duplicates, and record its source clip/frame index if recoverable.
Do not treat a sequence of JPGs as a video unless ordering and common source are verified. If the
original clip cannot be recovered, mark source/timing as unknown and use those stills only for an
explicitly separate static-image experiment, not for evaluating dynamic signing.

**Gate 0:** if videos have no trustworthy sign labels or no usable way to associate labels with
clips, stop before supervised training. First agree on an annotation format and label a small,
representative pilot. If continuous videos lack sign boundaries, annotate start/end times and an
ordered gloss sequence before attempting sign-by-sign recognition. Sentence translations alone
are not substitutes for those boundaries.

### Stage 0A — Separate word and sentence data; choose a tiny vocabulary pilot

An English word in a sentence is not guaranteed to map to one identical hand movement in every
context. ISL has its own grammar and lexical choices; English words and ISL signs are not
one-to-one tokens. A sign may vary by signer, region, inflection, or signing style, and neighboring
signs can change its timing or transitional movement. Some sentence meanings also use a phrase,
compound, or construction rather than a direct sign-for-each-English-word sequence. Have an ISL
fluent annotator confirm the intended gloss and any variants. Do not automatically pool two
different forms just because their English translations match.

Keep the two data types in separate manifests/tasks:

- **Word/isolated examples:** one verified sign instance per clip, with start/end trimmed or
  annotated consistently. Use these first for a small-vocabulary classifier.
- **Sentence/continuous examples:** retain the original sentence clip and its sentence translation;
  add an ordered gloss sequence and per-sign timestamps if the aim is to locate/recognize signs
  inside it. Sentence-level English text alone cannot supervise where each word-level sign occurs.
  Initially use these for visualization/annotation and later continuous-stream evaluation, not as
  extra isolated examples merely by saving frames.

Choose only 2–3 well-represented, visually distinguishable, annotation-verified glosses for the
first experiment. Do not select classes solely because their folders have the most JPGs: first
confirm independent source clips and the same label meaning. Add an **unknown/background** test
set containing rest, transitions, and signs outside the chosen vocabulary. A closed-set model that
must always choose one of two or three labels will otherwise mislabel every unfamiliar sign.

The initial candidate bundle selected for review is `YOU`, `I_ME_MINE_MY`, and `WHAT`, because
these folders currently have 110, 97, and 31 JPGs respectively. These counts are only extracted
image counts, not independent clip counts. The labels are provisional English descriptions; verify
the intended gloss/sense with a fluent reviewer and inspect complete ISL examples before deciding
whether each has a stable static pose suitable for a pose-only baseline. No authoritative
static/dynamic classification has yet been verified for these exact labels. The current static
prototype tests geometric similarity of one pose; it cannot recognize meaning carried by ordered
movement.

Four independent clips per gloss are enough only for a pipeline demonstration, not a reliable
accuracy claim. If those four are all that exists, use a tiny grouped train/validation/test exercise
only if source/session diversity makes the split meaningful; otherwise use them to test extraction,
visualization, and checkpoint plumbing, and gather more recordings before claiming recognition
accuracy. Never split frames from one source clip across those sets. Prefer several recording
sessions and signers; if only one signer is available, report that limitation and evaluate
session-held-out where possible.

**Pilot pass condition:** document the examples and their provenance, show that the model can
memorize a tiny training subset as a software sanity check, and then evaluate on truly independent
clips plus unknown/background examples. A high score from four highly related clips is not proof
that the vocabulary is learned robustly.

### Agreed video pilot — isolated sign segments and timing (2026-10-14)

The user-approved first task is to extract individual signs from sentence-level source clips, not
to assign the whole sentence folder label to every sign or train one sentence class. The newly
added `training_data/Videos_Sentence_Level/` collection currently contains 101 phrase folders,
687 videos, and approximately 7.1 GiB. In the inspected sample folders, videos decode at 25 fps
but have different resolutions (240x320 and 1920x1080) and durations (1.0–6.24 seconds). This is a
preliminary sample, not a replacement for a complete inventory.

Only after review, shortlist candidate glosses from `hi how are you`, `thank you so much`, and
`help me` for HELLO, THANK-YOU, and HELP. These folder/gloss associations are hypotheses: a fluent
reviewer must verify each sign's actual form and meaning in each clip, and mark absent, ambiguous,
or poor-tracking examples as rejected/uncertain. Never use a video filename such as `free` to
override its parent folder or to assert that it contains an isolated FREE sign.

For every accepted sign instance, preserve the unchanged source video and record its stable clip
ID, provisional phrase folder, reviewed gloss/variant, exact onset and offset timestamps, optional
meaningful motion-phase timestamps, signer/session IDs (pseudonymous where possible), source and
consent/license review status, and technical metadata. Do not invent unknown identities or
provenance. Keep the original sentence recording linked to each derived segment. Phase markers
(such as preparation, stroke, contact/hold, and return) are gloss-specific and optional where a
reviewer cannot identify them consistently; do not impose one phase template on every sign.

Use consistent timestamp-based frame sampling and extract ordered MediaPipe pose/hand landmarks,
per-landmark visibility, missing/tracking-quality masks, and frame timestamps into versioned
numeric sequence files. Retain the source video and render selected frames/skeletons with segment
and phase markers for review; individual frames are not independent training examples. Keep each
segment's natural duration and the true time between its frames. Do not time-stretch clips or force
different sign instances to the same wall-clock duration. Preserve variable sequence lengths and
provide total duration and reviewed phase durations separately so a later model can test whether
timing adds value to the motion path.

Before scaling beyond these candidate classes, compare the same sequence baseline with and without
duration/phase-timing features. Use actual repetitions performed at varied natural speeds, not
synthetically sped-up copies, to assess whether timing helps distinguish signs without making the
system brittle to a signer's pace. Keep the sentence-level originals for later continuous-stream
tests; do not treat phrase-level English labels as per-sign supervision.

**Video pilot verification gates:**

1. Inventory all source files (decode status, duration, frame rate, dimensions, hash, and folder);
   verify provenance/permissions and reviewer labels before training.
2. Validate annotation bounds and phase order against video playback, and replay extracted
   landmark overlays so tracking failures are visible and have explicit reject/uncertain reasons.
   Confirm sequence timestamps are ordered, segment duration is preserved, and saved features
   reload deterministically.
3. Freeze source/signer/session-grouped train/validation/test splits. Keep every segment and
   derivative from one source recording in a single split. If there are too few known independent
   groups, call the result a pipeline demonstration and do not claim signer-independent accuracy.
4. Check tiny-subset overfit and checkpoint reload, then report per-class precision/recall and
   confusion, unknown/rest/transition false accepts, boundary timing error, landmark coverage,
   inference time, and the path-only versus path-plus-timing comparison.
5. Only after offline gates pass, test a real-time webcam session with new repetitions at varied
   natural speeds and between-sign rests. Report live results separately; passing offline replay
   does not establish live-camera performance.

The generated CSV is the review worksheet. Its `label`, `verified_gloss`, `sign_motion`,
`source_clip_id`, `frame_index_or_timestamp`, `signer_id`, `session_id`, and `source_group` fields
can be completed/corrected after inspection; set `annotation_status`, `review_status`,
`group_review_status`, reviewer, and notes. `source_group` defaults to an unverified filename
family so nearby numbered files stay together; replace it with the actual shared source-video/
performance ID. Re-running `prepare` preserves those human-edited fields and recomputes features,
tracking metrics, and grouped splits.

### Approved product integration — Video Lab (2026-10-14; not implemented)

The user reported the separate video workflow was not working and approved a dedicated **Video Lab**
tab for dynamic-sign work, separate from the static-pose experiment. Integrate it into the existing
camera dashboard (normally port 8765); do not require the user to launch a second server as the
normal Video Lab experience.

The tab has three explicit areas:

- **Video tests:** select and replay source videos; once sequence extraction and a model are ready,
  run predictions on reviewed isolated segments. Show segment/gloss review status and tracking
  quality alongside results. Before a model is ready, make this playback/sequence diagnostics only
  and state **dynamic model not trained**.
- **Live test:** use the existing live camera stream and tracker. Show landmarks, activity or
  boundary state, sequence quality and model readiness. Do not display a dynamic gloss prediction
  before offline evaluation and live testing gates pass.
- **Learning:** a menu/library of reviewer-approved example videos, with confirmed gloss/variant,
  sign and optional phase boundaries, plus useful playback/scrubbing. Never use the sentence folder
  or filename as a confirmed gloss or teach unreviewed clips as ground truth.

The standalone `video_review.py` is an interim local annotation utility, not the integrated Video
Lab. It runs at port 8766 only while its separate process is open. One inspected 240x320 MP4 codec
was not natively playable in the browser; the reviewer uses OpenCV-decoded-frame playback as a
fallback. The integrated experience must provide this fallback or clearly report decode failure,
without changing/transcoding the supplied source.

**Integration gate:** the static-pose dashboard remains unchanged; prerecorded clips can be selected
and sought; live mode uses the existing tracker; Learning lists only reviewed examples; and every
view reflects its actual readiness without mock predictions. Then continue through the video-pilot
verification gates above. No dynamic prediction is a product feature until reviewed labeled
sequences pass grouped offline evaluation and live tests on new natural-speed repetitions.

### Stage 1 — Establish the task and leakage-safe split

Choose one initial task based on the inventory. Prefer isolated sign clips if the supplied material
contains them; this makes a small, interpretable first experiment. Keep boundary detection as a
separate evaluation. Do not mix continuous translation goals into an isolated-classifier score.

Freeze a split manifest **before model tuning**:

- Put every clip from a signer in only one split when signer IDs exist; also group sessions or
  recording days so near-identical conditions do not leak across splits.
- Keep source videos and every crop/transcode derived from them in the same split. Deduplicate
  using hashes and review likely near-duplicates before assigning groups.
- If there is only one signer, use session/day-held-out evaluation if possible and state that
  generalization to new signers remains unknown. If there is only one session, a reliable
  signer/session-independent claim is not possible; collect more data before making that claim.
- Reserve a locked test set. Tune preprocessing, thresholds, architecture, and early stopping on
  training/validation only. Do not repeatedly inspect test results and tune against them.

**Gate 1:** do not report generalization from a random clip split if related clips, signers, or
recording sessions cross the split. If groups are unknown, record that limitation and prioritize
collecting the missing metadata.

### Stage 2 — Make a reproducible baseline before training

Evaluate the same frozen test clips with:

1. A majority-class and a simple feature baseline (where labels permit).
2. The existing MUDRA checkpoint and its exact preprocessing, **only** for labels and inputs it
   supports. Record unavailable classes rather than mapping them silently.
3. A simple current-pipeline baseline if it can produce a class prediction. The phone-webcam
   boundary heuristic by itself is not a sign-classification baseline.

Save the model/checkpoint identity, preprocessing configuration, split-manifest hash, class map,
software versions, inference device, and baseline outputs. Report per-source and per-class scores,
not only a single overall accuracy.

### Stage 3 — Prove the data and training loop work on a small pilot

Start with the 2–3 verified glosses selected in Stage 0A, using source clips rather than treating
each extracted frame as an independent labeled example. Use a balanced subset and more than one
signer or session where possible. Four clips per class can exercise the pipeline but cannot support
a dependable generalization estimate. Reuse the existing landmark extraction/normalization ideas
if they match the actual inputs; do not build a second feature pipeline before comparing its output
with the reference pipeline.

Before a large run, verify:

- All chosen samples decode and map to the intended label and split.
- The model can intentionally overfit a tiny training subset (a pipeline sanity check, **not**
  evidence of generalization).
- Training loss decreases, predictions are not constant, and a saved checkpoint reloads to the
  same predictions.
- A label-shuffle sanity experiment does not produce meaningful held-out performance above an
  appropriate chance/class-prior baseline. Investigate any surprising result for leakage or a bug.
- The exact inference preprocessing and class mapping are shared by evaluation and the webcam
  integration path.
- Each accepted and rejected sample has an auditable quality decision, its reason, and a link to
  the source file/clip; rejection is not silent.

**Gate 2:** if the tiny subset cannot be overfit, checkpoint reload differs, labels are misaligned,
or shuffled-label validation looks suspiciously strong, fix the pipeline before scaling up.

### Stage 4 — Run the smallest useful model comparison

The current still-image baseline is an offline nearest-medoid pose matcher. Compare it next with a
nearest-exemplar pose matcher for catalog retrieval; neither is a motion-sensitive classifier.
Keep multiple reviewed examples per catalog sign rather than assuming one class medoid represents
every variant.
Once isolated source clips are available and labels verified, compare a regularized
landmark-sequence model with an intentionally simple temporal baseline. The current MUDRA BiLSTM is
a practical comparison if its input/features and label set match. Keep temporal order and
variable-length handling; avoid reducing a dynamic sign to one static pose.

Do not start with a vector database. For this small supervised experiment, store each accepted
sample's ordered landmark sequence as a versioned numeric array (for example, compressed `.npz`)
and keep labels, provenance, quality metrics, and split membership in a plain-text manifest.
The sequence array must retain timestamps/frame order, coordinates, per-landmark visibility or
confidence, and missing-value masks; include a schema version, coordinate convention, extractor
name/version, and preprocessing version. Use a separate preview renderer to display the skeleton
sequence over time, with controls to pause, scrub, and compare it to the source frame/clip.

#### Translation-, scale-, and body-proportion normalization

For a first static-pose prototype, express body landmarks relative to a torso center (for example,
the midpoint between the left/right shoulder midpoint and left/right hip midpoint) so that moving
the signer around the image does not translate the feature vector. Scale global coordinates by a
robust body measure such as shoulder width or torso length, with missing/low-confidence anchors
marked as invalid rather than silently replaced. Optionally align the torso axis after checking
that this does not remove meaningful signing-space orientation. Do not pass detector handedness
labels into the recognizer; keep image-mirror conventions explicit and validate whether mirroring
preserves each reviewed gloss before treating it as an equivalent signer variant.

Do **not** independently rescale every arm/leg bone to a unit length as the only representation:
that can erase meaningful reach, hand-to-face/chest location, relative signing-space distance, and
proportion cues. A safer first feature set is globally body-centered/body-scaled joint positions
plus optional unit bone directions or joint angles. Normalize handshape separately around the wrist
or palm using palm width, while preserving finger-joint angles and the hand's body-relative
position. Keep original coordinates and confidence alongside every transformed feature so the
normalization can be inspected and revised.

With only four images per word folder, a class mean is at best a toy nearest-prototype baseline:
images may be duplicate/adjacent frames, labels may be ambiguous, and an arithmetic mean can form
an implausible pose between valid variants. First verify source and sign identity. Compare a
normalized pose to every valid reference (or a small bank of exemplars) using only visible, reliable
joints; calibrate a distance/reject threshold on independent examples and include unknown signs.
Do not claim accuracy from matching the same four images used to form the class average. Static-pose
retrieval does not recognize signs whose meaning depends on movement.

Keep immutable source media (when permission allows) separate from derived previews and features.
Do not overwrite the supplied JPGs or videos. A CSV is a good canonical, portable index; Excel can
be generated from it for review but should not become the only authoritative record. Record
content hashes and relative paths so files can be reconciled if moved. A vector database is useful
later for searching learned embeddings at larger scale, not as the primary storage format for a
small, labeled training set.

Treat avatar animation as a related but separate output. The recognizer's landmark sequence can be
visualized and may inform retargeting, but a 2D webcam pose (including model-estimated z) is not
ground-truth 3D motion capture. Do not assume it directly provides accurate rig animation or finger
depth. If avatar work is pursued, define and version a rig-neutral skeleton, joint names, coordinate
axes, units, timestamps, confidence/missing markers, and retargeting rules; preserve the original
landmarks as well. Check that hand shape and signing-space relationships survive any mapping. A
reliable depth camera or calibrated multi-view capture may be needed for faithful 3D motion.

#### Tracking-quality gate and curation

Run the chosen landmark extractor on every candidate sequence and save per-frame/per-joint
confidence or visibility, detection coverage, missing runs, and any extraction errors. Render a
contact sheet or short landmark animation so failures such as missed hands, wrong-person tracking,
cropped fingers, and implausible jumps can be reviewed quickly.

Define acceptance thresholds on a manually reviewed pilot, before looking at model test scores.
For example, require adequate hand/pose coverage across the annotated sign interval, cap long
missing-hand gaps, and reject clips where fingers are too small or occluded for the intended label.
Use the thresholds consistently and save `accepted`, `rejected`, or `needs_review` plus a reason and
reviewer/date in the manifest. Do not silently drop rejected files. Keep originals and rejection
records so a better extractor can be tried later. Short gaps may be marked or handled with an
explicit, versioned method, but do not interpolate long gaps or discard confidence information and
then treat the result as observed motion.

Quality filtering must not create a misleading easy subset: report rejection rate by class, signer,
session, and recording condition. If one gloss is rejected much more often, improve capture or
annotation coverage rather than quietly training on a biased remainder.

Use a staged ablation only when data volume supports it:

- normalized body/hand landmarks, including relative hand-to-body position;
- landmark sequences plus motion/temporal features, if they improve validation;
- image/video or a multimodal path only if landmark-only errors show a real need, such as
  appearance detail or non-manual facial cues, and the available data and compute justify it.

Track small, reproducible runs (including fixed seeds/configuration). Do not add a Transformer,
large video model, synthetic augmentation, or extra data merely because a paper used it. Each
additional complexity must beat the simpler model on validation and then pass the frozen test/live
gates.

### Stage 5 — Evaluate classification and continuous signing separately

**Isolated clips:** report top-1 accuracy, macro-F1, balanced accuracy, per-class recall, confusion
matrix, and calibration (for example, expected calibration error). Include class counts and
confidence intervals or repeated-seed variability where the test set is large enough.

**Boundaries:** on manually annotated video, measure start/stop precision and recall, boundary
timing error (with a predeclared tolerance), missed/false segments, and false starts per minute.
Compare the heuristic to the annotations before using its output to crop training examples.

**Continuous streams:** join held-out clips or, preferably, use real continuous recordings with
annotated boundaries. Measure correct, wrong, and missed signs, extra predictions per true sign,
false predictions per idle minute, and segment/boundary quality. Test camera-like conditions:
full frame, the actual phone framing, realistic pauses/rest, and the live capture frame rate. A
concatenation of isolated clips is a useful controlled test but is not proof of natural-stream
performance.

Measure end-to-end inference latency/FPS on the actual phone-camera path, including landmark
extraction, segmentation, classification, and rendering. Record dropped frames and confidence
behavior while idle. Do not claim real-time suitability based on model-only inference timing.

### Stage 6 — Decide whether to continue, improve, or stop

Before opening the locked test set, define the minimum useful improvement with the project team.
Then:

- **Continue** only if the model beats the relevant frozen baseline on signer/session-held-out
  macro-F1 (or the agreed task metric), has no unacceptable per-class regressions, and improves
  real-stream outcomes without an unacceptable increase in false signs or latency.
- Require the measured difference to be larger than run-to-run noise; use repeated seeds and/or
  paired confidence intervals when the data supports them. If the test set is too small to resolve
  the difference, gather more independent examples instead of declaring a win.
- **Do not scale up** if training only improves random-split accuracy, fails on unseen signers,
  loses to the current checkpoint/baseline, or cannot run at the target camera rate. Inspect the
  confused classes, label quality, signer/session coverage, segmentation crops, and camera-domain
  mismatch; make one targeted change and rerun the same frozen comparison.
- Record unsuccessful experiments and retain the best checkpoint by validation criteria. Do not
  overwrite a working baseline without evidence.

### Stage 7 — Integrate only a verified model

For a passing model, package the label map, preprocessing/version, checkpoint, confidence/unknown
threshold, and runtime dependencies together. Add an explicit unknown/reject outcome so unsupported
signs are not confidently mislabeled as a known class. Run saved-video regression tests and a
manual webcam trial using the same camera/backend and resolution as the intended setup. The
currently known camera path is Camo through OpenCV camera 1 + Media Foundation at 1280x720; the
negotiated live frame size, not the requested size, is authoritative.

Keep video and landmark capture opt-in, disclose where files are stored, and obtain consent. Do not
enable persistent capture as an incidental side effect of model inference.

## Recommended data layout and audit trail

Keep originals, extracted frames, landmarks, previews, manifests, and model artifacts in distinct
locations. One workable initial layout is:

```text
training_data/
  originals/                 # immutable source clips/images, only if permitted
  Frames_Word_Level/         # supplied images; leave unchanged
  derived/
    landmarks/<extractor-version>/<sample-id>.npz
    previews/<sample-id>.mp4
  manifests/
    samples.csv              # authoritative sample, provenance, quality and split index
    labels.csv               # stable label ID, verified gloss, English notes/variants
    experiments.csv          # run ID, code/config/checkpoint and evaluation summary
```

The exact format may evolve; keep schema versions and never silently repurpose a column. `samples.csv`
should include at least: stable sample ID; relative original path and SHA-256; source video/clip ID;
frame range or timestamps; original folder label; verified gloss and label status; signer/session
pseudonymous IDs; origin/license/consent reference; dimensions/frame rate when available; extractor
and preprocessing versions; tracking coverage and missing-gap summaries; review status/reason;
group/split assignment; derived-feature path; and notes. Do not place real signer names or sensitive
consent documents in a shared training manifest.

For each experiment, record a run ID, timestamp, code revision or source hash, split-manifest hash,
label-map version, random seed, configuration, checkpoint path/hash, and metrics. Retain rejected
sample records and failed runs. Back up the manifest and permitted source data; do not make a
spreadsheet the only copy of provenance.

## What to provide with the video collection

When the videos are ready, include whatever is available for each file:

1. A label/annotation table and what each label means (verified ISL gloss, English meaning,
   sentence, or other), plus any known regional/variant forms.
2. Whether files are isolated examples, sentence/continuous videos, or extracted still frames. For
   continuous files, include start/end times and ordered glosses if already annotated; for frames,
   include source video ID and frame/time index if known.
3. Stable signer and recording-session/day IDs. Pseudonymous IDs are sufficient; do not provide
   real names.
4. The source of the videos, consent/permission, license or dataset terms, and allowed use.
5. Camera/framing details and any known duplicate, crop, or re-encoded copies.
6. Any current spreadsheet/CSV describing the files and how its labels were created.

If signer IDs or labels do not exist, say so; we can first design the smallest safe annotation and
metadata pass rather than pretending the data is ready to train.
