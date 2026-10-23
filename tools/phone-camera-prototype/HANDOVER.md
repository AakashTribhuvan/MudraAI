# MUDRA Phone Webcam and Finger Tracking — Handover

Last updated: 2026-10-22

## Workspace consolidation

This prototype now lives at `Mudra/tools/phone-camera-prototype` inside the canonical SignLine
workspace. Its duplicated `app/frontend` copy was removed; the production React/Electron frontend
and FastAPI backend are maintained once under the workspace root. Local source videos, stills,
reviews and prototype models remain under this tool's ignored `training_data/` directory. The
default `start.bat` launch is local-only. Pass `--public-tunnel` explicitly to expose the camera
dashboard through Cloudflare; anyone with that URL can view and control the live feed.

## 2026-10-19 UI refresh

The copied SignLine Communication page now uses a shorter 4:3 camera stage on narrow screens,
smaller mobile gutters, tighter spacing, and wrapping mode controls. The phone camera dashboard
and tracker behavior are unchanged. The mobile layout has not yet been verified on the connected
Galaxy A27.

## Purpose

This is a local prototype for using the Samsung Galaxy A27 camera with MUDRA on the Windows
development machine. It receives the phone feed through Camo, overlays upper-body and finger
landmarks, and runs an experimental static model. The production SignLine app is maintained at the
workspace root, separately from this experimental camera tracker.

This prototype currently uses **Camo** as the phone-to-Windows transport and virtual camera. It is
not yet a custom Android-to-PC USB streaming implementation, a complete MUDRA app, or a system-wide
webcam driver of our own.

## Current state

The user connected the Galaxy A27 over USB, configured Camo, and confirmed that Camo Studio shows
the phone camera.

Latest camera probe on 2026-10-13 (`python test.py --list-cameras`) returned:

```text
Camera 0 [DirectShow]: 640x480 (brightness 133.4/255; video present)
Camera 0 [Media Foundation]: 640x480 (brightness 135.8/255; video present)
Camera 1 [DirectShow]: 1280x720 (brightness 0.0/255; BLACK/NO VIDEO)
Camera 1 [Media Foundation]: 1280x720 (brightness 106.8/255; video present)
```

These are four **index/backend combinations**, not four physical cameras. Camera 0 is a live
640x480 source (identity not confirmed). Camera 1 + DirectShow returns black. Camera 1 + Media
Foundation is the live 1280x720 source and is likely Camo, though OpenCV's numeric index does not
reveal a Windows device name. Its brightness changed between scans (201.4 then 106.8), as expected
when the camera scene/light changes; the important distinction is that it is not near-black.
DirectShow is the bad pairing. The current `auto` mode should reject its black frames and continue
to Media Foundation, but explicitly choosing Media Foundation is the deterministic recovery route.

The newest camera list check on 2026-10-15 found camera 0 + Media Foundation live at 1280x720,
camera 0 + DirectShow black, and camera 1 live at 640x480 through both backends. `start.bat` now
starts the camera/dashboard directly; select Static Signs or Dynamic Signs in the website. If the
phone is reconnected or the preview is not the expected camera, run `python test.py --list-cameras`
and update the `--camera` index in `start.bat`.

Windows device enumeration now identifies the two camera devices as `Camo` and `Integrated Camera`.
Given the latest OpenCV dimensions (1280x720 at camera 0 / Media Foundation, 640x480 at camera 1),
the launcher and CLI defaults use camera 0 / Media Foundation as the Camo source. OpenCV does not
expose the device name from this probe, so confirm the visible feed after reconnecting Camo.

`start.bat` starts the camera server in a separate terminal and uses the standalone dashboard at
`http://127.0.0.1:8765`. It runs locally by default. Pass `--public-tunnel` only when you intend to
make the camera dashboard available through the public Cloudflare URL; anyone with that URL can
view/control the feed. Quick Tunnel URLs change on restart. Camera capture and inference run on the
laptop, while a remote device only uses the browser client. The launcher requests 1920x1080 camera
capture, reads frames on a background thread, retains only the newest frame, feeds 960px-wide images
to the models and streams a preview up to 1280px wide. Earlier native-input runs measured about
7-10 FPS; the live frame timing panel reports camera, model and preview-encoding costs after tracking
starts. Inference uses CPU XNNPACK. `--max-width 640` remains available for more speed at reduced
landmark detail.
On 2026-10-18, the permanent URL redirected to a Quick Tunnel that returned HTTP 530, and the local
dashboard was not running. This is consistent with a stopped/stale tunnel target; start `start.bat`
with Camo connected and keep both windows open to publish the new tunnel destination.
The server opens the standalone local dashboard at `http://127.0.0.1:8765`. Select the camera and
press **Start tracking**. The dashboard offers
the original static model (20 labels, 8 live eligible), expanded intro model (32 labels, 10 live
eligible), and dynamic tracker (0 trained labels).

The dashboard's **Flip camera** control toggles horizontal mirroring for both preview and model
input. Both static models refresh an active segment every five seconds, clearing pose confirmation
and re-arming speech; dynamic mode is not periodically reset. The lower-torso activity zone defaults
to 1.0 shoulder widths below the shoulder line (maximum 2.0); real-camera behavior still needs live
testing. The intro model includes `HELLO_HI` and `GOOD`, but both are preview-only due to insufficient
source-group evidence; no alphabet data is available and the model cannot decode the full
introduction.

Known-good command from that probe:

```powershell
python test.py --camera 0 --backend msmf --capture-width 1920 --capture-height 1080 --show-clean-inset --open-web-ui
```

The probe measured 1280x720 from the likely Camo source. Asking OpenCV for 1920x1080 is only a
request; the actual negotiated frame size printed by the program is authoritative.

The first Python run was:

```powershell
python test.py --camera 1
```

OpenCV reported that camera index 1 opened through DirectShow at 1920x1080, and MediaPipe created
its CPU/XNNPACK delegate. The user then observed a black preview at about 1 FPS. It was not a
successful camera test: opening a device and reading a frame did not prove the source contained
live image pixels. The revised probe showed that this specific DirectShow pairing returns black,
while Media Foundation returns a live 1280x720 frame.

The current test uses Pose Landmarker Lite for torso/arm landmarks and Hand Landmarker for the
21-point finger skeleton on up to two hands. It also has an initial activity-boundary state machine:
hand visibility plus raised-wrist posture can sustain a static hold, wrist speed can trigger dynamic
activity, and a sustained rest interval confirms the end. The preview shows state, elapsed segment
time, wrist speed and raised status. Terminal output gives the first-active and rest-onset boundary
times, debounce confirmation times, and segment duration. The heuristic has not yet been validated
with a signer, does not calibrate a personal neutral pose, and
does not save video or landmark data. It is not an ISL recognizer.

During candidate, active, and ending states, the preview draws a padded bounding box around detected
hand landmarks. Labels/colors distinguish `SIGN CANDIDATE`, `SIGN IN ACTION`, `STATIC SIGN`, and
`SIGN ENDING`; this indicates the hand area associated with the activity heuristic, not an identified
sign class. Repeating one static sign after movement is intentionally not implemented yet.

The new `web/` dashboard is a standalone responsive browser interface served by `test.py` with
Python's standard-library HTTP server. It streams the annotated live preview, displays state/FPS/
hand and wrist activity, and controls start/rest debounce, static-vs-dynamic mode delay, movement
sensitivity and the raised-hand zone. Tuning is in-memory for that run; no Electron packaging or persistence has been added. Default
binding is localhost. Use `--web-host 0.0.0.0` only on a trusted private network to view/control
the PC camera session from another device; do not expose the unauthenticated server to the public
internet. No frame files are persisted; when LAN binding is enabled, annotated frames are streamed
to dashboard clients on that network.
The UI uses a warm ivory, deep blue, teal and saffron palette with editorial type, a three-step
first-use guide, an illustrated camera waiting state and compact instrument-panel controls. The
visual direction is inspired only by pre-2020 editorial website patterns; no post-2019 website
references or assets are used. The original recording screen was inspected only as a feature
reference; its frontend was not changed.

The dashboard has Studio and Learning tabs. Learning serves accepted, unaugmented still-image
references and compares a held pose against the same experimental static classifier. A label strip
above the reference image names the English folder label and clearly marks it unverified. With
`--static-demo`, classes without a finite rejection threshold are withheld from practice (currently
`HOW` and `WATER`). Studio stats now sit in a high-contrast vertical panel beside the camera, with
the boundary sliders below. Still references do not teach movement-based signs, and folder names
remain unverified.

It requests 1920x1080 capture but uses the negotiated
native size by default; `--max-width` optionally downsizes the inference image while keeping the
displayed preview at the full camera frame size. Pose runs every other frame by default. The
optional `--show-clean-inset` overlays an unannotated picture-in-picture from that same frame.
A live smoke test confirmed camera 1 + Media Foundation delivered 1280x720 and both model tasks
loaded and ran on that frame. No person/hands were in the captured frame, so landmark detections
were zero. A separate 12-frame live loop ran hand inference every frame and pose every other frame
at 25.7 frames/s at 1280x720; this is a short no-person smoke measurement, not a sustained
signing/FPS result.

## Repository contents

| Path | Purpose |
|---|---|
| `test.py` | Opens a Windows camera with OpenCV and overlays pose, finger landmarks and sign-activity boxes. |
| `web_dashboard.py`, `web/` | Local HTTP dashboard, live MJPEG preview, status API and responsive controls. |
| `tests/test_sign_activity.py` | Synthetic tests for activity debounce, static holds, pauses, and rest-onset boundaries. |
| `static_sign_prototype.py` | Extracts normalized still-image body/hand pose, writes a quality/provenance CSV, and trains an exploratory nearest-medoid baseline. |
| `tests/test_static_sign_prototype.py` | Tests filename grouping, grouped splits, geometric normalization, and rejection-threshold behavior. |
| `video_review.py`, `web/video-review.*` | Standalone local sentence-video reviewer for playback, sign/phase timestamps and annotation CSV; not a trained model or dashboard tab. |
| `tests/test_video_review.py` | Reviewer API, annotation storage, timestamp bounds and media-range tests. |
| `training_data/INVENTORY.md` | Consolidated snapshot of the available stills, videos, static vocabulary and review status. |
| `requirements.txt` | Python packages used by the original MUDRA project: MediaPipe Tasks and OpenCV. |
| `models/pose_landmarker_lite.task` | MediaPipe pose model, copied from the existing MUDRA project. |
| `models/hand_landmarker.task` | MediaPipe hand model copied into this prototype for 21-point finger tracking. |
| `training_data/Frames_Word_Level/` | Supplied word-named image folders; treat as extracted stills until source clips, labels, and provenance are verified. |
| `training_data/manifests/`, `training_data/derived/`, `training_data/review/`, `training_data/models/` | Generated CSV index, pose feature archive, review contact sheets, and exploratory prototype model/evaluation. |
| `README.md` | Install and run steps plus the current Camo troubleshooting notes. |
| `docs/architecture.md` | Current signal path, boundaries and diagnosis. |
| `plan.md` | Staged, measurable data-audit and model-validation plan for the future ISL video collection. |

The standalone reviewer is a separate process on port 8766. Run `python video_review.py` and keep
its terminal open while using `http://127.0.0.1:8766/`; closing the process stops the reviewer.
Browser smoke tests found native playback worked for one 1920x1080 MP4, while a 240x320 MP4 codec
needed the reviewer's OpenCV-decoded-frame fallback. Playback/seek fallback and annotation-save
behavior were tested, but no actual annotations have been saved. The user reported the separate
workflow was not working and approved separate Static Signs/Dynamic Signs dashboard sections. The
dynamic section currently shows tracker-only live diagnostics and links to this separate reviewer;
video testing and a reviewed-example library are still outstanding.

**Handover state:** the Video Lab tab, dynamic sequence extractor/model, and reviewed-example
learning library are not implemented. The standalone reviewer is not the Video Lab. Dynamic
recognition must remain visibly "not trained" until reviewed per-sign data exists and passes the
offline gates. Keep phrase-folder names and video filenames out of prediction labels.

## Worktree and source-data handoff

At handover, the branch is `main` tracking `origin/main` at `9d41db1`. The current UI/model
separation and 20-label changes are local work in progress. The sentence-video directory is
user-supplied source data and remains untracked: do not delete it, modify its videos, or
stage/commit/push the raw collection without explicit approval.

The live camera/dashboard path still does not save camera frames or landmarks and does not use the
main MUDRA recognition backend. An optional live static-pose demo now compares the current
landmarks to the separate still-image experiment; it does not recognize dynamic signs or qualify as
a live ISL recognizer.

The current `training_data/Frames_Word_Level/` inventory is documented in
[plan.md](plan.md): 131 label folders and 1,036 JPGs were present, including 17 empty folders;
there were no video files in that directory. These images are not yet verified as independent
training clips or as correctly labeled ISL examples.

The static pilot now selects twenty candidate folders: `I_ME_MINE_MY`, `YOU`, `FOOD`, `HELP`,
`WHAT`, `HOW`, `COLLEGE_SCHOOL`, `NICE`, `THANK`, `DO`, `WATER`, `NOT`, `HURT`, `CRYING`, `THAT`,
`REALLY`, `OLD_AGE`, `ANGRY`, `SOME ONE`, and `TELL`. The ten additions were selected from the
largest remaining image folders (9–21 images each), not based on verified glosses or static/dynamic
classification. Folder labels remain unverified. Schema v10
retains the shoulder-based torso fallback when hips are not reliably visible and uses face-relative
hand features from the existing Pose Landmarker nose/ears (eyes as fallback).
Each wrist is represented relative to the nose in face-width units, with an index-finger direction
dot product toward the nose. FOOD predictions are rejected unless the face anchor is available.
This is a lightweight head-point proxy, not a dense face mesh or expression recognizer.

Detector handedness is no longer passed into feature extraction or prediction and is not used by
the tracking-quality gate. Hands are identified by detection index and placed into feature slots by
body-relative spatial position. The camera overlay draws an index-finger arrow and an approximate
2D direction cue relative to the face anchor; it does not estimate finger depth or camera-facing
direction. Horizontal flip augmentation is still an unverified assumption for asymmetric signs.

The model stores separate medoids/rejection thresholds per label and exact detected-hand count. It
compares one-hand input only with one-hand profiles and two-hand input only with two-hand profiles;
zero is not a wildcard. `HELP` is trained only from two-hand examples. The current model contains
prototypes for the selected folders, but a label/count profile is used for accepted live output only
when it has a finite rejection threshold. None of the ten new labels has one. The manifest has 11
accepted one-hand and 13 two-hand `HELP` originals, but only the 13 two-hand samples train `HELP`.
Count assignment is still brittle: the unchanged `HELP (1).jpg` source is recorded with one
detected hand in the manifest but was detected as two hands on repeated `predict-image` runs.
Exact-count filtering prevents cross-count prototype comparisons, but it cannot stabilize the raw
detector output; do not treat this smoke test as proof of reliable one/two-hand live recognition.

Observed 2026-10-14: THANK has worked when signed with the left hand; attempts with the right hand
were repeatedly classified as FOOD. Handedness is not an explicit model input, but this remains an
unresolved hand-side/asymmetry problem in the learned geometry and mirror augmentation. Treat this
as a demo limitation, not a reliable property of either sign.

Optional speech output (`--speak-recognized`) speaks an accepted static-pose label once it remains
unchanged for 0.5 seconds by default; use `--speech-hold` to adjust the interval. It may speak the
same label again after a static refresh or another segment boundary. Speech runs on a worker thread and handles multiple
utterances on the same engine. Sentence-context prediction is only a future experiment: after dynamic sign
sequences and reviewed glosses exist, evaluate a constrained next-gloss reranker separately, and
never let it silently turn an unknown visual match into a recognized sign.

The Learning tab overlays gold circles/crosshairs at the selected prototype's expected wrist
positions. It projects torso-relative medoid coordinates through the live shoulder/torso anchor and
shows the expected hand count. It is a position cue, not the reference's full finger shape, and
requires visible shoulders. Targets beyond the camera frame are drawn at the nearest visible edge
and labeled with their off-frame direction.

The latest 20-folder schema-v10 model accepted 888 augmented rows (473 original images) and scored
19.4% (13/67) on the original inferred-filename-group test split. This is a weak exploratory score:
labels/source groups are unverified and it does not measure performance on new signers. None of
the ten added labels has a calibrated rejection threshold; those labels safely return unknown and
are omitted from Learning. The live demo uses profile-specific group-holdout distance thresholds
and does not recognize movement.

While the dashboard is running, `http://127.0.0.1:8765/images` serves a searchable/filterable index
of all 1,036 JPGs under `training_data/Frames_Word_Level/`, including folders not selected for
training. Images are loaded through registered numeric IDs constrained to this dataset root and
the page uses native lazy image loading.

Product direction: make a reviewed, provenance-backed sign catalog the source of truth and use the
model as a secondary similarity retriever that can reject unknowns. Keep current small-pilot data
in versioned NumPy landmark files and the CSV manifest; do not add a vector database at this scale.
Dynamic signs require ordered sequences; word templates alone do not provide sentence translation.

Try the live static-pose demo (the schema-v10 model is already regenerated; rerun `prepare` and
`train` only when the selected source images or manual manifest labels/groups change):

```powershell
python static_sign_prototype.py train
python test.py --camera 0 --backend msmf --static-demo --open-web-ui
```

`--static-demo` turns on the local dashboard automatically. It evaluates only while the boundary
tracker reports a static hold, requires a stable candidate before showing it, and exposes the
nearest match/distance in the UI. Demo rejection thresholds are estimated with inferred
filename-group holdouts instead of the small validation split; they are explicitly experimental.
All five names still come from unverified folder labels, especially the combined
`I_ME_MINE_MY` class. Treat displayed matches as prototype feedback, not confirmed ISL recognition.

Run the separate offline still-image experiment from this folder:

```powershell
python static_sign_prototype.py prepare
python static_sign_prototype.py train
python static_sign_prototype.py predict-image "training_data\Frames_Word_Level\YOU\YOU (1).jpg"
```

Review `training_data/manifests/selected_static_samples.csv` and the per-label sheets under
`training_data/review/`. Every label is marked unverified pending review. The CSV can be edited to
correct class labels, enter verified gloss/motion/source/signer/session metadata, and replace
filename-inferred groups with reviewed source IDs; rerunning `prepare` preserves those human-edited
fields. Movement-dependent signs need ordered video/landmark sequences. Learning rotates accepted
original references; with `--static-demo`, labels lacking a calibrated rejection threshold are
excluded.

## Run

From this folder in PowerShell:

```powershell
python -m pip install -r requirements.txt
python test.py --list-cameras
python test.py --camera 0 --backend msmf --capture-width 1920 --capture-height 1080
```

`--list-cameras` probes each OpenCV camera index through both DirectShow and Media Foundation and
reports brightness for each readable index/backend pair. Use a pair that shows non-black frames.
The selected camera can be changed with `--camera N`; choose a backend with `--backend auto|dshow|msmf`.
By default, the program requests 1920x1080 and uses the negotiated frame size for inference. Change
the requested mode with `--capture-width N --capture-height N`; use `--max-width N` to downscale
inference frames when needed. For example:

```powershell
python test.py --camera 1 --backend msmf --capture-width 1920 --capture-height 1080 --show-clean-inset
```

The current default camera is index 0 / Media Foundation (1280x720 Camo candidate). Static-sign
tracking automatically refreshes active segments every five seconds.
Use the browser controls and live stream on this computer:

```powershell
python test.py --camera 0 --backend msmf --web-ui --open-web-ui
```

For a phone/tablet on the same trusted private Wi-Fi, add `--web-host 0.0.0.0` and open
`http://<PC-LAN-IP>:8765/` on the device. This controls the Windows camera tracker; it does not use
the phone browser's camera. The settings take effect during the run and return to their CLI/default
values at restart. The separate `Mudra` frontend was read as visual reference only; it was not edited.

To run the start/stop experiment, begin with:

```powershell
python test.py --camera 0 --backend msmf --start-hold 0.15 --stop-hold 0.7 --movement-threshold 1.0
```

The preview reports `STARTING`, `MOVING`, `STATIC HOLD` or `ENDING`, plus wrist speed and raised
status. The terminal prints start/end boundaries, confirmation times and segment duration.
`--start-hold` rejects
brief activity; `--stop-hold` delays ending across brief pauses; `--movement-threshold` is measured
in shoulder widths per second; `--raised-below-shoulder` adjusts the wrist-height cue in shoulder
widths. These are starting values, not ISL-standard thresholds. Test real signing before using
boundaries as training labels. See [README.md](README.md) and [docs/architecture.md](docs/architecture.md).

## Architecture and decisions

See [`docs/architecture.md`](docs/architecture.md).

- Use Camo's existing USB transport for the first milestone; this avoids writing and installing a
  custom Android APK, USB protocol and Windows virtual-camera source before we have a working video
  baseline.
- Consume Camo's **Camo Camera** virtual output from OpenCV. Camo Studio's preview is the upstream
  monitor, not the camera index our Python script should assume.
- Use Pose Landmarker Lite plus Hand Landmarker: pose draws torso and arms; Hand Landmarker estimates
  21 points per hand, including finger joints. This is landmark visualization only, not sign
  recognition.
- Request high-resolution phone video and process the negotiated native size by default. Downscale
  explicitly with `--max-width` when inference speed matters more than image detail.
- Use the boundary detector only as a visible, tunable experiment: raised static holds remain active;
  dynamic wrist movement can start activity; sustained inactivity confirms stop. There is no
  personalized neutral calibration or segment recording yet.
- Keep the initial flow local. No frame recording, upload, API or recognition logic is included.

## Camera diagnosis and recovery

1. **Black preview — backend mismatch confirmed by probe.** Camera 0 DirectShow returned 0.0
   brightness; Media Foundation returned a live 1280x720 Camo candidate frame. The current tracker
   defaults to `--camera 0 --backend msmf`. Use `--backend auto` to try both backends while skipping
   near-black frames. Camera 1 is another live source (640x480), likely the integrated camera.
2. **Tracking speed after enabling fingers is unmeasured.** Hand Landmarker runs every frame and
   Pose Landmarker every second frame. Start at the measured source size; if slow, use
   `--max-width 960`, then 640. This reduces inference size while the preview stays at full capture
   resolution. Compare the displayed FPS instead of assuming more pixels mean better live tracking.

The original black preview occurred because the first version accepted readable DirectShow frames
without validating their pixels. The revised probe skips near-black frames; force `msmf` to choose
the known working camera path directly.

The current route uses Camo over USB and then Camo's Windows virtual camera. OpenCV cannot read the
Android camera directly over ordinary USB without a phone-side streaming app/protocol and a matching
PC receiver; this prototype does not implement that alternative.

The MediaPipe startup messages shown in the first run are not evidence that the camera feed is
valid. The program got far enough to construct the model, but did not report the camera identity or
pixel brightness then.

Camo's documented flow is for Studio to publish a separate virtual **Camo Camera** device to other
apps. Camo says its Windows compatibility add-on is what makes video available to other apps; if
that add-on is missing, Studio may still show the phone while OpenCV cannot get the usable Camo
output. Studio's preview can remain open; Camo documents that the UI can be closed/minimized while
the service continues, and that another app can be using Camo's video. See [Camo setup](https://camo.com/support/camo/camo-getting-started),
[using Camo in other apps](https://camo.com/support/camo/camo-virtual-camera-in-apps), and
[Camo status/troubleshooting](https://camo.com/support/camo/how-to-use-camo).

### Diagnostic order

1. Keep the Galaxy connected by USB, Camo Camera running, and Camo Studio showing the phone.
2. Run `python test.py --list-cameras`; remember each row is one index/backend pair, not a unique
   physical device. Brightness near zero means that particular pair is black.
3. For the last known setup, run `python test.py --camera 0 --backend msmf`. If it is black or
   absent, rerun the scan and choose the live pair. Check Windows Camera with **Camo Camera**
   selected if the scan no longer finds a working Camo output.
4. Camo can show phone video in Studio while its separate Windows virtual-camera path is unavailable
   or enumerated differently. The Studio preview alone does not verify the OpenCV feed.
5. If tracking is slow, try `--max-width 960`, then `--max-width 640`; check whether FPS improves.

## Simple Camo setup for better tracking

- Choose the highest stable resolution profile in Camo Studio. The last measured Camo-like output
  was 1280x720; OpenCV cannot recover detail above what that virtual-camera output provides.
- Use the rear/main lens if easy to mount, point it steadily at the signer, and keep both hands in
  frame throughout signing.
- Use bright, even light from in front of the signer. Avoid strong backlight and dim rooms.
- Leave autofocus on unless it visibly hunts; if it does, lock focus on the signing area.
- While testing, disable digital zoom, portrait/background blur, filters/retouch, and auto-framing
  so the hands remain sharp, unaltered and uncropped.

The clean-camera inset is possible and implemented: `--show-clean-inset` displays a small
unannotated copy of the exact same current frame on top of the annotated preview. The large view is
still the original live camera image with pose and finger lines drawn over it; there is no separate
second camera capture.

## Next milestone / teammate checklist

The sentence-video collection is under `training_data/Videos_Sentence_Level/`: 101 phrase folders,
687 videos, approximately 7.1 GiB at the last inventory. The main dashboard now has separate
**Static Signs** and **Dynamic Signs** sections. Static Signs retains Studio and Learning. Dynamic
Signs currently shows tracker-only live diagnostics, a readiness warning, and a link to the separate
port-8766 reviewer; prerecorded prediction and a curated Learning library are not integrated yet.
Keep it separate from the still-image static model.

Proceed in this order:

1. Finish the Dynamic Signs area in the existing port-8765 dashboard: integrate video
   replay/selection and annotation, then make Learning show only reviewer-approved video examples.
   Until sequences and an evaluated model exist, show tracking/sequence quality and an explicit
   "dynamic model not trained" state—no guessed glosses.
2. Make clip review usable from that workflow. The interim `python video_review.py` reviewer runs
   separately at `http://127.0.0.1:8766/` and only while its process is open. It writes annotations
   to `training_data/manifests/video_sign_segments.csv`; no annotations were saved at handover.
   The standalone player includes decoded-frame fallback for browser-incompatible codecs.
3. Have an ISL-fluent reviewer verify the candidate glosses, actual sign boundaries, variants,
   provenance and consent. HELLO from `hi how are you`, THANK-YOU from `thank you so much`, and HELP
   from `help me` are hypotheses only. Folder and filename text is not per-sign ground truth.
4. Extract ordered, timestamped landmark sequences with visibility/missing masks and source links
   from accepted segments. Preserve original videos, variable durations, and natural signing speed.
5. Run the grouped, offline validation gates in [plan.md](plan.md), including unknown/rest/transition
   tests and timing ablation. If independent signer/session groups are insufficient, report only a
   pipeline demonstration. Test live on new natural-speed repetitions only after offline gates pass.

The video reviewer syntax checks and focused tests passed; the full unit suite passed 46 tests at
the time of its implementation. Browser smoke tests verified listing 687 clips, codec fallback
seeking, and boundary controls. These tests do not validate ISL labels or model accuracy. Keep this
work in `MudraPhoneWebcam`, not the separate main `Mudra` project.

Run synthetic tracker, bounding-box, dashboard endpoint and settings tests with
`python -m unittest discover -s tests`. These verify local logic and HTTP wiring, not real-camera
accuracy. Video reviewer and range-serving tests can be run with
`python -m unittest tests.test_video_review`.

## Change log

### 2026-10-14

- Removed blink-triggered resets and Face Landmarker from the webcam runtime. Static-sign tracking
  now refreshes every five seconds, the web UI can toggle horizontal camera mirroring, and the
  training-data inventory records the available source sets and current static-model labels.

- Added the standalone sentence-video reviewer, including codec fallback, timestamp/phase
  annotation, provenance fields and CSV persistence. It has not been used to save real annotations.
- Added separate Static Signs and Dynamic Signs sections to the dashboard. Dynamic currently shows
  tracker-only live diagnostics and links to the standalone reviewer; integrated video testing,
  reviewed-example Learning, and a dynamic model remain unimplemented.
- Expanded the static pilot to 20 candidate image-folder labels and rebuilt its report/model. The
  ten additions have no independent held-out groups or usable rejection thresholds and remain
  unavailable for live recognition.

### 2026-10-18

- Added a separately trained 32-class expanded static vocabulary and selectable intro model in both
  the SignLine workspace and local dashboard. Ten labels currently have experimental live thresholds;
  `HELLO_HI` and `GOOD` remain preview-only. Added explicit missing-word/alphabet limitations to the
  training inventory.
- Expanded the default lower-torso signing activity zone and aligned the dashboard range.

### 2026-10-13

- Created the separate phone-webcam prototype and standalone `test.py`.
- Changed from hand/finger landmarks to Pose Landmarker Lite, added 640px frame reduction, and
  added frame-brightness diagnostics after the first Camo/OpenCV run showed black video at about
  1 FPS. Automatic camera selection now checks DirectShow and Media Foundation independently and
  rejects near-black candidate frames instead of treating a successful device open as a valid feed.
- Added this handover and the architecture document so the current state, unknowns and diagnosis
  are explicit. No new phone-side or Windows-side app was built in this change.
- Updated capture to request 1920x1080 and infer at the negotiated native resolution by default,
  with optional `--max-width` downscaling. The user reports Camo Studio still receives the phone
  while OpenCV's feed has stopped; the failure is downstream of Studio's phone input, but the
  virtual-camera output/backend has not yet been checked. The updated capture mode and its FPS remain
  unverified on the connected phone.
- Recorded the two-index/four-backend camera probe: camera 1 + Media Foundation is live 1280x720,
  while camera 1 + DirectShow is black. Added this as the recovery route, enabled Hand Landmarker
  finger skeletons, added an optional clean-feed inset, and documented Camo setup tips. The combined
  models loaded and ran on a live camera frame; a 12-frame no-person smoke loop measured 25.7 FPS.
  Real hand visibility and sustained performance while signing remain unverified.
- Added the first sign-activity boundary experiment in `test.py`. Raised hands can sustain a static
  hold; wrist motion can start dynamic activity; sustained rest confirms stop. Added configurable
  debounce and threshold flags, visible state/activity measurements, and terminal boundary events.
  Updated the README and architecture. Synthetic transition tests are pending and live signer
  validation is still required; no video/landmark segments are persisted.
- Added a responsive, self-hosted browser dashboard to the phone-camera prototype, with in-session
  sliders and an MJPEG preview. Added a separate mode-change debounce to calm rapid static/dynamic
  label changes, plus API and tracker tests. Dashboard use from another device requires explicit
  trusted-LAN binding; no data files are saved or sent outside the configured host.
- Added `plan.md` with a staged, research-grounded approach to auditing supplied ISL videos,
  validating boundaries and labels, comparing against existing baselines, and requiring held-out
  and live-stream evidence before scaling training. The main `Mudra` project remains read-only.
- Added an offline static-pose pilot for `YOU`, `I_ME_MINE_MY`, and `WHAT`: tracking-quality
  extraction, a provenance/hash/group/split CSV, per-label contact sheets, torso-centered and
  body-scaled pose features, and a grouped nearest-medoid baseline. Metrics are exploratory because
  the folder labels and filename-derived groups are not verified; the live camera/dashboard path
  was not changed.
- Clarified in `plan.md` that static-vs-dynamic must be verified from the intended ISL gloss and a
  complete example, not inferred from English folder labels. Documented torso-centered,
  body-scaled landmark features, unit-direction arm-bone features, palm-width handshape
  normalization, and retained torso-relative wrist locations for signing-space information.
  Added minimum hand-confidence/size screening and documented its limits. Refreshed the dashboard
  guidance and camera placeholder without changing the tracker or adding capture/storage behavior.
- Added schema-v5 shoulders-only torso-center estimation when hips are not reliably visible,
  recorded the anchor source in the manifest, and added a `HELP` two-tracked-hands guard. Hand
  counts now use the raw detections, independent of left/right feature-slot assignment; the
  regenerated manifest has 11 one-hand and 13 two-hand accepted `HELP` originals. The labels remain
  unverified, and this guard does not establish reliable classification. Refreshed the model/report
  (37.5%, 15/40 original test images), added an unverified English folder-label
  banner above Learning references, and moved high-contrast runtime stats beside the camera.
  Twenty-six synthetic tests pass; live phone signing has not been retested.
- Narrowed the active pilot to `I_ME_MINE_MY`, `YOU`, `FOOD`, `HELP`, and `WHAT`. Bumped features
  to schema v6 (178 dimensions) and added face-relative wrist position plus index-finger direction
  using nose/ear pose landmarks; `FOOD` now fails closed without that anchor. Added the face anchor
  to the extraction manifest and camera overlay. Rebuilt the model from 552 accepted original/mirror
  rows; inferred-group test score is 43.6% (17/39), still exploratory. No new model or Python package
  was needed. A direct `FOOD (1).jpg` smoke prediction still returned `YOU`, confirming that
  face-relative features have not fixed multi-class accuracy. All 30 unit tests pass and the
  JavaScript syntax check is clean; live phone-camera recognition remains to be validated.
