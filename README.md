# MUDRA: Indian Sign Language Learning & Real-Time Communication

**The web app is branded SignLine** (2026-10-16): demo sign-in with User and Admin workspaces, Game Mode
lessons, a sign library (words, fingerspelling letters, phrases), an emergency-help assistant and an admin overview.
Sign-in is a demo with no real accounts; see `HANDOVER.md` Section 5.2. Code and packages keep the name MUDRA.

MUDRA turns Indian Sign Language (ISL) into text and speech from an ordinary webcam, and turns
text or speech into ISL with a 3D avatar. It combines a React web app, a FastAPI recognition
service and a PyTorch landmark model trained on public and team-recorded ISL data.

Only body and hand landmark coordinates leave the browser, never camera images.

**Taking over the project (people or AI models): read [`HANDOVER.md`](HANDOVER.md) first.** It is
updated at every push and documents the architecture, interfaces, data, models and open problems.

## Status (23 Oct 2026)

| Area | State |
|---|---|
| Web app (Home, Communication, Learning, Game Mode, Progress, Teach a sign, Settings; Admin pages) | Working; the avatar signs the recorded vocabulary and fingerspells other words; installable on phones (web app manifest) |
| Recognition model | `isl_mudra_combined_v15b_bilstm`: 339 signs + "no sign" (v9 + the app's 214 recorded takes of 33 signs, incl. YES, NO, MY, NAME and 9 phrases; your held-out takes 62/66, 0 mix-ups); v8's recipe (hands resting in view, low camera frame rates) plus the app's recorded takes; not worse than v8 on any test stream (the numbers below are for an earlier model, v6; v8 and v9 are in HANDOVER Section 14) |
| Recognition modes | SignLine words (v15b, default); Merged: v15b + 8 words from the teammate's models, each word from the most accurate model that knows it; Auto: Merged words and A-Z fingerspelling from one camera stream (`docs/TEAMMATE_MODELS.md`) |
| Avatar signs real words | Typed or spoken words that SignLine recognises are performed from real recorded signs (landmarks of the training videos turned into arm, hand and finger rotations; an approximation). The owner's own *Teach a sign* takes are used for words without a training video. Other words are fingerspelled with the alphabet photos' hand shapes (arms placed in front of the chest) and the viewer says so; NAMASTE is built from landmarks. The old hand-made word animations are switched off (they did not show the signs) |
| Avatar choice | Drop a rigged Mixamo `.fbx` or `.glb` into `app/frontend/src/assets/avatars/` and pick it in the dropdown on the 3D avatar; the signs are retargeted from the default Y Bot |
| Model choice (from a teammate's branch) | Communication lets you pick the word model; each shows its measured accuracy. The default is v15b. A teammate's static-pose model is an opt-in experiment: on our test clips it got 0 of 22 shared-word signs right (v9: 22 of 22) |
| Barriers between words | Taught words get their own boundary so look-alike words are not mixed up; *Teach a sign* warns about inconsistent takes and look-alike words. Measured with 150 taught words: mix-ups 7.0% → 5.0%. See [`docs/SCALING_VOCABULARY.md`](docs/SCALING_VOCABULARY.md) |
| Live recognition (Communication) | *Words*: sign start/end detection, automatic word adding, spoken sentence, "How to sign it" examples; *Spell* and *Phrase* modes (below) |
| New words from the Teach a sign page | Work immediately (few-shot), stored for retraining |
| Data pipeline | MediaPipe landmarks, leakage-safe splits, label audit |
| Fingerspelling letters A-Z and 99 phrases | New (from the team's alphabet photos and sentence videos): letters 70.9% / phrases 58.3% first guess on held-out data (phrases are always chosen by the user from the top guesses); in Learning, Game Mode and Communication (*Spell* / *Phrase*). Not yet measured on a live webcam. See [`docs/SIGN_LIBRARY.md`](docs/SIGN_LIBRARY.md) |
| Emergency help (SOS circle, bottom-right) | Prototype in demo mode: explain an emergency with buttons, signs or a personal gesture, check and correct what SignLine understood, confirm, share with an in-app demo responder desk. Contacts no one; signs are simulated because the model does not know most emergency signs. In a real emergency in India, call 112. See [`docs/SIGNLINE_EMERGENCY.md`](docs/SIGNLINE_EMERGENCY.md) |
| Google Meet / Zoom (new, prototype) | Chrome extension `app/meeting-extension` puts the MUDRA panel in the meeting: what others say is recognised on this computer and signed by the avatar while they speak, with a live caption; your signs are spoken in a local voice into a virtual cable used as the meeting microphone. Tested on a mock meeting page and with recorded speech, not yet in a real call (`docs/MEETING_INTEGRATION.md`) |
| Known gap | Live use on a laptop webcam is not yet reliable: the browser tracks at ~6 fps on some machines and the model has seen few signers (66.7% on unseen recording sessions) |
| Honest reading of the numbers | Test clips come from the same signers and sessions as training clips, so the 97-99% figures below are "new takes by known signers". Accuracy for new people has not been measured. See the audit: [`docs/AUDIT_2026-10-13.md`](docs/AUDIT_2026-10-13.md) |

## Results

Live-use benchmark: held-out signs joined into continuous streams with transitions and idle
movement, delivered at 12 frames/s like a laptop webcam. Settings were chosen on validation
streams; these are held-out test numbers, with the corrected labels (see Data quality). The test
clips share signers and recording sessions with training (90% of INCLUDE test clips have a same-word
training take 1-2 files away; every team-word test clip a take within 2 minutes), and the streams are
simulated, so these are upper bounds for new users. Only `ok` words are scored; the app also adds
unsure guesses (50-70%), which on the same streams changes correct words by 0 to +1.7 points and wrong
words by -0.5 to +0.9 points.

| Source | Framing | Correct | Wrong | Missed |
|---|---|---|---|---|
| INCLUDE original videos, MediaPipe landmarks (498 signs) | normal | 93.8% | 1.2% | 5.0% |
| | close-up webcam | 90.6% | 2.0% | 7.4% |
| INCLUDE pose release (816 signs) | normal | 92.6% | 2.1% | 5.3% |
| | close-up webcam | 90.0% | 2.0% | 8.1% |
| Team words (181 signs) | normal | 97.2% | 0.0% | 2.8% |
| | close-up webcam | 90.1% | 2.2% | 7.7% |

Close-up webcam = the frame ends between the chin and the waist and the hands are only visible
when raised, which is how most people use the app. Extra words: at most 0.002 per sign; false
words while idle: at most 0.1 per minute.

The first row is the closest proxy for the webcam: the same landmark pipeline the browser runs.
The first model, trained on the pose release only, scored 41.8% correct / 7.0% wrong on that row.

**Data quality.** An embedding audit found that one INCLUDE recording session (Home words,
MVI_4896-4955) has its folder boundaries off by one: the third take of each word is the next word's
sign. 20 clips (both the video and the pose-release copy) were relabelled; the evidence and the list
are in `HANDOVER.md` Section 12 (local copy: `data/metadata/splits/combined_words_v3/label_fixes.json`). No video is filed under two
words, and the pose-release and video copies of every clip agree (median correlation 0.96).

Clip-level accuracy (complete pre-cut clips, corrected labels; same signers and sessions as training):
- INCLUDE original videos re-extracted with MediaPipe: 99.0%; team words 99.4% (new takes recorded
  minutes after training takes of the same word).
- INCLUDE 263 words, pose release: 97.4%. Published references on the same official split: 85.6%
  (original paper), 93.5% (OpenHands SL-GCN), 97.7% (HWGAT).
- All 997 test recordings counted once (the INCLUDE video and pose copies of a recording are one
  recording): 97.6%. The often-quoted 98.2% counts 498 recordings twice.
- Team words, on a recording day held out from training: 98.0% (99 clips of 20 words, earlier
  combined model; probably the same people on another day).
- Earlier rule-based / Random-Forest system: about 44%.

New words from recordings (few-shot, no retraining): tested on 25 team words the model never saw,
from k recorded examples each (held-out test clips, averaged over 5 draws). The test clips were
recorded in the same sessions as the examples (within 2 minutes), so these are same-session numbers.

| Examples per word | Recognised | Wrong word | Missed | Known words taken over |
|---|---|---|---|---|
| 1 | 70.1% | 12.0% | 17.9% | 0.10% |
| 3 | 79.5% | 11.5% | 9.1% | 0.14% |
| 5 | 81.6% | 10.9% | 7.5% | 0.14% |

Almost all wrong words (10.1-10.7 points) are the model confidently reading the new sign as a
known look-alike; without custom words that happens for 18.7% of them. Retraining with the
recordings is the real fix, so recordings are stored in the training format.

**New recording conditions (the honest number for new users).** INCLUDE was recorded in sessions
(day, outfit, camera setup). Holding out 9 whole sessions from training and testing on them:

| Unseen sessions | Clip accuracy | Top-5 | Streams: correct | Streams: wrong |
|---|---|---|---|---|
| INCLUDE real video (529 clips) | 66.7% (96.5% when sessions are mixed); pose-release copies 66.8% | 91.3% | 48.4% | 9.5% |
| Team words (181 clips; **not** held out by day: same signers and sessions as training) | 96.7% | 100% | 91.2% | 0% |

Every held-out word had been trained on only 2-4 sessions, and accuracy follows that count
(64% with 2, 69% with 3-4, 92% with 5+). Mirror, stronger geometric augmentation and pose
dropout did not close the gap (60-65%), so more signers per word is the fix, not more training
tricks. A signer can appear in several sessions, so this is session-independent, not strictly
signer-independent. Raising the confidence bar trades correct words for fewer wrong ones on new
users: at 0.9, 39.3% correct / 5.3% wrong; at 0.95, 30.1% / 2.3%.

**Limits:**
- No dataset has signer IDs; the session hold-out above is the closest measured proxy for new
  people. Recordings made in the app carry the signer's name, so they can provide the real test.
- The idle movements in the benchmark are synthetic.
- NAMASTE, GOODBYE, HOME and PERSON have no training data yet (WATER, HELP, YES and NO were recorded on *Teach a sign*).

## How it works

```
Browser                               Backend (FastAPI)                      Model
MediaPipe Tasks hands + pose  --WS--> /stream: SignSegmenter            --> BiLSTM over 24 frames
(on device, ~30 fps)                  resting -> signing -> ended            of hand shape, hand
                                      classify each whole sign once          position vs shoulders,
UI: Signing... / "Looks like X" <---- (3-cut smoothing, threshold,           upper-body pose
    word added / "Did you mean"        cooldown, "no sign" class)
```

1. **Landmarks:** 2 x 21 hand points and 33 pose points per frame. Hands are matched to the
   signer's left/right by the nearest pose wrist.
2. **Features (162 per frame):** wrist-relative hand shape plus hand location relative to the
   shoulders, so chin vs chest signs stay distinct.
3. **Segmentation:** a sign starts when hands are raised or moving, and ends after 0.4 s of rest.
   Tuned on validation streams (`training/configs/streaming.json`).
4. **Decision:** the whole sign is classified once. A "no sign yet" class covers rest, partial
   signs and fidgets; a word is added at 70% confidence or more (chosen on validation streams); between 50% and 70% the best guess is added and two alternatives are offered as one-tap replacements.

## Quick start (Windows)

1. Run `start_all.bat`. It creates the Python environment, installs dependencies and starts both servers.
2. Open http://localhost:3000, choose the demo user, then **Communication -> Start Camera Stream**.
3. *Words*: sign one word from start to finish, then lower your hands. The word is added when the sign ends.
   *Spell*: hold each letter still for about a second. *Phrase*: sign one phrase, then pick it from the top guesses.
4. To teach it a new word, open **Teach a sign**, enter your name, pick or type the word and record it
   5 times. The word works in Communication straight away and is saved for the next training run.
5. **SOS** (bottom-right, every page) opens the demo emergency assistant; it contacts no one.

Backend: http://localhost:8000 (`/docs`, `/health`, `/labels`, `/predict`, `/recordings`,
`/reference/{sign}`, `/library`, `/letters/predict`, `/phrases/predict`, WebSocket `/stream`). If port 8000 is taken, start the backend with
`MUDRA_PORT=8010` and set `VITE_API_URL=http://localhost:8010` in `app/frontend/.env`.
Emergency API: `/api/v1/emergency/...`; responder desk at http://localhost:3000/responder (user `demo.responder`, password in `app/backend/var/demo_responder_credentials.txt` after the first start, or set `SIGNLINE_EM_DEMO_RESPONDER_PASSWORD`).
Requires Python 3.12+ (tested on 3.14) and Node 20+. An NVIDIA GPU is optional (training only).
The backend `requirements.txt` pins the CUDA 12.8 PyTorch build used for training; for a CPU-only
machine install `torch` from the default index instead.

## Project structure

```
Mudra/
├── README.md, HANDOVER.md        overview; full handover (architecture, interfaces, models, change log) - read first
├── AGENTS.md, CLAUDE.md          rules for people and AI agents working on the code
├── start_all.bat                 Windows: creates the Python environment, installs packages, downloads the speech model,
│                                 starts both servers
├── tools/                        setup/ (speech model download, Chrome GPU setting), live_benchmark/ (camera tracking +
│                                 recognition on this PC), meeting_tests/ (speech, voice route, extension tests); see its README
├── app/
│   ├── backend/                  FastAPI server (python main.py; port MUDRA_PORT, default 8000)
│   │   ├── main.py               entry point
│   │   ├── mudra_api/            recognition API
│   │   │   ├── app.py            routes: /health /labels /models /predict /stream /recordings /reference /library ...
│   │   │   ├── settings.py       environment settings (model paths, ports, CORS)
│   │   │   ├── recognizer.py     word model loading + whole-clip prediction
│   │   │   ├── recordings.py     Teach a sign: saved takes, custom-word bank
│   │   │   ├── references.py     one real example per word (the avatar's signs)
│   │   │   ├── library.py        fingerspelling letters and phrases
│   │   │   ├── extra_models.py, auto_mode.py, static_pose.py, static_top10.py, isl500_video.py, safe_joblib.py
│   │   │   │                     teammate models, Merged and Auto modes, safe loading of their files
│   │   │   └── speech.py, speech_api.py   meetings: /tts (local voice), /asr + /captions (local speech recognition)
│   │   ├── signline_emergency/   emergency assistant: config/, services/, api/, migrations/, models.py, retention.py
│   │   ├── models/               deployed models (model.pt + model_card.json each): word model, letters, phrases,
│   │   │                         teammate models
│   │   ├── tests/                pytest: API, streaming, library, references, speech, emergency/
│   │   ├── var/                  local data, not in git: emergency database and videos, speech models
│   │   └── requirements.txt, .env.example, alembic.ini
│   ├── frontend/                 React 18 + Vite + Tailwind + Three.js (npm run dev, port 3000)
│   │   ├── public/               MediaPipe models/ and wasm/, ybot.glb avatar, brand/ icons, web app manifest
│   │   └── src/
│   │       ├── App.jsx, main.jsx routes, role guards, start-up loader
│   │       ├── pages/            Communicate, Practice, Record (Teach a sign), Learn, Progress, Settings;
│   │       │                     user/, admin/, auth/, game/, emergency/, legal/, meeting/ (Meet / Zoom panel),
│   │       │                     dev/ (avatar QA, development builds only)
│   │       ├── features/         emergency/, game/, practice/, meeting/ (captions, sign planning, voice), vocabulary.js
│   │       ├── components/       shared UI: ui/, brand/, avatar/, communicate/, practice/, AccessibilityModal
│   │       ├── avatar/           3D avatar: useAvatarRenderer.js, landmarkRetarget.js (recorded signs -> bones),
│   │       │                     synthSigns.js, Animations/ (letter fallbacks, rest pose), avatarCatalog.js
│   │       ├── services/         handTracker (MediaPipe), signRecognizer (WS /stream), islDictionary, clients
│   │       ├── context/, layouts/  accessibility and session state; app shell
│   │       └── assets/avatars/   optional extra avatar files (local only, see its README)
│   └── meeting-extension/        Chrome extension for Google Meet / Zoom Web (load unpacked; docs/MEETING_INTEGRATION.md)
├── training/                     model training (Python package mudra_ml)
│   ├── src/mudra_ml/             landmarks, features, datasets, models, training, evaluation, streaming segmenter
│   ├── scripts/                  extract, split, train, export, compare / evaluate (see training/README.md)
│   ├── configs/                  data.yaml, streaming.json, fewshot.json, experiments/ (one YAML per model version)
│   └── tests/                    unit and end-to-end tests
├── data/                         README only: datasets, landmarks, recordings and splits stay on the local machine
└── docs/
    ├── MEETING_INTEGRATION.md    Google Meet / Zoom: extension, speech in and out, setup, tests, limits
    ├── SIGN_LIBRARY.md           fingerspelling letters and phrases
    ├── SIGNLINE_EMERGENCY.md     emergency assistant
    ├── SCALING_VOCABULARY.md     barriers between words, growing the vocabulary
    ├── TEAMMATE_MODELS.md        teammate models, Merged and Auto modes
    ├── AUDIT_2026-10-13.md       audit of results and claims
    ├── architecture.md           short architecture table
    ├── MUDRA_MASTER_PLAN.md      original analysis, dataset research, dated progress log
    ├── blueprints/               versioned development plans
    └── planning/                 the client's original review documents
```

Kept out of git (`.gitignore`): `node_modules/`, `.venv/`, datasets and recordings (`data/raw`, `data/processed`,
`data/metadata`), training runs (`experiments/`), `app/backend/var/` (local databases, emergency videos, speech
models), extra avatar files, `.env` files.

## Training workflow

See `training/README.md`. In short:

```powershell
app\backend\.venv\Scripts\activate
python training/scripts/extract_landmarks.py --source own --input data/raw/own
python training/scripts/make_splits.py --manifests data/metadata/manifest_own.csv --name own
python training/scripts/train.py --config mudra_words.yaml --name mudra_words
python training/scripts/export_model.py --run experiments/runs/mudra_words_<time> --name <model> --data-source combined
python training/scripts/eval_continuous.py --model app/backend/models/<model> \
    --split data/metadata/splits/combined_words_v3/val.csv --scenarios normal,closeup --fps-cap 12 --tune-threshold-only
```

`mudra_words.yaml` reproduces the deployed model (about 45 minutes on an RTX 5060).
Letters and phrases are rebuilt with the steps in `docs/SIGN_LIBRARY.md`.
Tests: `pytest training app/backend/tests` (112) and `npm test` in `app/frontend` (18).

## Data and licences

- **INCLUDE** (Sridhar et al., ACM MM 2020): pose release via AI4Bharat OpenHands and the original videos (Zenodo), CC-BY-4.0.
- **Team ISL recordings** (61 words, 101 sentences, 12,637 alphabet photos): provided by the MUDRA team; the
  sentences feed the phrase model and the photos the letter model. Source and licence to be confirmed before any
  public release (the sentence videos may be ISL-CSLTR, CC-BY-4.0).
- Raw videos and extracted landmarks are not committed; see `data/README.md` for the layout.

## Roadmap

Done: sign start/end detection, smoothing and commit rules, close-up webcam robustness, in-app
recording and few-shot new words, session-independent test, INCLUDE label audit and fixes, SignLine redesign
with Game Mode, fingerspelling letters and phrases, the emergency-help prototype.

Next, in order (details: `HANDOVER.md` Section 16):
1. a live webcam test of words, letters and phrases, with session saving so failures can be replayed
2. more signers: teammates recording on *Teach a sign* (emergency words first), then a v9 model and real
   recognition in the emergency assistant; request FDMSE-ISL, CISLR and iSign
3. a test set of people kept out of training, to measure accuracy for new people
4. more phrase videos and webcam-quality letter clips
5. real accounts, review by Deaf ISL users or an ISL teacher, confirmed data licences
6. HTTPS hosting, in-browser inference, CI for the tests

Tried and not adopted (measured, see the progress log): velocity features, ending a sign when the
hands are held still, a zoomed hand search during extraction, hand gap filling, pose dropout,
mirror augmentation, and an RGB CNN branch.

Details are in `docs/MUDRA_MASTER_PLAN.md`.
