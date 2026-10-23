# tools/

Setup helpers and the test tools behind the measurements in `HANDOVER.md` (Section 21) and
`docs/MEETING_INTEGRATION.md`. Everything they write goes to `experiments/tools/` (git-ignored). Run them from the
repository root with the backend's Python (`app\backend\.venv\Scripts\python.exe`) and Node 20+; most need the app
running (`start_all.bat`: web app on 3000, server on 8000, or on 8010 with `MUDRA_PORT=8010` +
`VITE_API_URL=http://localhost:8010` as on the development PC; the tools use `http://127.0.0.1:3000`).

## setup/

| Tool | What it does |
|---|---|
| `download_asr_model.py` | Downloads the streaming speech model the meeting features need into `app/backend/var/asr_models/` (default 2023-06-26 zipformer, 310 MB download, 68 MB kept; `--model small` = 20M, 42 MB). `start_all.bat` runs it when the model is missing. |
| `chrome_gpu_high_performance.ps1` | Lets Chrome use the laptop's NVIDIA / AMD card (Windows per-app GPU preference); `-Undo` removes it. Restart Chrome afterwards (`chrome://restart`). Measured on the development laptop: hand tracking 46 -> 17 ms per frame. |

## live_benchmark/ - camera tracking and recognition, end to end

A real Chrome window (off-screen) opens the real Communicate page with a fake webcam that plays held-out test signs,
so frame rate, tracking time and recognition can be measured on this computer's own GPU.

```
python tools/live_benchmark/make_fake_cam.py 8            # 8 held-out INCLUDE test signs -> fakecam/signs.y4m (25 fps; needs data/)
python tools/live_benchmark/make_fake_cam.py 8 12 fakecam12   # same at 12 fps (a webcam in dim light)
node tools/live_benchmark/bench_cam.mjs run1 GPU 36       # label, delegate preset, seconds; FLAGS="--force_high_performance_gpu", CAMDIR=fakecam12
python tools/live_benchmark/bench_score.py run1           # tracking fps / ms, signs right / wrong, sign end -> word latency
node tools/live_benchmark/gpu_probe.mjs                   # which GPU Chrome draws with
node tools/live_benchmark/cam_caps.mjs                    # the real webcam's capabilities (turns it on for ~2 s, keeps no image)
```

## meeting_tests/ - Google Meet / Zoom features

| Tool | What it does |
|---|---|
| `make_speech_wav.ps1` | Test speech with known word timings (Windows voice -> 16 kHz WAV + JSON). |
| `asr_compare.py <model folders>` | Word errors, first-partial delay and speed of speech models on the WAVs in `experiments/tools/meeting_tests/asr_compare/` (`sN.wav`, `sN.txt`, `sN.json`). |
| `meeting_e2e.mjs <wav> <words.json> <out.png>` | The meeting panel with the WAV as a fake microphone: word start -> caption -> commit -> avatar latency (`summarize_e2e.py` prints it). |
| `cable_e2e.mjs` | Outgoing voice through VB-Audio Virtual Cable: Test route, then what the meeting microphone (CABLE Output) carries, transcribed locally. Needs VB-Cable. |
| `prep_ext_test.py` | Builds a test copy of the extension (mock-meeting permission, fixed ID from a local key, test hooks) and the mock meeting page; prints the extension ID. |
| `ext_e2e.mjs <extDir> <extId> <out.png>` | Extension on the mock meeting (headless): overlay, panel, captions from a caption room (`room_producer.mjs` streams the WAV), Stop. |
| `ext_real.mjs <extDir> <extId> <out.png>` | Real tab capture: a visible Chrome window and the extension's shortcut Alt+Shift+M pressed with real keystrokes (`focus_keys.ps1`). |

```
python tools/meeting_tests/prep_ext_test.py                 # -> extension ID
python -m http.server 5599 --bind 127.0.0.1 -d experiments/tools/meeting_tests/mock_meeting
node tools/meeting_tests/ext_e2e.mjs "<repo>\experiments\tools\meeting_tests\ext_test" <ID> out.png
```

Tab capture cannot be forced without a user action (Chrome refuses the stream with `--allowlisted-extension-id`),
which is why `ext_real.mjs` uses real keystrokes.
