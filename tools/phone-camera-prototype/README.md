# Phone camera and tracking prototype

This optional local tool is part of the unified Mudra/SignLine workspace. It uses Camo or another
Windows camera source for OpenCV/MediaPipe tracking, with its own dashboard and video-review tools.
The main SignLine web/Electron app lives at the repository root under `app/`; this prototype's old
copied frontend was removed so it cannot drift from the main app.

## Run locally

1. Connect the phone camera with Camo, or use a local webcam.
2. Install `requirements.txt` into the workspace Python environment at `..\..\..\.venv`.
3. Double-click `start.bat` and use the local dashboard at `http://127.0.0.1:8765`.

The default launcher is local-only. `start.bat --public-tunnel` starts the optional Cloudflare
tunnel; anyone who gets that URL can view/control the live camera dashboard. Do not use it unless
you intend to expose that feed.

## Kept data

The `training_data/` directory contains the prototype's local source videos, stills, derived
landmarks, review images and experimental models. It is ignored by the main Git repo to avoid
publishing signer media or local evaluation artifacts.

The static-pose recognizer is experimental. The production app continues to use the validated
SignLine model and its model-selection controls.
