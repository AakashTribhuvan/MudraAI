# SignLine desktop and mobile apps

The frontend can run inside Electron on Windows, macOS and Linux. Electron does not run on iOS or Android. The existing PWA manifest supports installing SignLine from a supported mobile browser; a Capacitor wrapper is the native-app path if a store-installable phone app is needed later.

`Mudra` is the canonical combined project: it already contains the React app, FastAPI service, training pipeline and validated v11a word model. The `MudraTraining` copy has the same v11a weights. The separate phone-webcam prototype is a Camo camera diagnostic; SignLine can use that camera through its normal camera selector, while a phone opening the PWA uses its own browser camera. Experimental static-pose matching from the prototype is not the default recognizer.

On phones, Communication requests a portrait camera stream when available, sizes the preview to the live frame, and draws each detected hand's padded box in camera coordinates clipped to the frame. This keeps the box inside the preview when the phone camera negotiates a different orientation or resolution.

## Desktop development (no installer)

1. Install a current Node.js LTS release and Python 3.12 or newer.
2. Double-click `start_application.bat` in the repository root. It installs missing dependencies, starts the FastAPI backend in a separate window, and starts the Electron development app. Keep the backend window open while using SignLine.

You can also run `npm run electron:dev` from `app/frontend` after installing dependencies and starting the backend. The window opens maximized with the SignLine title bar and borders, without the File/Edit/View menu. The Electron launcher clears inherited Node-only mode and uses its own port (3100), so another browser development server on port 3000 does not hijack the desktop window. Camera access is requested when a camera feature is opened. The existing `start_all.bat` still opens the web app in a browser.

For a production-like local run without making an installer, run `npm run build:electron`, then `npx electron .` from `app/frontend`. Keep the backend running and reachable at `http://localhost:8000` (or configure `VITE_API_URL` before building).

## Install on a phone

### Current route: install the PWA

Deploy the web frontend over HTTPS with its backend reachable from the phone, then:

- **Android:** open the HTTPS site in Chrome, tap the three-dot menu, and choose **Install app** (or **Add to Home screen**).
- **iPhone:** open the HTTPS site in Safari, tap **Share**, then **Add to Home Screen**.

Phone camera use requires HTTPS and permission. The backend URL must also be reachable from the phone; `localhost` on the phone means the phone itself. The desktop `start_application.bat` does not install the app on a phone.

### Future native wrapper: Capacitor

Electron cannot create phone apps. To make installable Android/iOS packages later, add Capacitor to the Vite frontend, set its web output directory to `dist`, add the Android and iOS platforms, and create the native projects. Configure camera permissions and use a phone-reachable HTTPS backend URL, then build/sign the Android APK/AAB in Android Studio and the iOS app in Xcode (macOS required). Test camera access, routing, storage, and API connectivity on physical devices before release. This is a separate mobile build path; this Electron setup does not create phone packages.

## Packaging later

The Electron Builder metadata and SignLine icons are in `app/frontend/package.json` and `app/frontend/electron/`. Once the app and backend distribution strategy are settled, installers can be generated with platform-appropriate Electron Builder runs. A Windows build cannot produce a properly signed iOS app; iOS requires Xcode on macOS. At present installers are intentionally not generated, and the Python backend/model files are not bundled with Electron.
