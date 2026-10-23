# SignLine Avatar Lab

An isolated Vite site for diagnosing and tuning the 3D signing avatar. It reuses the production renderer, avatar models, and backend sign-reference endpoint; changes made here do not alter the main website's pages or navigation.

## Run locally

Start the MUDRA backend on port 8000, then run the lab from this directory:

```powershell
npm run dev
```

Open `http://localhost:5173`. The backend's default CORS configuration already allows this origin. To use another backend, set `VITE_API_URL` before starting Vite.

The lab reports backend availability, reference clip count, current avatar/rig state, animation queue, active step, and clip frame progress. Use the text box, sample phrases, model picker, and speed slider to reproduce playback issues. Keep any production integration behind a separate review after the signs have been checked with an ISL-fluent signer.
