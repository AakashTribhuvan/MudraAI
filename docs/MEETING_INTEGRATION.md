# MUDRA in online meetings (Google Meet, Zoom)

Two directions, all on the user's computer:

- **Incoming** (hearing participant -> deaf user): the meeting's audio is recognised locally while people speak; the
  MUDRA avatar signs stable words and short phrases from MUDRA's recorded vocabulary, with a live caption.
- **Outgoing** (deaf user -> others): the user's signs (webcam -> MUDRA word model) become speech in a local Windows
  voice, played into a virtual audio cable that the meeting uses as its microphone.

Status on 2026-10-21 (details in "What was tested"):

| Mode | Status |
|---|---|
| Chrome extension overlay on Meet / Zoom Web | Built. Tested on a local mock meeting page in a normal Chrome window: real tab audio capture started by the extension's shortcut (Alt+Shift+M, real keystrokes = a user invocation like the toolbar click), captions and signs from the captured audio, page audio kept playing, Stop. Not tested in a real Meet or Zoom call. |
| Panel on its own (`/meet-panel`): share a meeting tab or the screen with audio | Built; pipeline tested end to end with a recorded voice as the audio source. Real tab or screen sharing in a call: not tested. Works for the Zoom desktop app via "share screen + system audio". |
| Voice into the meeting (VB-Audio Virtual Cable) | Built and **tested with VB-Cable 4.5** on the development PC: Test route "Verified" (level 0.178 on CABLE Output); signed words spoken into CABLE Input and transcribed back from CABLE Output (what a meeting would send): "hallo how are you", "i need water". Not yet heard by a second participant in a real call. |
| Google Meet add-on (side panel, main stage) | Code only. Needs registration (Google Cloud project, Marketplace SDK) and HTTPS hosting. Untested. |
| Zoom App (inside the Zoom client) | Code only. Needs a Zoom Marketplace app, HTTPS and the domain allow list. Untested. |

## Architecture

```
Meet / Zoom Web tab                                 MUDRA server (127.0.0.1:8010)
 ├ content.js: closed Shadow DOM box (draggable)      ├ WS /asr?room=R   16 kHz PCM -> sherpa-onnx streaming ASR
 │   └ panel.html (extension) -> iframe               │                  -> partial / final events to the sender
 │       └ http://localhost:3000/meet-panel?room=R    │                     and to every listener of room R
 │           avatar, caption, signs -> voice          ├ WS /captions?room=R   listen only
 │           listens to WS /captions?room=R  ◄────────┤ GET /captions/active  rooms with a live capture
 │           POST /tts -> WAV -> <audio>.setSinkId ───┤ POST /tts   text -> WAV (Windows voice, System.Speech)
 │                         (CABLE Input)              └ GET /speech/status
 └ (tab audio)
background.js (toolbar click) -> tabCapture.getMediaStreamId
offscreen.js: tab audio -> speakers (meeting stays audible) + 16 kHz PCM chunks -> WS /asr?room=R
```

Incoming pipeline in the panel (`app/frontend/src/features/meeting/`):

0. Server side, `speech.py`: a **speech gate** passes audio to the recogniser only from 0.3 s before speech (100 ms
   chunk RMS >= 0.01) until 1.2 s after it; long silence (a muted meeting, the cable between sentences) is never
   decoded. Without it the 2023-06-26 model wrote "and" over and over on silence (measured on the cable). The model
   and the voice process are loaded when the server starts (a first meeting otherwise waited ~3-8 s for the load).
1. `speechSources.js` - transcript events from one of: the extension's room (`listenToRoom`), a shared tab or screen
   streamed to `/asr` (`streamToMudraASR`, AudioWorklet 100 ms chunks, only new audio is sent), or Chrome's on-device
   recogniser on the shared track (`chromeOnDevice`: `SpeechRecognition` with `processLocally` and `start(track)`,
   probed working in Chrome 155; Chrome downloads its English pack once on a click).
2. `transcriptCommitter.js` - a word is committed once it has not changed for 450 ms and is not the last, still
   growing, word of the guess; the final text commits the rest. Each word once: repeated guesses, repeated finals and
   rewrites of committed words never duplicate.
3. `signPlanner.js` - committed words -> signs from GET /labels (words with a recorded clip): longest match first
   ("how are you" -> HOW_ARE_YOU), plurals and a few spoken forms ("thanks", "home" -> HOUSE). A word that may start
   a longer sign waits up to 700 ms. Words the recogniser splits in two are joined back when the whole word is a sign
   ("to morrow" -> TOMORROW, "to day" -> TODAY; "to" waits up to 1.5 s for its second half). Articles, auxiliaries and fillers are skipped (ISL leaves them out). A word without
   a sign is **never given an invented sign**: it stays in the caption and is listed as "No ISL sign in MUDRA's
   vocabulary yet"; optional fingerspelling for short words (`spellUnknown`).
4. `signScheduler.js` - one avatar sequence per phrase (`executeSignSequence(text, { append: true })`, new in
   `useAvatarRenderer.js`: ordered queue, phrase markers with first-frame callbacks, `cancelPhrase`, `clearQueue`).
   Backlog bound: a waiting phrase is dropped when it would be signed more than 6 s after it was heard, at most 4
   wait; the panel speeds the avatar from 1.25x up to 1.75x the recorded pace while phrases wait. Skipped words are
   shown. Stop clears everything (including phrases still being prepared).
5. Echo guard: a heard phrase made mostly of what MUDRA said in the last 8 s is not signed back.

Outgoing (`useOutgoingSpeech.js`, `voiceOutput.js`): the existing camera tracker (`handTracker.js`) and word model
stream (`signRecognizer.js`, WS `/stream`); only confident results (`status: ok`) are spoken (an unsure guess is shown,
not said); after a 1.4 s pause the words since the last utterance are spoken once (`buildMultilingualSentence`).
`POST /tts` -> WAV -> `<audio>` + `setSinkId(CABLE Input)`. Statuses: Preparing, Speaking, "Sent to the meeting
microphone" (only when the output is the cable), "Played on <device> only - the meeting did not hear it".
**Test route** plays a phrase into the chosen output and listens on "CABLE Output" for it: "Verified" only when the
sound really arrives where the meeting's microphone listens.

Reused, not duplicated: `useAvatarRenderer.js` / `landmarkRetarget.js` (the avatar and its recorded / letter signs),
`fetchVocabulary` (GET /labels), `handTracker.js`, `signRecognizer.js` (WS `/stream`), `buildMultilingualSentence`
(`islDictionary.js`, through `canonicalSentence`), the backend's CORS origins. What is spoken: a curated sentence for that
exact sequence of two or more signs (HELLO + WATER -> "Hello, I need water."), otherwise the signed words themselves
(WATER alone -> "Water.", not the Communicate page's "I need water."), a repeated word once. The extension's `pcm-worklet.js` repeats the 20-line worklet of
`speechSources.js` because extension pages cannot load the web app's modules.

## Setup (Windows, Chrome)

1. **MUDRA**: `start_all.bat` (server on 8010, web app on 3000). Speech recognition needs the model once:
   `pip install sherpa-onnx` (in `app/backend/.venv`, in `requirements.txt`) and the English streaming model
   (k2-fsa, Apache-2.0) unpacked to `app/backend/var/asr_models/` (git-ignored; keep `*.int8.onnx` and `tokens.txt`):
   default `sherpa-onnx-streaming-zipformer-en-2023-06-26` (310 MB download, 68 MB kept) from
   `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2`;
   the smaller `sherpa-onnx-streaming-zipformer-en-20M-2023-02-17` (128 MB download, 42 MB kept) also works
   (`MUDRA_ASR_MODEL_DIR`). Check: http://127.0.0.1:8010/speech/status.
2. **Extension**: Chrome -> `chrome://extensions` -> Developer mode -> Load unpacked -> `app/meeting-extension`.
   Pin the MUDRA button. Options: the web app and server addresses (defaults localhost:3000 / 127.0.0.1:8010).
3. **Voice into the meeting** (outgoing): install VB-Audio Virtual Cable (https://vb-audio.com/Cable/, donationware,
   needs admin and a restart). In the MUDRA panel: Voice route -> Speak into **CABLE Input** -> Test route
   ("Verified ..."). In Meet / Zoom: microphone = **CABLE Output**. (Chrome lists VB-Cable 4.5 as "CABLE Input",
   "CABLE In 16ch" and "CABLE Output"; Windows calls the playback end "Speakers (VB-Audio Virtual Cable)". Installing
   it did not change the Windows default speakers here.) Keep the meeting's speaker on your normal speakers
   or headphones (never CABLE Input), so the meeting's sound does not go back into the virtual microphone.
4. In a call (Meet in Chrome or Zoom Web at `app.zoom.us/wc/...`): click the MUDRA button (or press Alt+Shift+M;
   change it at `chrome://extensions/shortcuts`) -> the panel opens with
   "● Listening to this tab"; you still hear the meeting. Start signing for the camera. Stop / ✕ / the button again
   stops; leaving or closing the tab stops too.
5. Zoom desktop app (no extension there): open http://localhost:3000/meet-panel (or the extension button on any
   other tab opens it as a window) -> "Share a meeting tab or screen audio" -> share the screen with "Share system
   audio". Keep the window beside Zoom.

## What was tested (2026-10-21, development PC: Windows 11, Chrome 155, RTX 5060 laptop)

- Backend: `tests/test_speech.py` (6): rooms and origin checks, bad chunks refused, status without a model, TTS WAV,
  real streaming recognition of a Windows-voice sentence with words arriving before the sentence ends.
- Frontend: `src/features/meeting/__tests__/meeting.test.js` (12): stable-word commit, no duplicates, rewrites,
  longest-sign planning and waiting, no invented signs, aliases / plurals, ordered bounded queue, lag drop, Stop,
  TTS routing statuses, server errors, echo guard, cable detection.
- Incoming end to end (headless Chrome, a 14.5 s Windows-voice recording of 21 words as the audio input, real
  `/asr`, real avatar): captions and signs while the speaker continued; HOW ARE YOU, I NEED, WATER, HELP signed.
- Extension (Chrome 155, unpacked, on a local mock meeting page that plays the recording): overlay shown, page audio
  kept playing, the page's own button stayed clickable under the overlay, captions and avatar in the panel (HELLO
  signed 650 ms after it was first heard), Stop in the panel ended listening and cleared the badge (tab audio from a
  test script streaming to the same room).
- Extension with **real tab capture** (normal Chrome window, no automation flags for capture): Alt+Shift+M pressed with
  real Windows keystrokes -> badge ON, offscreen capture running, `/captions/active` listed the room, captions of the
  mock meeting's audio streamed into the panel and the avatar signed I NEED WATER, HELP ME, HELLO, HOW ARE YOU; the
  page's audio kept playing; the shortcut again stopped it (in 1 of 2 runs the test window did not get the keyboard
  focus back for the second press; the panel's Stop is the reliable way). Forcing capture without an invocation
  (`--allowlisted-extension-id`) does not work: Chrome refuses the stream ("Requested device not found").
- Outgoing (headless Chrome, fake camera): camera on/off (no live tracks after Stop), two confident signs spoken
  via `/tts`, an unsure guess not spoken, honest "speakers only" status without a cable.
- Outgoing through **VB-Cable 4.5** (installed 2026-10-21; normal Chrome window, blank test video instead of the
  webcam, signs injected into the recogniser hook): the panel chose CABLE Input by itself, Test route "Verified"
  (CABLE Output level 0.178), "Sent to the meeting microphone: Hello how are you." / "I need water."; a listener on
  CABLE Output (what Meet / Zoom record) transcribed by the local recogniser: "hallo how are you", "i need water".
  Sign -> voice on the cable: 4.1-4.6 s, of which 1.4 s is the deliberate end-of-sentence pause.
- Speech models compared on 10 Windows-voice recordings (5 sentences x 2 voices, 166 words; synthetic voices, not
  real human or Indian-accented speech): 20M model 16.9% word errors without the gate / 22.9% with it (it drops the
  first word after silence: "hello" -> "o", "please" -> "'s"); 2023-06-26 model 20.5% without the gate (an invented
  "and" at every start) / **15.1% with it** (4 of 10 recordings perfect; the rest mostly split words like "to day",
  "time table", and "Pune" -> "booney" / "pun"). First-partial delay median 0.73 s vs 0.80 s. The 2023-06-26 model
  with the gate is the default.
- **Not tested**: a real Meet / Zoom call (sign-in, real page permissions), a second participant hearing the voice,
  Zoom Web, the Meet add-on, the Zoom App.

## Measured latency

Incoming, from each word's start in the recording (word timings from the Windows voice), headless Chrome:

| Step | Measured |
|---|---|
| first partial transcript | 0.50-1.12 s (2023-06-26 model with the gate; 20M model: 0.45-0.81 s) |
| word committed (stable or final) | 1.0-1.9 s |
| first avatar frame of its phrase, avatar free | 1.96 s (HELLO, the first word) |
| same while earlier phrases are being signed | 3.3-5.1 s (dropped beyond 6 s) |
| ASR compute | 8-13x faster than real time (2023-06-26), 20-29x (20M); load 3-8 s at server start |

In that run (14.5 s, 21 words) every content word with a sign was planned (HELLO, HOW ARE YOU, THANK YOU, I NEED,
WATER, HELP, ME, GOOD MORNING, FRIEND, YOU, TOMORROW); 5 phrases were signed while the speaker talked and WATER, HELP,
ME were skipped to keep up.

Outgoing: sign committed -> 1.4 s pause (end of sentence) -> TTS 0.75-0.94 s for a new phrase (cached: 0 ms) -> audio.

## Known limitations

- **Signing is slower than speech.** MUDRA's recorded signs take 1.0-2.7 s each even after trimming the rest frames
  (measured: I 1.0 s, HELLO 2.1 s, NEED 2.0 s, HOW_ARE_YOU 2.7 s); speech is about 3 words per second. With continuous
  speech the avatar signs the first phrases, then drops phrases it would sign more than 6 s late (shown as
  "Skipped to keep up"). The caption always has every word.
- **Vocabulary**: 326 signs. Common words like "please", "my", "see", "for" have no sign and stay caption-only.
- **ASR accuracy**: 15.1% word errors on synthetic voices (see above); "joining" -> "adjoining", "Pune" -> "booney",
  "exam" -> "examine". English only; real meetings, other microphones and Indian-accented speech are not measured.
  Chrome's on-device recogniser (an option in the panel) is not compared yet.
- **Inside Meet / Zoom pages**: the panel needs camera and audio-output permission through two frames; if the
  meeting page's permission policy blocks this, use **Pop out** (the same panel in its own window).
- Other participants can only hear the voice when the meeting microphone is CABLE Output; MUDRA cannot change the
  meeting's microphone itself.

## Native surfaces: what is needed

**Google Meet add-on** (`clientSurfaces.js`, `?surface=meet-sidepanel` / `meet-mainstage`):
1. Host the web app on HTTPS (the add-on frames must be public HTTPS URLs).
2. Google Cloud project -> enable "Google Workspace Marketplace SDK" -> App configuration -> Meet add-on: side panel
   URL `https://<host>/meet-panel?surface=meet-sidepanel`, main stage URL `https://<host>/meet-panel?surface=meet-mainstage`.
3. Build with `VITE_MEET_CLOUD_PROJECT_NUMBER=<project number>`; add the host to `MUDRA_CORS_ORIGINS`.
4. The side panel joins the extension's capture on the same computer (GET /captions/active; Chrome asks for
   local-network access from the HTTPS page). "Show on main stage" starts an add-on activity so the avatar appears
   on the main stage; other participants' computers would need a reachable MUDRA server (not built: needs a hosted
   relay). The Meet Media API (direct meeting audio) is a developer preview closed to new sign-ups, so it is not used.
5. Publish privately to your Workspace domain or publicly (Google review).

**Zoom App** (`?surface=zoom`, Zoom Apps SDK from `https://appssdk.zoom.us/sdk.min.js`):
1. Zoom App Marketplace -> Develop -> General app -> Zoom App surface: home URL `https://<host>/meet-panel?surface=zoom`,
   domain allow list `<host>`, `appssdk.zoom.us`; capabilities getRunningContext, getMeetingContext, getUserContext.
2. Zoom requires HTTPS and OWASP security headers (Strict-Transport-Security, X-Content-Type-Options,
   Content-Security-Policy, Referrer-Policy) on the home URL; serve the built app accordingly (or a tunnel for development).
3. Meeting audio inside the Zoom client would need RTMS (Realtime Media Streams), which needs Zoom approval and a
   server-side receiver; not built. Until then the panel uses the local capture (extension or screen + system audio).

## Privacy

Meeting audio is decoded in memory on this computer and dropped; it is never written to disk, logged or sent to an
external service (the server logs only seconds of audio and phrase counts per stream). Captions live only in the open
panel. The camera frames stay in the browser (landmarks go to the local server, as on the Communicate page). Chrome's
on-device recogniser is used only with `processLocally`. Capture always shows "● Listening" (and the extension's ON
badge), and Stop is one click.
