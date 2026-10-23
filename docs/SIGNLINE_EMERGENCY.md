# SignLine Emergency Communication Assistant

Status: **prototype, demo mode** (2026-10-16). It contacts no one. Reports reach only the SignLine
demo responder desk inside this app. The app tells users: in a real emergency in India, call 112.

The feature was added without changing the existing app: a floating circular **SOS** button sits at
the bottom-right of every page (above the mobile bottom navigation). The backend is a separate
package (`app/backend/signline_emergency`) mounted into the existing FastAPI app with one line.

## What a user can do

1. Open **SOS** → *Start emergency communication* (no sign-in needed).
2. Communicate in one of three ways, and switch between them at any time:
   - **Buttons**: medical, police or safety, fire, accident, rescue, general help, something else, then a detail
     (e.g. *Medical → Chest pain*). Selections are stored as said by the user, never as a prediction.
   - **Signs**: each sign is shown with its recognition confidence. Unclear signs (below 0.35) are
     kept and marked *not recognised*; unsure ones (below 0.6) are marked *unsure*. Tap a sign to
     correct or remove it; *Undo* and *Clear* work step by step. The raw recognition is never edited.
   - **My gestures**: a personal gesture (set up on *My emergency gestures*) maps to an intent and a
     prepared message, e.g. "My Medical SOS" → *I need medical assistance. Please communicate with me
     using text.* Optional confirmation before use and a cooldown.
3. See **what SignLine understood**: a category ("Possible: Medical help"), a summary that repeats
   what was reported ("You reported chest pain."), which signs are uncertain, the rule that produced
   each interpretation, alternatives and what is still unknown.
4. Answer at most two **questions** at a time, only those that change the report (e.g. *Is anyone
   unconscious?* only when the help is for another person). Every question can be skipped; none is
   asked twice; questions that stop applying are withdrawn.
5. Optionally add a **location**: browser location (the browser asks permission), a typed place, or
   in demo mode a clearly labelled simulated placeholder with no coordinates.
6. **Review** the report, correct the category or the message, and **confirm**. Nothing is shared
   before this.
7. **Share** it. The status shows *Shared with the configured destination*, then *Acknowledged by a
   responder* and *Resolved*, or *Delivery failed (not received)* with *Try again*.

Responders sign in at `/responder/login` and see only delivered reports: facts grouped by where they
came from, the sign sequence with confidences, answers, location if shared, communication needs, the
status timeline and delivery attempts. They can acknowledge, add notes (responders only) and resolve.

## Personal gestures with a video, emergency numbers and the dial screen (2026-10-21)

- **Add gesture** is a three-step wizard: details (name, description, category, message, emergency service),
  an optional camera recording, then consent and save. The camera is requested only when the user presses
  *Record gesture*; there is a live preview, a red *Recording* indicator, Start / Stop / Preview / Retake and an
  automatic stop after 20 s (`features/emergency/recorder.js`, browser APIs injected so it is unit-tested).
  Denied permission, no camera, a busy camera or a browser without `MediaRecorder` all offer *Save without a
  recording*. The video is uploaded only with the consent box ticked; without it the gesture is saved alone.
- **Recording does not train recognition.** Gestures are still used by tapping them; camera recognition of
  personal gestures is future work.
- Every saved gesture is a card that opens a **detail page**: all details, the video, *Edit*, *Retake video*,
  *Delete video*, *Delete* (gesture and video), *Test gesture* (sends nothing) and *Request emergency assistance*.
- **Request emergency assistance** opens a review screen: the emergency service and number (changeable;
  default 112), an optional location (typed, or the browser's location after its permission prompt), the
  message line by line with provenance (`USER_REPORTED` / `SYSTEM_GENERATED`), and the video. *Call emergency
  number* asks for confirmation, then opens the phone's dialer with a `tel:` link: the user presses call, the
  app states it cannot tell whether the call connected, and a call does not carry the message or video.
  *Share emergency report* is separate: the phone's share sheet, with the video only if the user ticks it and
  the device can attach files (otherwise text only, or copy to the clipboard). The share sheet can only reach
  *Sharing started*: SignLine never claims delivery for it.
- **Send to the SignLine responder desk (demo)** on the same screen goes through the normal flow:
  `POST /gestures/{id}/request-assistance` creates a session from the gesture, adds the location, reviews and
  confirms it (status *Ready to share*; nothing shared). The session page then shares it with the existing
  share button and shows *Delivered* (with the simulated receipt), *Acknowledged*, *Resolved* or *Not delivered*
  with *Try again*. The gesture video goes with it only if the user ticks the consent box: the incident gets its
  own encrypted copy (`video_sharing_status` ATTACHED -> DELIVERED / FAILED), so deleting the gesture does not
  remove what was shared. Responders see the gesture report, the chosen number and, for a delivered report
  with a consented video, a *Show video* button (`GET /responder/incidents/{id}/video`; every view is audited).
- **Emergency numbers** live in `config/emergency_contacts.json` (regions; India national entries: 112, 100,
  101, 102, 108, 103, 1073, 1033, 181, 1098, 1930, 1070, 139). All are `verified: false` until checked against
  a current authoritative source (112.gov.in, india.gov.in/directory/helpline); state entries can be added.
- **Video privacy:** this is the one place SignLine stores video. Only after recording consent (time stored),
  under a random key in `app/backend/var/emergency_media/` (git-ignored), served only to the owning device
  (`X-Device-Key`), checked on upload (WebM / MP4 by content, not only the declared type; at most 25 MB and
  30 s), replaced on retake, overwritten and deleted with its gesture, never logged, analysed or used for
  training. **Encrypted at rest** with AES-256-GCM (`cryptography`; the storage key is authenticated, so a file
  cannot be renamed into another record; a damaged or swapped file is refused with 503). The key is
  `SIGNLINE_EM_MEDIA_KEY` (base64, 32 bytes) or is generated once into `app/backend/var/emergency_media.key`
  (git-ignored; losing it makes stored videos unreadable).
- **Retention** (`retention.py`, at server start and every `SIGNLINE_EM_RETENTION_INTERVAL_H` hours, default 24):
  ended sessions older than `SIGNLINE_EM_RETENTION_DAYS` with their incident and its video copy; gesture videos
  `SIGNLINE_EM_GESTURE_VIDEO_RETENTION_DAYS` (default 365) after recording (the gesture stays); video files no
  record refers to, after an hour. The log keeps counts only.

## Honesty rules built into the feature

- **Recognition is simulated in demo mode.** The deployed SignLine model (v8, 314 words) does not know
  most emergency signs (HELP, CHEST, PAIN, BREATHING, FIRE, SMOKE, ACCIDENT, UNSAFE, FOLLOWING …; it
  does know e.g. INJURY, SICK, DOCTOR, HOSPITAL, POLICE, FEVER). Signs come from a deterministic mock
  provider with fixed confidence values, or from the sign list on the page. This is labelled
  *Simulated recognition* in the UI.
- **No diagnosis, no urgency guess.** Rules repeat what was reported. Priority is set only by explicit
  answers (immediate danger, someone unconscious or trapped = *High*); otherwise it is not assessed.
- **Provenance on every fact:** `USER_REPORTED` (signed, tapped, typed, answered), `AI_INFERRED` (rule
  interpretation, uncertainty), `USER_CONFIRMED` (category and summary confirmed on review),
  `SYSTEM_GENERATED` (ids, times, statuses, simulated location).
- **Distinct states:** report prepared, shared, acknowledged and resolved are different statuses; a
  delivery is *delivered* only when the destination received it. The mock dispatcher delivers to the
  in-app responder desk and returns receipts marked simulated (`SIM-…`).
- **Privacy:** no video or images are stored except a personal gesture's demonstration video, and only with
  the user's recording consent (see above); location only when the user shares it; analytics keep
  only event types, times, session ids and non-sensitive fields (category, counts, source), never
  signs, messages, summaries or locations; routine logs carry no report content.

## Architecture

```
SOS button (every page) → /emergency pages (React) → /api/v1/emergency (FastAPI router)
                                                        │
 RecognitionEvent ← ISLRecognitionProvider (mock | real adapter)
        │
 SignSequenceManager (raw events + normalized sequence with revisions, undo)
        │
 context_engine.evaluate (JSON rules) → question_manager.refresh (JSON questions)
        │
 workflow: review → confirm (profile_builder: incident + facts with provenance) → share (dispatch provider)
        │
 state_machine (only listed transitions; history + audit on every change)
```

| Path (under `app/backend/signline_emergency/`) | Role |
|---|---|
| `config.py` | settings from `SIGNLINE_EM_*` environment variables |
| `db.py`, `models.py`, `migrations/` | SQLAlchemy 2 models (17 tables), Alembic migration `0001` (runs at startup) |
| `config/emergency_rules.json` | categories, rules (id, version, priority, all / any / none signals, summary, missing, questions) |
| `config/emergency_questions.json` | questions, answer types, conditions (`no_location`, `answer:K=V`, `missing_signal:S`) |
| `config/emergency_buttons.json` | button options and the signals each selection adds |
| `config/demo_scenarios.json` | mock vocabulary and scenarios A–I |
| `services/recognition/` | `RecognitionEvent` contract, mock provider, real-model adapter (`translate_stream_result`) |
| `services/sequence_manager.py` | raw + normalized sequence, correct / remove / undo / reset, repeats flagged |
| `services/context_engine.py` | deterministic rule evaluation, ranking, summaries, uncertainty, no-emergency message |
| `services/question_manager.py` | which questions to ask, answer validation, skip, superseding |
| `services/profile_builder.py` | incident profile and facts with provenance |
| `services/workflow.py` | ties it together; every status change goes through `set_status` |
| `services/state_machine.py` | states, allowed transitions, labels |
| `services/dispatch.py`, `services/location.py` | mock dispatch provider; location validation |
| `services/auth.py` | scrypt passwords, HMAC-signed responder tokens, session tokens, rate limiter |
| `services/audit.py` | audit trail and privacy-first analytics |
| `api/router.py` | REST endpoints, access checks |
| `seed.py`, `retention.py` | demo responder account; purge of old ended sessions |

Frontend: `app/frontend/src/features/emergency/` (API client, SOS button, shared components) and
`app/frontend/src/pages/emergency/` (start, session, gestures, about, responder pages). Routes:
`/emergency`, `/emergency/session/:id`, `/emergency/gestures` (also `/settings/emergency-gestures`),
`/emergency/about`, `/responder/login`, `/responder`, `/responder/incidents/:id`. A web app manifest
(`public/manifest.webmanifest`, with an *Emergency help* shortcut) lets phones install SignLine to the
home screen.

### States

`SESSION_CREATED → COMMUNICATING → POSSIBLE_EMERGENCY_DETECTED → AWAITING_USER_CONFIRMATION →
USER_CONFIRMED → READY_TO_SHARE → SENT_TO_CONFIGURED_DESTINATION → ACKNOWLEDGED → RESOLVED`, plus
`DELIVERY_FAILED` (retry goes back to `READY_TO_SHARE`) and `CANCELLED` (not possible after sharing).
Responders can only set `ACKNOWLEDGED` and `RESOLVED`. The report is locked after confirmation.

### Access

- Emergency users are anonymous. The browser creates a random device key (`X-Device-Key`, kept in
  localStorage) that owns its gestures and communication needs. Each session returns an access token
  (`X-Session-Token`, kept in sessionStorage for that tab, stored on the server only as a SHA-256 hash).
  A wrong token and an unknown session give the same 404.
- Responders: `POST /responder/login` → bearer token (HMAC-SHA256, expiry `SIGNLINE_EM_TOKEN_TTL_S`).
  The role is read from the database on every request.
- Rate limits (in memory, per server process): login 10/min per IP, session creation per device
  (`SIGNLINE_EM_RATE_LIMIT_PER_MIN`, default 30) and 60/min per IP, sharing 10/min per session.

## API (`/api/v1/emergency`)

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /config` | – | demo mode, providers, notice, categories, intents, buttons, communication needs, demo vocabulary and scenarios |
| `POST /sessions` `{mode, scenario_id?}` | device | new session → `{session, access_token}` |
| `GET /sessions/{id}` | session | full view: status, raw events, sequence, context, questions, location, incident, history |
| `POST /sessions/{id}/mock-sign` `{sign, confidence}` | session | demo only: one simulated sign |
| `POST /sessions/{id}/scenario` `{action: next\|all\|pause\|resume\|restart}` | session | scenario playback |
| `POST /sessions/{id}/selections` `{need, detail?}` | session | button selection |
| `POST /sessions/{id}/gestures/{gesture_id}/trigger` | session | use a personal gesture |
| `POST /sessions/{id}/provider-events` | session | recognition-provider input (real adapter: a `/stream` `sign_ended` message) |
| `POST /sessions/{id}/sequence/correct\|remove\|undo\|reset` | session | edit the normalized sequence |
| `POST /sessions/{id}/questions/{qid}/answer\|skip` | session | questions |
| `PUT\|DELETE /sessions/{id}/location` | session | share or remove a location |
| `POST /sessions/{id}/review\|correction\|confirm\|share\|cancel` | session | review and sharing flow |
| `GET\|POST /gestures`, `PATCH\|DELETE /gestures/{id}`, `POST /gestures/{id}/test` | device | personal gestures (test sends nothing; delete removes the video too) |
| `PUT\|GET\|DELETE /gestures/{id}/video` | device | demonstration video: multipart `file`, `duration_s`, `recording_consent=true` (422 without consent or for a bad file, 503 if storage fails) |
| `POST /gestures/{id}/report` `{contact_id?, location_text?}` | device | the message for the dial / share screen, line by line with provenance, `dial_link`; sends nothing |
| `GET /contacts?region=` | – | emergency numbers (`config/emergency_contacts.json`), default 112 |
| `POST /gestures/{id}/request-assistance` `{contact_id?, location?, include_video, sharing_consent}` | device | session + confirmed incident from the gesture, status READY_TO_SHARE → `{session, access_token, report}`; video only with `sharing_consent` |
| `GET /responder/incidents/{id}/video` | responder | the consented video of a delivered report (audited) |
| `GET\|PUT /preferences` | device | communication needs |
| `POST /responder/login`, `GET /responder/me` | –, responder | responder sign-in |
| `GET /responder/incidents?status=`, `GET /responder/incidents/{id}` | responder | delivered reports only |
| `POST /responder/incidents/{id}/acknowledge\|resolve\|notes` | responder | responder actions |
| `GET /responder/analytics?days=` | responder | aggregated counts only |

Errors: 401 (no or invalid device key / responder token), 404 (session or item not found or not
yours), 409 (not allowed in the current status, e.g. editing after confirmation), 422 (invalid input),
429 (rate limit or gesture cooldown).

## Configuration

`app/backend/.env.example` lists every variable: `SIGNLINE_EM_DATABASE_URL` (default SQLite at
`app/backend/var/signline_emergency.db`), `SIGNLINE_EM_SECRET_KEY` (unset = random per start, so
responder logins end on restart), `SIGNLINE_EM_TOKEN_TTL_S`, `SIGNLINE_EM_DEMO_MODE`,
`SIGNLINE_EM_RECOGNITION_PROVIDER` (`mock` | `real`), `SIGNLINE_EM_DISPATCH_PROVIDER` (only `mock`),
`SIGNLINE_EM_DISPATCH_DESTINATION`, `SIGNLINE_EM_DEMO_RESPONDER_PASSWORD`, `SIGNLINE_EM_RETENTION_DAYS`,
`SIGNLINE_EM_RATE_LIMIT_PER_MIN`, `SIGNLINE_EM_RUN_MIGRATIONS`.

Demo responder: username `demo.responder`. Its password comes from
`SIGNLINE_EM_DEMO_RESPONDER_PASSWORD`, or is generated on first start into
`app/backend/var/demo_responder_credentials.txt`. `app/backend/var/` is git-ignored. To reset
everything, stop the server and delete `app/backend/var/`.

Retention: `python -m signline_emergency.retention [--days N] [--dry-run]` (run from `app/backend`)
deletes ended sessions (with their events, answers, incident and facts) and analytics older than the
retention period.

## Connecting the real recognition model

`services/recognition/real_signline_adapter.py` maps a `/stream` `sign_ended` message onto a
`RecognitionEvent` (tested). To use it: (1) extend the model with the emergency vocabulary
(`demo_scenarios.json` → `vocabulary`); (2) in the browser, forward each `sign_ended` message of the
active emergency session to `POST /sessions/{id}/provider-events`; (3) set
`SIGNLINE_EM_RECOGNITION_PROVIDER=real`. The rules, questions, profiles and responder workflow only see
`RecognitionEvent`s, so nothing else changes (a test swaps the provider to show this).

## Tests

`app/backend/tests/emergency/` (54 tests; 9 in `test_gesture_assistance.py`: prepared-not-shared, delivered with
receipt, acknowledged, video to responders only with sharing consent and only after delivery, the incident's own
copy, delivery failure, encryption at rest, tampering and wrong key, key from env or file, retention of incident /
gesture / orphan videos and dry run. Before that 45 tests, a fresh SQLite database and media folder each; 16 in
`test_gesture_video.py`: gestures with and without video, retake, delete with video, consent, file type /
content / size / duration checks, owner-only access, storage failure, contacts, report content and provenance,
destination change, dial-link sanitising, no video or report details in analytics). Frontend:
`features/emergency/__tests__/recorder.test.js` (permission granted / denied / no camera / unsupported, start /
stop / preview / retake / too short, camera released, dial links, share with and without video, statuses never
claim delivery). Earlier tests: scenarios A–I end to end;
state-machine rules; confirmation lock; cancellation; unknown, low-confidence and repeated signs;
remove / undo / reset; user correction wins; buttons as user-reported; answer validation and skip;
location question withdrawn after sharing; location validation; gestures (CRUD, test, trigger,
cooldown, other devices); preferences in the incident; session-token access; responder auth,
visibility of delivered reports only, acknowledge / note / resolve; password hashing and token
tampering; login rate limit; analytics contain no report content; provider swap; real-adapter
translation; every matched rule recorded.

## Known limits

- Camera recognition is not connected (see above); personal gestures are tapped, not detected by camera.
- Emergency numbers are not verified (and no state-specific entries yet). Overwriting a deleted video is best
  effort on SSDs; the files are encrypted, so leftover blocks are unreadable without the key.
- The only dispatch destination is the in-app demo responder desk; no SMS, call or emergency-service
  integration exists or is claimed.
- Rate limits and the random default secret key are per server process; a multi-process deployment
  needs a shared store and a fixed `SIGNLINE_EM_SECRET_KEY`.
- SQLite is for the prototype; set `SIGNLINE_EM_DATABASE_URL` for a server database.
- Status labels are written for the user ("Waiting for your confirmation") and are reused on the
  responder page.
- English only.
