# Introduction signing and recognition guide

Permanent demo: <https://mudra-camera-gateway.aakashtribhuvan2006.workers.dev/model-test>.
The Worker redirects to the active laptop Quick Tunnel. `start_all.bat` starts/reuses the local API
and frontend, waits for both to answer, and opens the tunnel/Worker window. Keep that window open.
The app is only online while the computer, network, backend, frontend and tunnel are running.

## How to present a sign to the recognizer

1. Open **Communication**, choose **Words**, and start the camera stream.
2. Keep your face, shoulders, torso and signing hands visible, with steady light and a plain background.
   The recognizer uses hands in relation to the upper-body pose, not just a cropped hand.
3. Sign **one complete word**, including its normal movement. Do not freeze the hand midway through a
   dynamic sign or try to hold every word as a still pose.
4. Pause briefly or lower your hands between words so the segmenter can see the sign end. Wait for the
   word to appear before starting the next one.
5. In **Spell** mode, form one alphabet letter at a time and hold it steady for about a second. Spell
   abbreviations and words the word model does not contain this way.
6. In **Phrase** mode, sign a complete trained phrase and choose the closest returned candidate; phrases
   are not automatically committed.

If the app reports that recognition is offline, check `http://127.0.0.1:8000/health` locally. A healthy
API reports `model_loaded: true`; do not start a second `main.py` while it is already healthy (Windows
socket error 10048 means port 8000 is occupied). Then check `http://127.0.0.1:3000/health` and the
tunnel window. `start_all.bat` now checks/reuses local services and starts the tunnel automatically.

## Your introduction: available labels

The deployed **Words** model contains these exact labels from the requested introduction:

| Spoken idea | Word-model label | Availability |
|---|---|---|
| Hello | `HELLO` | In the deployed 314-word model |
| Good morning | `GOOD_MORNING` | In the deployed 314-word model |
| We | `WE` | In the deployed 314-word model |
| Students | `STUDENT` | Singular model label; no `STUDENTS` label |
| From | — | No exact `FROM` label in the deployed word model |
| K.S.E. | — | No school/acronym label; spell `K`, `S`, `E` in Spell mode |
| This / is | — | No exact `THIS` or `IS` label in the deployed word model |
| SignLine | `SIGN` only | No `LINE` or `SIGNLINE` word label; spell `SIGNLINE` in Spell mode |

Thus the word model can attempt **HELLO**, **GOOD_MORNING**, **WE**, and **STUDENT** as separate
signs. Spell the unsupported parts (such as K.S.E., FROM, THIS IS and LINE) letter by letter. The
phrase model has related whole-phrase classes such as `WHICH_COLLEGESCHOOL_ARE_YOU_FROM` and
`NICE_TO_MEET_YOU`, but not the complete introduction as one trained phrase.

Communication now defaults to the 17-word **Common words** video model. It uses mixed-source training
and scored 45/51 (88.2%, macro-F1 0.877) on its signer-held-out clip test; live-webcam accuracy is
unknown. The 14-word **ISL500 video pilot** remains selectable as an alternative when its local
artifact is installed. Its labels are:
`FOOD`, `HELP`, `I`, `MYSELF`, `NICE`, `NO`, `THANKYOU`, `WATER`, `WHAT`, `YOU`, `HELLO`, `WE`,
`STUDENT`, and `THIS`. It scored 80.5% accuracy / 0.79 macro-F1 on 41 clips from three signers
kept out of training and model selection. That small clip test is not a live-webcam accuracy
estimate; use it experimentally and check each result. `FROM`, `ARE`, `INVENTOR`, `PUNE`, and
`SIGNLINE` are still not pilot labels. The newer 26-class experiment scored 48/77 (62.3%) on its
held-out clip set and is hidden from the live selector until it improves.

The separate static-image prototype includes `FROM` among its labels, but it is not connected to the
live recognizer and should not be selected as a working camera model.

## How the models were trained

The app's deployed v8 word model was already part of the Mudra repository; it was **not retrained**
as part of creating the separate `MudraTraining` clone. It was trained on isolated-sign sequences from
the 263-word INCLUDE source plus team word recordings (61 additional dataset labels in the training
card). MediaPipe extracts hand and upper-body landmarks; a per-frame landmark network and BiLSTM model
the time sequence, then attention pooling and a classifier choose among 314 words plus a background/no
sign class. In the app, the browser sends landmarks, not camera images. The backend waits for a
complete sign segment before classifying it.

The static model uses still images from `Frames_Word_Level`, MediaPipe pose/hand features normalized
to torso and palm geometry, and nearest-medoid matching per label/hand-count profile. Communication
uses the model artifact and continuous held-pose recognition flow from `MudraPhoneWebcam`; a label
must remain stable for 450 ms before it is added. The source model's exploratory held-out score was
13/73 (17.8%). This is not a live-camera accuracy estimate, and only ten labels have finite rejection
thresholds. The alphabet model remains separate and unchanged.

## What the accuracy figures do and do not say

- **Words, v8:** 98.06% top-1 and 99.93% top-5 on 1,495 held-out clips in the model card; per-recording
  top-1 is about 97.5%. The held-out clips share signers/recording sessions with training. This is not
  evidence of 98% accuracy for a new signer or a real webcam session.
- **Unseen people:** the best documented proxy is about 67% clip accuracy on nine held-out INCLUDE
  recording sessions using an earlier v3 recipe. Sessions are not guaranteed to be different people.
- **Alphabet:** 70.9% top-1 on 1,070 held-out photos grouped by colour/source; not tested live.
- **Phrases:** 58.3% top-1 / 81.3% top-5 on 96 held-out videos from the same signer pool; the app asks
  the user to choose a candidate.
- **Actual webcam:** the current model has not passed a live test with a person signing. Camera frame
  rate, framing, lighting and signer variation can lower results. Treat recognition as a demo and
  check/correct every word.
