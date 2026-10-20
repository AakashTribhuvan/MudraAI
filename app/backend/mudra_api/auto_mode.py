"""Auto mode: words (the merged models) and fingerspelled letters from one camera stream, no mode switching.

Words: the merged StreamingRecognizer (our word model + the teammate models' extra words). Letters: the letter
model on held hand shapes, with the Spell mode's rule (the same confident letter LETTER_STREAK checks in a row,
LETTER_POLL_S apart; a held letter is not repeated until the shape changes or the hands go down). Letters are
looked for only while no word sign is in progress, with a hand raised in front of the body, and not within
WORD_QUIET_S after a word. Moving from letter to letter can look like a short word: a word that ends while
spelling (a letter within SPELLING_S) is kept only at >= SPELLING_WORD_MIN confidence.
"""

from __future__ import annotations

import numpy as np
from mudra_ml.letters import LetterModel
from mudra_ml.streaming import Frame, StreamingRecognizer

LETTER_POLL_S = 0.25
LETTER_FRAMES = 4
LETTER_STREAK = 3
WORD_QUIET_S = 1.0
SPELLING_S = 2.0
SPELLING_WORD_MIN = 0.9
RAISED_BELOW_SHOULDERS = 1.2  # wrist at most this many shoulder widths below the shoulder line


def hands_raised(frame: Frame) -> bool:
    if not frame.pose_present or not frame.present.any():
        return False
    pose = frame.pose
    shoulder_y = (pose[11, 1] + pose[12, 1]) / 2
    width = abs(pose[12, 0] - pose[11, 0])
    if width < 1e-4:
        return False
    return any(frame.present[h] and (frame.hands[h, 0, 1] - shoulder_y) / width < RAISED_BELOW_SHOULDERS for h in range(2))


class AutoRecognizer:
    def __init__(self, words: StreamingRecognizer, letters: LetterModel | None, width: int, height: int):
        self.words, self.letters, self.width, self.height = words, letters, width, height
        self.recent: list[list[np.ndarray]] = []
        self.last_poll = -1e9
        self.streak: tuple[str | None, int] = (None, 0)
        self.last_letter: tuple[str, float] | None = None
        self.released = True
        self.last_word_t = -1e9

    def push(self, frame: Frame) -> list[dict]:
        events = []
        ev = self.words.push(frame)
        if ev:
            if ev["event"] == "sign_ended":
                r = ev["result"]
                spelling = self.last_letter is not None and frame.t - self.last_letter[1] < SPELLING_S
                if r.get("status") == "ok" and spelling and r.get("confidence", 0) < SPELLING_WORD_MIN:
                    ev = {**ev, "result": {**r, "status": "uncertain", "sign": None, "reason": "spelling"}}
                elif r.get("status") == "ok":
                    self.last_word_t = frame.t
            events.append(ev)
        letter = self._letter(frame)
        if letter:
            events.append(letter)
        return events

    def _letter(self, frame: Frame) -> dict | None:
        if self.letters is None:
            return None
        hands = [frame.hands[h] for h in range(2) if frame.present[h]]
        if not hands or not hands_raised(frame):
            self.recent.clear()
            self.streak, self.released = (None, 0), True  # hands down: the same letter may come again
            return None
        self.recent = (self.recent + [hands])[-LETTER_FRAMES:]
        if self.words.segmenter.state == "signing" or frame.t - self.last_word_t < WORD_QUIET_S:
            self.streak = (None, 0)
            return None
        if frame.t - self.last_poll < LETTER_POLL_S:
            return None
        self.last_poll = frame.t
        r = self.letters.predict_frames(self.recent, self.width, self.height)
        letter = r["letter"] if r["status"] == "ok" else None
        count = self.streak[1] + 1 if letter and letter == self.streak[0] else (1 if letter else 0)
        self.streak = (letter, count)
        if self.last_letter and letter != self.last_letter[0]:
            self.released = True
        if not letter or count < LETTER_STREAK or (not self.released and self.last_letter and letter == self.last_letter[0]):
            return None
        self.last_letter, self.released, self.streak = (letter, frame.t), False, (letter, 0)
        return {"event": "letter", "letter": letter, "sign": f"LETTER_{letter}", "confidence": r["confidence"],
                "top_k": r["top_k"], "model": "letters"}
