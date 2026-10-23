"""Confidence-hold gating and asynchronous text-to-speech for recognized signs."""

from __future__ import annotations

import math
import queue
import threading


class SignRecognitionHistory:
    """Keep recent stable sign labels and re-arm when a new segment starts."""

    def __init__(self, limit: int = 12):
        if limit < 1:
            raise ValueError("Sign history limit must be positive.")
        self.limit = limit
        self.events: list[dict[str, float | str]] = []
        self._segment_label: str | None = None

    def reset_segment(self) -> None:
        self._segment_label = None

    def observe(self, timestamp: float, label: str | None) -> bool:
        if label is None or label == self._segment_label:
            return False
        self._segment_label = label
        self.events.append({"label": label, "timestamp": timestamp})
        del self.events[:-self.limit]
        return True


class StablePredictionGate:
    """Emit an accepted label once after it remains unchanged for a hold interval."""

    def __init__(self, hold_s: float = 0.5):
        if not math.isfinite(hold_s) or hold_s <= 0:
            raise ValueError("Speech hold time must be a positive finite number.")
        self.hold_s = hold_s
        self.candidate: str | None = None
        self.candidate_since: float | None = None
        self.emitted = False

    def reset(self) -> None:
        self.candidate = None
        self.candidate_since = None
        self.emitted = False

    def update(self, timestamp: float, candidate: str | None) -> str | None:
        if candidate is None:
            self.reset()
            return None
        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = timestamp
            self.emitted = False
            return None
        if (
            not self.emitted
            and self.candidate_since is not None
            and timestamp - self.candidate_since >= self.hold_s
        ):
            self.emitted = True
            return candidate
        return None


class Pyttsx3Speaker:
    """Speak labels on a worker thread so synthesis does not stall camera tracking."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._failure: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="mudra-text-to-speech",
            daemon=True,
        )
        self._thread.start()

    def speak(self, label: str) -> None:
        self.raise_if_failed()
        self._queue.put(label.replace("_", " "))

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError(f"Text-to-speech failed: {self._failure}") from self._failure

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            while True:
                text = self._queue.get()
                if text is None:
                    break
                engine.say(text)
                engine.runAndWait()
            engine.stop()
        except Exception as exc:
            self._failure = exc
