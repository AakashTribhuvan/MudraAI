"""Meeting speech: local TTS, streaming ASR, caption rooms, origin and room checks."""

import io
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from mudra_api.settings import Settings  # noqa: E402
from mudra_api.speech import CaptionRooms, StreamingASR, WindowsTTS, origin_allowed, valid_room  # noqa: E402
from mudra_api.speech_api import speech_router  # noqa: E402

ROOM = "test-room-0123456789"


class FakeSession:
    """Every chunk adds one word; 'stop' finishes the phrase (no model needed)."""

    def __init__(self):
        self.words, self.samples, self.segment = [], 0, 0

    def accept(self, pcm):
        self.samples += len(pcm) // 2
        self.words.append(f"w{len(self.words)}")
        return [{"event": "partial", "segment": 0, "text": " ".join(self.words)}]

    def flush(self):
        self.segment = 1
        return [{"event": "final", "segment": 0, "text": " ".join(self.words)}]


class FakeASR:
    available, name, error = True, "fake", None

    def session(self):
        return FakeSession()


def client(asr=None, tts=None):
    app = FastAPI()
    app.include_router(speech_router(Settings(), tts=tts, asr=asr or FakeASR(), rooms=CaptionRooms()))
    return TestClient(app)


def test_room_and_origin_rules():
    assert valid_room(ROOM) and not valid_room("short") and not valid_room("bad room id with spaces!")
    allowed = ["http://localhost:3000"]
    assert origin_allowed("http://localhost:3000", allowed)
    assert origin_allowed("chrome-extension://abcdefghijklmnop", allowed)
    assert origin_allowed(None, allowed)  # local tools and tests send no Origin
    assert not origin_allowed("https://evil.example", allowed)


def test_asr_events_reach_the_sender_and_room_listeners():
    c = client()
    with c.websocket_connect(f"/captions?room={ROOM}") as listener:
        assert listener.receive_json()["event"] == "listening"
        assert c.get("/captions/active").json()["rooms"] == []
        with c.websocket_connect(f"/asr?room={ROOM}") as cap:
            assert cap.receive_json()["event"] == "ready"
            assert listener.receive_json()["event"] == "ready"
            assert [r["room"] for r in c.get("/captions/active").json()["rooms"]] == [ROOM]
            cap.send_bytes(b"\x00\x00" * 1600)
            assert cap.receive_json()["text"] == "w0"
            assert listener.receive_json()["text"] == "w0"
            cap.send_text('{"type": "stop"}')
            final = cap.receive_json()
            assert final["event"] == "final" and final["text"] == "w0"
            assert listener.receive_json()["event"] == "final"


def test_bad_chunks_and_foreign_origins_are_refused():
    c = client()
    with c.websocket_connect("/asr") as cap:
        cap.receive_json()
        cap.send_bytes(b"\x00" * 3)  # odd length: not 16-bit PCM
        assert cap.receive_json()["event"] == "error"
    with pytest.raises(Exception):
        with c.websocket_connect("/asr", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
    with pytest.raises(Exception):
        with c.websocket_connect("/captions?room=short") as ws:
            ws.receive_json()


def test_status_reports_missing_asr_model(tmp_path):
    c = client(asr=StreamingASR(tmp_path / "none"))
    s = c.get("/speech/status").json()
    assert s["asr"]["available"] is False and s["asr"]["error"] and s["local_only"] is True


@pytest.mark.skipif(os.name != "nt", reason="the TTS uses the Windows voices")
def test_tts_returns_a_wav_and_rejects_empty_text():
    tts = WindowsTTS()
    try:
        c = client(tts=tts)
        r = c.post("/tts", json={"text": "thank you"})
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
        w = wave.open(io.BytesIO(r.content))
        assert w.getnchannels() == 1 and w.getnframes() / w.getframerate() > 0.3
        assert c.post("/tts", json={"text": ""}).status_code == 422
        assert c.post("/tts", json={"text": "hi", "voice": "No Such Voice"}).status_code == 422
    finally:
        tts.close()


ASR_DIR = Settings().asr_model_dir


@pytest.mark.skipif(os.name != "nt" or not StreamingASR(ASR_DIR).available, reason="needs the Windows voices and the ASR model")
def test_real_asr_streams_spoken_words_before_the_sentence_ends():
    tts = WindowsTTS()
    try:
        wav = tts.synthesize("hello, how are you? thank you. I need water, please help me")
    finally:
        tts.close()
    w = wave.open(io.BytesIO(wav))
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
    y = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 16000 / w.getframerate())), np.arange(len(x)), x).astype("<i2")
    y = np.concatenate([np.zeros(4800, dtype="<i2"), y])  # 0.3 s lead-in, as in a running meeting
    c = client(asr=StreamingASR(ASR_DIR))
    events = []
    with c.websocket_connect("/asr") as cap:
        assert cap.receive_json()["event"] == "ready"
        for i in range(0, len(y), 1600):
            cap.send_bytes(y[i:i + 1600].tobytes())
        cap.send_text('{"type": "stop"}')
        while True:
            ev = cap.receive_json()
            events.append(ev)
            if ev["event"] == "stopped":
                break
    partials = [e for e in events if e["event"] == "partial"]
    text = " ".join(e["text"] for e in events if e["event"] == "final")
    assert "how are you" in text and "need water" in text and "help me" in text
    # words arrive while the sentence is still being spoken
    first_water = next(e for e in partials if "water" in e["text"])
    assert first_water["audio_s"] < len(y) / 16000 - 0.5


class CountingRec:
    """Stands in for the sherpa-onnx recognizer: records how much audio reaches the decoder."""

    def __init__(self):
        self.fed = 0

    def create_stream(self):
        return self

    def accept_waveform(self, sr, x):
        self.fed += len(x)

    def is_ready(self, s):
        return False

    def get_result(self, s):
        return "word" if self.fed else ""

    def is_endpoint(self, s):
        return False

    def reset(self, s):
        self.fed = 0


def test_speech_gate_keeps_silence_from_the_recogniser():
    from mudra_api.speech import ASRSession
    rec = CountingRec()
    s = ASRSession(rec)
    silence = (np.zeros(1600, dtype="<i2")).tobytes()
    for _ in range(50):  # 5 s of a muted meeting: nothing decoded, nothing invented
        assert s.accept(silence) == []
    assert rec.fed == 0
    tone = (np.sin(np.arange(1600) / 5) * 8000).astype("<i2").tobytes()
    events = s.accept(tone)  # speech: the gate opens with 0.3 s of the audio before it
    assert rec.fed >= 4800 + 1600 and events and events[0]["event"] == "partial"
    for _ in range(13):  # 1.3 s quiet after speech: the phrase is finished and the gate closes
        events = s.accept(silence)
    assert events and events[-1]["event"] == "final" and not s.open
