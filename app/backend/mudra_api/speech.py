"""Local speech for meetings: text -> WAV (Windows voices) and streaming speech recognition (sherpa-onnx).

Nothing here leaves the computer and nothing is stored: TTS writes a temporary WAV that is read back and deleted,
ASR keeps only the decoder state of the open stream. Meeting audio is never written to disk.

TTS: a persistent PowerShell process uses System.Speech (the Windows SAPI voices, e.g. Microsoft David / Zira), so a
phrase costs one synthesis, not a PowerShell start (about 1 s). Other platforms: unavailable (HTTP 503), never faked.

ASR: a streaming zipformer transducer (sherpa-onnx, English, int8, 16 kHz). The client sends 16-bit mono PCM chunks;
partial hypotheses come back while the person speaks and a final one at each endpoint (a pause). Results go to the
sender and to every listener of the same caption room (the meeting panel, a Meet add-on or Zoom App surface).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("mudra_api.speech")

MAX_TTS_CHARS = 400
ASR_SAMPLE_RATE = 16000
# Speech gate: the recogniser only gets audio from shortly before speech starts until shortly after it stops. On long
# silence (a muted meeting, the virtual cable between sentences) the 2023-06-26 model otherwise invents words ("and",
# measured 2026-10-21), and decoding silence costs CPU. RMS of a 100 ms chunk, full scale = 1.0.
GATE_OPEN_RMS = 0.01  # about -40 dBFS; speech in a meeting tab is 0.03-0.2
GATE_PRE_S = 0.3  # audio kept from before the opening chunk (the start of the first word)
GATE_HANG_S = 1.2  # quiet audio still decoded after speech (the 0.8 s endpoint rule needs it)

_PS_WORKER = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
[Console]::Out.WriteLine('READY ' + (($s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }) -join '|'))
[Console]::Out.Flush()
while ($true) {
  $line = [Console]::In.ReadLine()
  if ($line -eq $null) { break }
  if ($line.Trim() -eq '') { continue }
  try {
    $p = $line.Trim().Split(' ')
    $text = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($p[0]))
    if ($p[3] -ne '-') { $s.SelectVoice([System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($p[3]))) }
    $s.Rate = [int]$p[2]
    $s.SetOutputToWaveFile([System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($p[1])), $fmt)
    $s.Speak($text)
    $s.SetOutputToNull()
    [Console]::Out.WriteLine('OK')
  } catch {
    $s.SetOutputToNull()
    [Console]::Out.WriteLine('ERR ' + $_.Exception.Message.Replace("`n", ' '))
  }
  [Console]::Out.Flush()
}
"""


class TTSUnavailable(RuntimeError):
    pass


class WindowsTTS:
    """System.Speech in one long-running PowerShell process; requests are serialised (one synthesizer)."""

    def __init__(self, cache_size: int = 64) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.voices: list[str] = []
        self.error: str | None = None
        self._cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._cache_size = cache_size

    @property
    def available(self) -> bool:
        return os.name == "nt"

    def _start(self) -> None:
        if not self.available:
            raise TTSUnavailable("Text to speech uses the Windows voices; this server is not running on Windows.")
        # the script goes in -EncodedCommand (UTF-16LE base64), so stdin carries only the requests
        script = base64.b64encode(_PS_WORKER.encode("utf-16-le")).decode("ascii")
        self._proc = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert self._proc.stdin and self._proc.stdout
        line = self._proc.stdout.readline().strip()
        if not line.startswith("READY"):
            self._proc.kill()
            self._proc = None
            raise TTSUnavailable(f"Windows speech did not start: {line or 'no output'}")
        self.voices = [v for v in line[6:].split("|") if v]
        log.info("Text to speech ready (Windows voices: %s)", ", ".join(self.voices) or "none")

    def synthesize(self, text: str, voice: str | None = None, rate: int = 0) -> bytes:
        text = " ".join(text.split())[:MAX_TTS_CHARS]
        if not text:
            raise ValueError("Nothing to say.")
        rate = max(-10, min(10, int(rate)))
        key = (text, voice or "", rate)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            if voice and voice not in self.voices:
                raise ValueError(f"Unknown voice {voice!r}; installed: {', '.join(self.voices)}")
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="mudra_tts_")
            os.close(fd)
            try:
                enc = lambda s: base64.b64encode(s.encode("utf-8")).decode("ascii")  # noqa: E731
                assert self._proc and self._proc.stdin and self._proc.stdout
                self._proc.stdin.write(f"{enc(text)} {enc(path)} {rate} {enc(voice) if voice else '-'}\n")  # text mode writes \r\n
                self._proc.stdin.flush()
                reply = self._proc.stdout.readline().strip()
                if reply != "OK":
                    raise TTSUnavailable(f"Windows speech failed: {reply or 'the speech process stopped'}")
                wav = Path(path).read_bytes()
            finally:
                Path(path).unlink(missing_ok=True)
            self._cache[key] = wav
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return wav

    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc = None


class StreamingASR:
    """One sherpa-onnx online recognizer (loaded on first use); one decoder stream per connection."""

    def __init__(self, model_dir: Path, num_threads: int = 2) -> None:
        self.model_dir = model_dir
        self.num_threads = num_threads
        self._rec = None
        self._lock = threading.Lock()
        self.error: str | None = None

    def _files(self) -> dict[str, Path]:
        d = self.model_dir
        pick = lambda part: next(iter(sorted(d.glob(f"{part}-*.int8.onnx"))), None) or next(iter(sorted(d.glob(f"{part}-*.onnx"))), None)  # noqa: E731
        return {"encoder": pick("encoder"), "decoder": pick("decoder"), "joiner": pick("joiner"), "tokens": d / "tokens.txt"}

    @property
    def available(self) -> bool:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            self.error = "sherpa-onnx is not installed (pip install sherpa-onnx)."
            return False
        files = self._files()
        if not all(f is not None and f.exists() for f in files.values()):
            self.error = f"No streaming ASR model in {self.model_dir} (see docs/MEETING_INTEGRATION.md)."
            return False
        return True

    @property
    def name(self) -> str:
        return self.model_dir.name

    def recognizer(self):
        with self._lock:
            if self._rec is None:
                if not self.available:
                    raise RuntimeError(self.error)
                import sherpa_onnx
                f = self._files()
                self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=str(f["tokens"]), encoder=str(f["encoder"]), decoder=str(f["decoder"]), joiner=str(f["joiner"]),
                    num_threads=self.num_threads, sample_rate=ASR_SAMPLE_RATE, feature_dim=80,
                    # beam search: greedy search lost whole phrases with the 20M model (e.g. "thank you. I need water")
                    decoding_method="modified_beam_search", max_active_paths=4, enable_endpoint_detection=True,
                    # a pause of 0.8 s after speech ends the phrase; 1.6 s of silence before any speech resets
                    rule1_min_trailing_silence=1.6, rule2_min_trailing_silence=0.8, rule3_min_utterance_length=15.0,
                )
                log.info("Streaming ASR ready (%s)", self.name)
            return self._rec

    def session(self) -> "ASRSession":
        return ASRSession(self.recognizer())


@dataclass
class ASRSession:
    """Feed PCM chunks; get (partial text, final text or None) after each chunk."""

    rec: object
    stream: object = None
    segment: int = 0
    last_partial: str = ""
    samples: int = 0
    started: float = field(default_factory=time.perf_counter)
    gate: bool = True
    open: bool = False
    quiet: int = 0  # samples of quiet audio since the gate opened or speech was last heard
    pre: list = field(default_factory=list)
    gated_samples: int = 0

    def __post_init__(self) -> None:
        self.stream = self.rec.create_stream()

    def _close_gate(self) -> list[dict]:
        """Speech ended and no endpoint came: finish the phrase so the next one starts fresh."""
        self.open = False
        text = _text(self.rec.get_result(self.stream))
        out = []
        if text:
            out.append({"event": "final", "segment": self.segment, "text": text, "audio_s": round(self.samples / ASR_SAMPLE_RATE, 3)})
            self.segment += 1
        self.rec.reset(self.stream)
        self.last_partial = ""
        return out

    def accept(self, pcm16: bytes) -> list[dict]:
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self.samples += len(x)
        if self.gate:
            loud = len(x) > 0 and float(np.sqrt(np.mean(x * x))) >= GATE_OPEN_RMS
            if not self.open:
                self.pre.append(x)
                while sum(len(c) for c in self.pre) > GATE_PRE_S * ASR_SAMPLE_RATE + len(x):
                    self.pre.pop(0)
                if not loud:
                    self.gated_samples += len(x)
                    return []
                self.open, self.quiet = True, 0
                x = np.concatenate(self.pre)
                self.pre = []
            elif loud:
                self.quiet = 0
            else:
                self.quiet += len(x)
                if self.quiet > GATE_HANG_S * ASR_SAMPLE_RATE:
                    return self._close_gate()
        self.stream.accept_waveform(ASR_SAMPLE_RATE, x)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        text = _text(self.rec.get_result(self.stream))
        out: list[dict] = []
        if text != self.last_partial and text:
            out.append({"event": "partial", "segment": self.segment, "text": text, "audio_s": round(self.samples / ASR_SAMPLE_RATE, 3)})
        self.last_partial = text
        if self.rec.is_endpoint(self.stream):
            if text:
                out.append({"event": "final", "segment": self.segment, "text": text, "audio_s": round(self.samples / ASR_SAMPLE_RATE, 3)})
                self.segment += 1
            self.rec.reset(self.stream)
            self.last_partial = ""
        return out

    def flush(self) -> list[dict]:
        """End of audio: decode what is left and finish the open phrase."""
        self.stream.accept_waveform(ASR_SAMPLE_RATE, np.zeros(int(0.5 * ASR_SAMPLE_RATE), dtype=np.float32))
        self.stream.input_finished()
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        text = _text(self.rec.get_result(self.stream))
        if not text:
            return []
        self.segment += 1
        return [{"event": "final", "segment": self.segment - 1, "text": text, "audio_s": round(self.samples / ASR_SAMPLE_RATE, 3)}]


def _text(result) -> str:
    t = result if isinstance(result, str) else getattr(result, "text", "")
    return " ".join(t.split()).lower()


class CaptionRooms:
    """Caption rooms: what one capture recognises is sent to every listener of the room (in memory only)."""

    def __init__(self) -> None:
        self._rooms: dict[str, set] = {}
        self._producers: dict[str, float] = {}  # rooms with a live capture (an /asr stream) -> start time

    def producing(self, room: str, on: bool) -> None:
        if on:
            self._producers[room] = time.time()
        else:
            self._producers.pop(room, None)

    def active(self) -> list[dict]:
        return [{"room": r, "since": t, "listeners": max(0, self.listeners(r) - 1)} for r, t in sorted(self._producers.items(), key=lambda x: -x[1])]

    def join(self, room: str, ws) -> None:
        self._rooms.setdefault(room, set()).add(ws)

    def leave(self, room: str, ws) -> None:
        members = self._rooms.get(room)
        if members is not None:
            members.discard(ws)
            if not members:
                del self._rooms[room]

    def listeners(self, room: str) -> int:
        return len(self._rooms.get(room, ()))

    async def publish(self, room: str, message: dict, skip=None) -> None:
        for ws in list(self._rooms.get(room, ())):
            if ws is skip:
                continue
            try:
                await ws.send_json(message)
            except Exception:  # a listener that went away
                self.leave(room, ws)


def valid_room(room: str | None) -> bool:
    """Room ids are unguessable tokens chosen by the capture side (16-64 url-safe characters)."""
    return bool(room) and 16 <= len(room) <= 64 and all(c.isalnum() or c in "-_" for c in room)


def origin_allowed(origin: str | None, allowed: list[str]) -> bool:
    """WebSockets bypass CORS, so the speech sockets check the Origin themselves: the web app's origins and Chrome
    extension pages (the meeting extension's offscreen capture page). No Origin = a local tool or test client."""
    if not origin:
        return True
    return origin in allowed or "*" in allowed or origin.startswith("chrome-extension://")


async def run_blocking(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)
