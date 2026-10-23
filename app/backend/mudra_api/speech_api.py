"""Meeting speech endpoints: POST /tts (text -> WAV), WS /asr (PCM -> captions), WS /captions (listen to a room).

    POST /tts            {"text": "...", "voice": optional, "rate": -10..10} -> audio/wav (local Windows voice)
    GET  /speech/status  what is available (voices, ASR model) and why not
    WS   /asr?room=R     binary frames: 16 kHz mono 16-bit PCM; text frame {"type":"stop"} ends the stream.
                         Replies {"event":"ready"|"partial"|"final"|"error", ...}; the same events go to room R.
    WS   /captions?room=R  listen only: the events of room R (a second surface, e.g. the Meet add-on side panel)
    GET  /captions/active  rooms with a live capture on this computer (lets an add-on panel join the extension's room)

Audio is decoded and dropped; nothing is stored or logged except counts.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from mudra_api.settings import Settings
from mudra_api.speech import (MAX_TTS_CHARS, CaptionRooms, StreamingASR, TTSUnavailable, WindowsTTS, origin_allowed,
                              run_blocking, valid_room)

log = logging.getLogger("mudra_api.speech")


def _warm_up(asr: StreamingASR, tts: WindowsTTS) -> None:
    try:
        if asr.available:
            asr.recognizer()
    except Exception as exc:  # reported by /speech/status and the first /asr connection instead
        log.warning("ASR warm-up failed: %s", exc)
    try:
        if tts.available:
            tts.synthesize("ready")
    except Exception as exc:
        log.warning("TTS warm-up failed: %s", exc)

MAX_CHUNK_BYTES = 64_000  # 2 s of 16 kHz PCM; clients send ~100 ms


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TTS_CHARS)
    voice: str | None = Field(default=None, max_length=80)
    rate: int = Field(default=0, ge=-10, le=10)


def speech_router(settings: Settings, tts: WindowsTTS | None = None, asr: StreamingASR | None = None,
                  rooms: CaptionRooms | None = None, warm_up: bool = False) -> APIRouter:
    router = APIRouter()
    tts = tts or WindowsTTS()
    asr = asr or StreamingASR(settings.asr_model_dir)
    rooms = rooms or CaptionRooms()
    if warm_up:  # load the ASR model and start the voice process now, not when the first meeting starts (~3 s)
        threading.Thread(target=_warm_up, args=(asr, tts), name="speech-warm-up", daemon=True).start()

    @router.get("/speech/status")
    def status() -> dict:
        asr_ok = asr.available
        return {
            "tts": {"available": tts.available, "engine": "windows-sapi" if tts.available else None,
                    "voices": tts.voices, "error": tts.error},
            "asr": {"available": asr_ok, "engine": "sherpa-onnx" if asr_ok else None, "model": asr.name if asr_ok else None,
                    "sample_rate": 16000, "error": None if asr_ok else asr.error},
            "local_only": True,
        }

    @router.get("/captions/active")
    def active_rooms() -> dict:
        return {"rooms": rooms.active()}

    @router.post("/tts")
    async def speak(req: TTSRequest) -> Response:
        try:
            t = time.perf_counter()
            wav = await run_blocking(tts.synthesize, req.text, req.voice, req.rate)
            ms = round((time.perf_counter() - t) * 1000)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except TTSUnavailable as exc:
            tts.error = str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(wav, media_type="audio/wav", headers={"Cache-Control": "no-store", "X-TTS-Ms": str(ms)})

    async def _accept(ws: WebSocket, room: str | None) -> bool:
        if not origin_allowed(ws.headers.get("origin"), settings.cors_origins):
            await ws.close(code=1008)
            return False
        if room is not None and not valid_room(room):
            await ws.close(code=1008)
            return False
        await ws.accept()
        return True

    @router.websocket("/asr")
    async def recognise(ws: WebSocket) -> None:
        room = ws.query_params.get("room")
        if not await _accept(ws, room):
            return
        if not asr.available:
            await ws.send_json({"event": "error", "detail": asr.error})
            await ws.close()
            return
        session = await run_blocking(asr.session)
        if room:
            rooms.join(room, ws)
            rooms.producing(room, True)

        async def emit(events: list[dict]) -> None:
            for ev in events:
                ev["t_server"] = time.time()
                await ws.send_json(ev)
                if room:
                    await rooms.publish(room, ev, skip=ws)

        await emit([{"event": "ready", "engine": "sherpa-onnx", "model": asr.name, "sample_rate": 16000}])
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    chunk = msg["bytes"]
                    if len(chunk) > MAX_CHUNK_BYTES or len(chunk) % 2:
                        await emit([{"event": "error", "detail": "PCM chunks must be 16-bit and at most 2 s"}])
                        continue
                    await emit(await run_blocking(session.accept, chunk))
                elif msg.get("text"):
                    try:
                        cmd = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    if cmd.get("type") == "stop":
                        await emit(await run_blocking(session.flush))
                        await emit([{"event": "stopped"}])
                        break
        except WebSocketDisconnect:
            pass
        finally:
            if room:
                rooms.leave(room, ws)
                rooms.producing(room, False)
            log.info("ASR stream closed: %.1f s of audio, %d phrases", session.samples / 16000, session.segment)

    @router.websocket("/captions")
    async def listen(ws: WebSocket) -> None:
        room = ws.query_params.get("room")
        if not room or not await _accept(ws, room):
            if not room:
                await ws.close(code=1008)
            return
        rooms.join(room, ws)
        await ws.send_json({"event": "listening", "room_listeners": rooms.listeners(room)})
        try:
            while True:
                if (await ws.receive()).get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            rooms.leave(room, ws)

    router.tts = tts  # for shutdown
    return router
