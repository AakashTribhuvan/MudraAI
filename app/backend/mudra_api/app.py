"""FastAPI application factory."""

from __future__ import annotations

import json
import logging
import time

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from mudra_ml.config import project_path
from mudra_ml.fewshot import FewShotConfig
from mudra_ml.streaming import ModelBundle, SegmenterConfig, StreamingRecognizer
from pydantic import BaseModel, Field

from mudra_api import __version__
from mudra_api.library import Library
from mudra_api.recognizer import SignRecognizer, readable
from mudra_api.recordings import RecordingStore, frames_to_sequence, parse_frame
from mudra_api.references import ReferenceClips
from mudra_api.schemas import LetterRequest, PredictRequest, PredictResponse
from mudra_api.settings import Settings
from signline_emergency.api import emergency_router

log = logging.getLogger("mudra_api")


class RecordingRequest(BaseModel):
    label: str = Field(min_length=1, max_length=64)
    signer: str = Field(default="anonymous", max_length=64)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: list[dict] = Field(min_length=1, max_length=1200)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="MUDRA API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "X-Device-Key", "X-Session-Token"],
    )

    recognizer: SignRecognizer | None = None
    bundle: ModelBundle | None = None
    load_error: str | None = None
    segmenter_cfg = SegmenterConfig.load(project_path("training/configs/streaming.json"))
    try:
        recognizer = SignRecognizer(settings.model_dir, settings.device, settings.uncertain_threshold)
        bundle = ModelBundle(settings.model_dir, str(recognizer.device))
        bundle.threshold = recognizer.threshold
        log.info("Loaded model %s (%d signs) on %s", recognizer.name, len(recognizer.classes), recognizer.device)
    except FileNotFoundError as exc:
        load_error = f"Model files not found in {settings.model_dir}: {exc.filename}"
        log.warning(load_error)
    references = ReferenceClips(settings.reference_split)
    library = Library(settings.letters_model_dir, settings.letters_landmarks, settings.phrases_model_dir,
                      settings.phrases_split, settings.device)
    store = RecordingStore(settings.recordings_dir, bundle, FewShotConfig.load(project_path("training/configs/fewshot.json")))
    if len(store.bank):
        log.info("Custom words from app recordings: %s", ", ".join(store.bank.labels))

    def require_model() -> SignRecognizer:
        if recognizer is None:
            raise HTTPException(
                status_code=503,
                detail=f"{load_error}. Export one with training/scripts/export_model.py.",
            )
        return recognizer

    @app.get("/")
    def root() -> dict:
        return {"project": "MUDRA", "api_version": __version__, "docs": "/docs"}

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": recognizer is not None,
            "model": recognizer.name if recognizer else None,
            "num_signs": len([c for c in recognizer.classes if not c.startswith("_")]) if recognizer else 0,
            "device": str(recognizer.device) if recognizer else None,
            "error": load_error,
            "custom_words": len(store.bank),
        }

    @app.get("/labels")
    def labels() -> dict:
        rec = require_model()
        signs = [c for c in rec.classes if not c.startswith("_")]
        tested = rec.card.get("per_word_test", {})
        return {"count": len(signs), "labels": [
            {"sign": c, "label": readable(c), "has_reference": c in references.available,
             **({"test_correct": tested[c]["correct"], "test_clips": tested[c]["clips"]} if c in tested else {})}
            for c in signs]}

    @app.get("/reference/{sign}")
    def reference(sign: str) -> dict:
        """One real training example of a sign as landmark frames, so people can see which sign the model learned."""
        found = references.get(sign.upper()) or library.reference(sign.upper())
        if found is None:
            raise HTTPException(status_code=404, detail=f"no training example for {sign}")
        sign_id = found["sign"]
        label = f"Letter {sign_id.removeprefix('LETTER_')}" if sign_id.startswith("LETTER_") else readable(sign_id)
        return {**found, "label": label}

    @app.get("/model")
    def model_card() -> dict:
        rec = require_model()
        card = {k: v for k, v in rec.card.items() if k not in ("feature_mean", "feature_std", "classes")}
        card["num_signs"] = len([c for c in rec.classes if not c.startswith("_")])
        card["active_threshold"] = rec.threshold
        return card

    events_log = settings.events_log

    def log_sign(event: dict) -> None:
        """One line per finished sign (decision summary only, never landmarks) for diagnosing live use."""
        res = event.get("result") or {}
        line = {"event": event["event"], **(event.get("stats") or {}),
                "status": res.get("status"), "sign": res.get("sign"), "custom": bool(res.get("custom")),
                "top": [(c["sign"], round(c["confidence"], 3)) for c in res.get("top_k", [])],
                "background": round(float(res.get("background", 0.0)), 3)}
        log_line(line)

    def log_line(line: dict) -> None:
        line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **line}
        try:
            events_log.parent.mkdir(parents=True, exist_ok=True)
            with events_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
        except OSError:
            pass

    session_dir = settings.session_dir
    # Browser diagnostics written to the events log (numbers and the camera's name; never landmarks)
    DIAG_KEYS = ("camera_fps", "tracking_fps", "inference_ms", "delegate", "camera", "width", "height")
    DIAG_LOG_EVERY_S = 60.0
    max_session_frames = 30 * 60 * 20  # 20 minutes at 30 fps

    def save_session(frames: list, width: int, height: int, diag: dict) -> None:
        """Opt-in (MUDRA_SESSION_DIR): one live session's landmarks, for offline replay."""
        if session_dir is None or len(frames) < 30:
            return
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            seq = frames_to_sequence(frames, width, height, {"source": "live_session", "client": diag,
                                                              "segmenter": segmenter_cfg.to_dict(),
                                                              "model": bundle.name if bundle else None})
            seq.save(session_dir / f"session_{time.strftime('%Y%m%d-%H%M%S')}.npz")
        except (OSError, ValueError):
            pass

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        """Live recognition. Client sends {"type":"start","width","height"} then
        {"type":"frame","t":seconds,"hands":[{landmarks,handedness,score}],"pose":[[x,y,z,v]]|null}.
        Server replies only on changes: sign_started, sign_ended (+result), too_short."""
        await ws.accept()
        if bundle is None:
            await ws.send_json({"event": "error", "detail": load_error})
            await ws.close()
            return
        session: StreamingRecognizer | None = None
        frames: list = []
        size, diag, diag_logged = (0, 0), {}, -1e9
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "start":
                    save_session(frames, *size, diag)
                    frames, size = [], (int(msg["width"]), int(msg["height"]))
                    session = StreamingRecognizer(bundle, segmenter_cfg, *size, store.bank)
                    continue
                if msg.get("type") == "diag":  # client tracking diagnostics (fps, delegate), kept with a saved session
                    diag = {k: msg[k] for k in DIAG_KEYS if k in msg}
                    now = time.monotonic()
                    if now - diag_logged >= DIAG_LOG_EVERY_S:  # first report of a session, then once a minute
                        diag_logged = now
                        log_line({"event": "client_diag", **diag})
                    continue
                if msg.get("type") != "frame" or session is None:
                    continue
                frame = parse_frame(msg)
                if session_dir is not None and len(frames) < max_session_frames:
                    frames.append(frame)
                event = session.push(frame)
                if event and event["event"] in ("sign_ended", "too_short"):
                    log_sign(event)
                if event:
                    if "guess" in event:
                        event["guess"]["label"] = readable(event["guess"]["sign"])
                    if "result" in event:
                        for item in event["result"]["top_k"]:
                            item["label"] = readable(item["sign"])
                        sign = event["result"]["sign"]
                        event["result"]["label"] = readable(sign) if sign else None
                    await ws.send_json(event)
        except WebSocketDisconnect:
            save_session(frames, *size, diag)
            return

    @app.get("/events")
    def events(limit: int = 500) -> dict:
        """Latest lines of the live decision log (sign results and browser diagnostics; never landmarks),
        newest first, for the admin Records and Reports pages."""
        limit = max(1, min(int(limit), 5000))
        if not events_log.exists():
            return {"total": 0, "events": []}
        lines = events_log.read_text(encoding="utf-8", errors="replace").splitlines()
        out = []
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
            if len(out) >= limit:
                break
        return {"total": len(lines), "events": out}

    @app.get("/recordings")
    def recordings() -> dict:
        signs = store.summary()
        for s in signs:
            s["label"] = readable(s["sign"])
        return {"total": sum(s["recordings"] for s in signs), "signs": signs,
                "custom_words": [{"sign": w, "label": readable(w), "examples": n}
                                 for w, n in zip(store.bank.labels, store.bank.counts)]}

    @app.post("/recordings")
    def add_recording(req: RecordingRequest) -> dict:
        try:
            frames = [parse_frame(f) for f in req.frames]
            saved = store.add(req.label, req.signer, frames, req.width, req.height)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid recording: {exc}") from exc
        saved["label_readable"] = readable(saved["label"])
        saved["custom_active"] = saved["label"] in store.bank.labels
        saved["in_model"] = saved["label"] in store.known
        saved["quality"] = store.quality().get(saved["label"])
        return saved

    @app.delete("/recordings/{sample_id}")
    def delete_recording(sample_id: str) -> dict:
        if not store.delete(sample_id):
            raise HTTPException(status_code=404, detail="recording not found")
        return {"deleted": sample_id}

    @app.get("/library")
    def library_listing() -> dict:
        """Letters and phrases available besides the word model's vocabulary (/labels)."""
        return library.listing()

    @app.post("/letters/predict")
    def predict_letter(req: LetterRequest) -> dict:
        if library.letter_model is None:
            raise HTTPException(status_code=503, detail="no letter model loaded")
        frames = [[np.asarray(h.landmarks, np.float32) for h in f.hands] for f in req.frames]
        result = library.letter_model.predict_frames(frames, req.width, req.height, req.top_k)
        result["sign"] = f"LETTER_{result['letter']}" if result["letter"] else None
        for t in result["top_k"]:
            t["sign"], t["label"] = f"LETTER_{t['letter']}", t["letter"]
        return result

    @app.post("/phrases/predict", response_model=PredictResponse)
    def predict_phrase(req: PredictRequest) -> dict:
        if library.phrase_model is None:
            raise HTTPException(status_code=503, detail="no phrase model loaded")
        return library.phrase_model.predict(req, settings.min_hand_rate)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> dict:
        return require_model().predict(req, settings.min_hand_rate)

    app.include_router(emergency_router())  # SignLine Emergency (self-contained, /api/v1/emergency)
    return app
