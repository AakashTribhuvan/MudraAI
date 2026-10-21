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
from mudra_api.speech_api import speech_router
from mudra_api.auto_mode import AutoRecognizer
from mudra_api.extra_models import TEAMMATE_TEST_ACCURACY, ExtraModels, MergedFallback, WordSource, evaluation, ours, own_threshold
from mudra_api.isl500_video import ISL500VideoStreamingRecognizer
from mudra_api.static_pose import StaticPoseModel, StaticPoseStreamingRecognizer
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
        expose_headers=["X-TTS-Ms"],
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
    references = ReferenceClips(settings.reference_split, world_dir=settings.reference_world_dir)
    app_manifest = settings.recordings_dir / "metadata" / "manifest_app.csv"
    references.add_app_recordings(app_manifest)  # the owner's own takes sign the words without a training video
    # Selectable word models (GET /models; the stream start message picks one with "model").
    # "sequence" (the deployed word model) is the default; experimental models are opt-in.
    intro_static: StaticPoseModel | None = None
    try:
        intro_static = StaticPoseModel(settings.intro_static_model)
        log.info("Loaded experimental static-pose model (%d profiles, %d live labels)",
                 len(intro_static.classes), len(intro_static.live_labels))
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        log.info("Static-pose model is unavailable: %s", exc)
    extra = ExtraModels(settings.common_word_video_model_dir, settings.isl500_video_model,
                        settings.signline_intro_video_model, settings.device)
    # the merge: every word from the most accurate model that knows it (see extra_models.py); the static-pose
    # models take part only to show which of their words fall below the accuracy bar
    static_sources = [WordSource("intro_static", sorted(ours(c) for c in intro_static.live_labels), lambda *a: None,
                                 TEAMMATE_TEST_ACCURACY["intro_static"])] if intro_static is not None else []
    if settings.static_top10_model.exists():
        try:
            from mudra_api.static_top10 import StaticTop10Model
            top10 = StaticTop10Model(settings.static_top10_model)
            static_sources.append(WordSource("static_top10", sorted(ours(c) for c in top10.classes), lambda *a: None,
                                             TEAMMATE_TEST_ACCURACY["static_top10"]))
        except (OSError, ValueError, KeyError) as exc:
            log.info("Static top-10 model is unavailable: %s", exc)
    merged = MergedFallback(extra.sources(static_sources), set(bundle.classes)) if bundle is not None else None
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

    @app.get("/models")
    def models() -> dict:
        """Word models the stream can use, each with its measured accuracy and where it was measured."""
        out = []
        if recognizer is not None:
            count = len([c for c in recognizer.classes if not c.startswith("_")])
            m = recognizer.card.get("metrics", {})
            version = recognizer.name.removeprefix("isl_mudra_combined_").split("_")[0]
            out.append({"id": "sequence", "name": f"SignLine words ({version})", "model": recognizer.name,
                        "default": True, "experimental": False,
                        "trained_words": count, "live_words": count,
                        "test_accuracy": m.get("test", {}).get("accuracy"), "accuracy_samples": m.get("test_samples"),
                        "accuracy_split": "held-out clip test (same signers as training)"})
        if merged is not None and recognizer is not None:
            per_model = {}
            for w, mid in merged.routing.items():
                per_model.setdefault(mid, []).append(w)
            merged_info = {"trained_words": count + len(merged.extra), "live_words": count + len(merged.extra),
                           "words": merged.extra, "words_by_model": per_model, "excluded_words": merged.excluded,
                           "test_accuracy": m.get("test", {}).get("accuracy"),
                           "accuracy_split": "SignLine test; added words unmeasured on our data",
                           "our_test": {mid: evaluation(d) for mid, d in (
                               ("common_word_video", settings.common_word_video_model_dir),
                               ("isl500_video", settings.isl500_video_model.parent),
                               ("signline_intro_video", settings.signline_intro_video_model.parent))}}
            out.append({"id": "merged", "name": f"Merged: SignLine words + {len(merged.extra)} teammate words",
                        "default": False, "experimental": True, **merged_info})
            if library.letter_model is not None:
                out.append({"id": "auto", "name": "Auto: merged words + A-Z letters", "default": False,
                            "experimental": True, **merged_info, "letters": True})
        return {"models": out}

    @app.get("/labels")
    def labels(model: str = "sequence") -> dict:
        if model == "intro_static":
            if intro_static is None:
                raise HTTPException(status_code=503, detail="The static-pose model is not installed.")
            live = sorted(intro_static.live_labels)
            return {"count": len(live), "labels": [{"sign": s, "label": readable(s), "has_reference": False} for s in live]}
        if model in ("common_word_video", "isl500_video"):
            words = extra.common_words() if model == "common_word_video" else extra.isl500_words()
            if not words:
                raise HTTPException(status_code=503, detail="That model is not installed.")
            return {"count": len(words), "labels": [
                {"sign": s, "label": readable(s), "has_reference": s in references.available} for s in sorted(set(words))]}
        if model not in ("sequence", "merged", "auto"):
            raise HTTPException(status_code=400, detail="Unknown recognition model.")
        rec = require_model()
        signs = [c for c in rec.classes if not c.startswith("_")]
        tested = rec.card.get("per_word_test", {})
        return {"count": len(signs), "labels": [
            {"sign": c, "label": readable(c), "has_reference": c in references.available,
             **({"test_correct": tested[c]["correct"], "test_clips": tested[c]["clips"]} if c in tested else {})}
            for c in signs] + ([
            {"sign": c, "label": readable(c), "has_reference": False, "model": merged.routing[c]}
            for c in merged.extra] if model in ("merged", "auto") and merged is not None else [])}

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
        session: StreamingRecognizer | StaticPoseStreamingRecognizer | ISL500VideoStreamingRecognizer | AutoRecognizer | None = None
        frames: list = []
        size, diag, diag_logged = (0, 0), {}, -1e9
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "start":
                    save_session(frames, *size, diag)
                    frames, size = [], (int(msg["width"]), int(msg["height"]))
                    model_id = msg.get("model", "sequence")
                    if model_id == "intro_static" and intro_static is not None:
                        session = StaticPoseStreamingRecognizer(intro_static, *size)
                    elif model_id == "sequence":
                        session = StreamingRecognizer(bundle, segmenter_cfg, *size, store.bank)
                    elif model_id in ("merged", "auto") and merged is not None:
                        session = StreamingRecognizer(bundle, segmenter_cfg, *size, store.bank, fallback=merged)
                        if model_id == "auto":
                            session = AutoRecognizer(session, library.letter_model, *size)
                    elif model_id == "common_word_video" and extra.common_word is not None:
                        session = StreamingRecognizer(extra.common_word, own_threshold(segmenter_cfg), *size)
                    elif model_id == "isl500_video" and extra.isl500 is not None:
                        session = ISL500VideoStreamingRecognizer(extra.isl500, own_threshold(segmenter_cfg), *size)
                    else:
                        await ws.send_json({"event": "error", "detail": f"Recognition model {model_id!r} is not available."})
                        session = None
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
                pushed = session.push(frame)
                for event in pushed if isinstance(pushed, list) else [pushed] if pushed else []:
                    if event["event"] in ("sign_ended", "too_short"):
                        log_sign(event)
                    if event["event"] == "letter":
                        await ws.send_json(event)
                        continue
                    if "guess" in event:
                        event["guess"]["label"] = readable(event["guess"]["sign"])
                    if "result" in event:
                        for item in event["result"]["top_k"]:
                            item["sign"] = ours(item["sign"])
                            item["label"] = readable(item["sign"])
                        sign = event["result"]["sign"] = ours(event["result"]["sign"]) if event["result"]["sign"] else None
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
            references.add_app_recordings(app_manifest)
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
    app.include_router(speech_router(settings, warm_up=True))  # meeting speech: /tts, /asr, /captions (local only)
    return app
