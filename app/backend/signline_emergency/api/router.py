"""REST API: /api/v1/emergency/...

Emergency users are anonymous: a device key (X-Device-Key, random, made by the browser) owns gestures and
preferences, and each session returns an access token (X-Session-Token) that is required for every call
on that session. Responders log in and send `Authorization: Bearer <token>`; roles are checked here, on
the server, for every responder endpoint.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import CONFIG_DIR, get_settings
from ..db import get_db
from ..models import (
    AnalyticsEvent,
    CustomGesture,
    DispatchAttempt,
    EmergencyIncident,
    EmergencySession,
    Question,
    ResponderAction,
    ResponderNote,
    SignEvent,
    StatusHistory,
    User,
    UserPreference,
)
from ..services import question_manager as qm
from ..services import state_machine as sm
from ..services import workflow as wf
from ..services.audit import audit, track
from ..services.auth import (
    DEVICE_KEY_RE,
    issue_token,
    limiter,
    new_session_token,
    read_token,
    session_token_ok,
    verify_password,
)
from ..services import gesture_report as gr
from ..services.context_engine import category_label, rules_config
from ..services.location import LocationIn, to_record
from ..services.media_store import LocalMediaStore, MediaError, MediaStorageError, get_media_store, validate_video
from ..services.recognition import RecognitionEvent, get_provider
from ..services.recognition.mock_provider import scenario, scenario_config
from ..services.sequence_manager import SequenceError, SignSequenceManager

NOTICE = ("SignLine Emergency is a prototype. In demo mode it contacts no one: reports go only to the SignLine demo "
          "responder desk inside this app. In a real emergency in India, call 112.")


@contextmanager
def errors():
    """Domain errors -> HTTP errors with a readable message."""
    try:
        yield
    except wf.WorkflowError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except (SequenceError, qm.AnswerError, sm.InvalidTransition) as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(422, exc.errors()[0]["msg"]) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _rate(key: str, limit: int | None = None) -> None:
    if not limiter.allow(key, limit):
        raise HTTPException(429, "too many requests, please wait a moment")


# ------------------------------------------------------------------ access
def device_key(x_device_key: Annotated[str | None, Header()] = None) -> str:
    if not x_device_key or not DEVICE_KEY_RE.match(x_device_key):
        raise HTTPException(401, "missing or invalid device key")
    return x_device_key


def emergency_session(session_id: str, db: Annotated[Session, Depends(get_db)],
                      x_session_token: Annotated[str | None, Header()] = None) -> EmergencySession:
    session = db.get(EmergencySession, session_id)
    # same answer for "no such session" and "wrong token", so ids cannot be probed
    if session is None or not session_token_ok(x_session_token, session.access_token_hash):
        raise HTTPException(404, "session not found")
    return session


def responder(db: Annotated[Session, Depends(get_db)], authorization: Annotated[str | None, Header()] = None) -> User:
    token = (authorization or "").removeprefix("Bearer ").strip()
    user_id = read_token(token) if token else None
    user = db.get(User, user_id) if user_id else None
    if user is None or not user.is_active:
        raise HTTPException(401, "sign in as a responder")
    if user.role not in ("RESPONDER", "ADMIN"):
        raise HTTPException(403, "responder role required")
    return user


DB = Annotated[Session, Depends(get_db)]
EmSession = Annotated[EmergencySession, Depends(emergency_session)]
DeviceKey = Annotated[str, Depends(device_key)]
Responder = Annotated[User, Depends(responder)]


# ------------------------------------------------------------------ views
def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def _raw(e: SignEvent) -> dict:
    return {"event_id": e.id, "event_type": e.event_type, "sign": e.sign, "confidence": e.confidence,
            "source": e.source, "gesture_id": e.gesture_id, "start_time": _iso(e.start_time), "end_time": _iso(e.end_time)}


def _question(q: Question) -> dict:
    return {"id": q.id, "key": q.key, "text": q.text, "answer_type": q.answer_type, "options": q.options,
            "status": q.status, "reason": q.reason}


def _history(db: Session, session_id: str) -> list[dict]:
    rows = db.scalars(select(StatusHistory).where(StatusHistory.session_id == session_id).order_by(StatusHistory.id))
    return [{"from": h.from_status, "to": h.to_status, "label": sm.LABELS[h.to_status], "actor": h.actor_type,
             "reason": h.reason, "at": _iso(h.created_at)} for h in rows]


def _dispatches(db: Session, incident_id: str) -> list[dict]:
    rows = db.scalars(select(DispatchAttempt).where(DispatchAttempt.incident_id == incident_id).order_by(DispatchAttempt.created_at))
    return [{"id": d.id, "provider": d.provider, "destination": d.destination, "status": d.status,
             "receipt_id": d.receipt_id, "error": d.error, "simulated": d.simulated, "at": _iso(d.created_at)} for d in rows]


def _incident(db: Session, inc: EmergencyIncident, for_responder: bool = False) -> dict:
    view = {"id": inc.id, "session_id": inc.session_id, "category": inc.category, "category_label": category_label(inc.category),
            "summary": inc.summary, "user_confirmed": inc.user_confirmed, "status": inc.status,
            "status_label": sm.LABELS[inc.status], "sharing_status": inc.sharing_status,
            "communication_needs": inc.communication_needs, "location": inc.location, "priority": inc.priority,
            "demo": inc.demo, "created_at": _iso(inc.created_at),
            "facts": [{"kind": f.kind, "key": f.key, "value": f.value, "provenance": f.provenance} for f in inc.facts],
            "dispatch_attempts": _dispatches(db, inc.id),
            "gesture_id": inc.gesture_id, "emergency_contact": gr.find_contact(inc.emergency_contact_id),
            "emergency_message": inc.emergency_message,
            "report_sharing_status": inc.sharing_status, "video_sharing_status": inc.video_sharing_status,
            "video": None if not inc.video_storage_key else {
                "mime_type": inc.video_mime_type, "duration_seconds": inc.video_duration_seconds,
                "sharing_consent": inc.sharing_consent, "sharing_consent_at": _iso(inc.sharing_consent_at)}}
    attempts = view["dispatch_attempts"]
    view["delivery_receipt"] = next((a["receipt_id"] for a in reversed(attempts) if a["status"] == "DELIVERED"), None)
    if for_responder:
        notes = db.execute(select(ResponderNote, User.display_name).join(User, User.id == ResponderNote.user_id)
                           .where(ResponderNote.incident_id == inc.id).order_by(ResponderNote.id)).all()
        view["notes"] = [{"text": n.text, "by": name, "at": _iso(n.created_at)} for n, name in notes]
        view["history"] = _history(db, inc.session_id)
        view["sequence"] = SignSequenceManager(db, inc.session).normalized()
    return view


def session_view(db: Session, session: EmergencySession) -> dict:
    manager = SignSequenceManager(db, session)
    sequence, ctx = wf.context_of(db, session)
    category, summary = wf.effective(session, ctx)
    questions = list(db.scalars(select(Question).where(Question.session_id == session.id).order_by(Question.created_at)))
    answers = qm.answers_of(db, session.id)
    sc = scenario(session.scenario_id) if session.scenario_id else None
    revisions = manager.revisions()
    last = revisions[-1] if revisions else None
    return {
        "id": session.id, "status": session.status, "status_label": sm.LABELS[session.status], "mode": session.mode,
        "demo": session.demo, "editable": session.status in wf.EDITABLE,
        "allowed_next": sorted(sm.TRANSITIONS[session.status]),
        "scenario": None if sc is None else {
            "id": sc["id"], "title": sc["title"], "description": sc["description"], "hint": sc.get("hint"),
            "cursor": session.scenario_cursor, "total": len(sc.get("events", [])) or 1, "paused": session.scenario_paused},
        "raw_events": [_raw(e) for e in manager.raw()],
        "sequence": sequence,
        "can_undo": bool(last) and SignSequenceManager._effective(last) > 0,
        "revision_count": len(revisions),
        "context": {**ctx, "category": category, "category_label": category_label(category) if category else None,
                    "summary": summary, "user_corrected": bool(session.user_category or session.user_summary)},
        "questions": [_question(q) | {"answer": answers.get(q.key) if q.status == "ANSWERED" else None} for q in questions],
        "location": session.location,
        "general_message": session.general_message,
        "incident": _incident(db, session.incident) if session.incident else None,
        "history": _history(db, session.id),
        "created_at": _iso(session.created_at),
        "notice": NOTICE,
    }


def _commit_view(db: Session, session: EmergencySession) -> dict:
    db.commit()
    return session_view(db, session)


# ------------------------------------------------------------------ request bodies
class SessionIn(BaseModel):
    mode: Literal["SIGN", "GESTURE", "BUTTONS"] = "SIGN"
    scenario_id: str | None = Field(default=None, max_length=40)


class MockSignIn(BaseModel):
    sign: str = Field(min_length=1, max_length=80)
    confidence: float = Field(default=0.9, ge=0, le=1)


class ScenarioIn(BaseModel):
    action: Literal["next", "all", "pause", "resume", "restart"]


class SelectionIn(BaseModel):
    need: str = Field(max_length=40)
    detail: str | None = Field(default=None, max_length=40)


class CorrectIn(BaseModel):
    event_id: str = Field(max_length=40)
    sign: str = Field(min_length=1, max_length=80)


class RemoveIn(BaseModel):
    event_id: str = Field(max_length=40)


class AnswerIn(BaseModel):
    value: str | list[str]


class ReportCorrectionIn(BaseModel):
    category: str | None = Field(default=None, max_length=40)
    summary: str | None = Field(default=None, max_length=1000)


class GestureIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    intent: str = Field(max_length=40)
    message: str = Field(min_length=1, max_length=500)
    enabled: bool = True
    require_confirmation: bool = True
    cooldown_s: int = Field(default=10, ge=0, le=3600)
    image_ref: str | None = Field(default=None, max_length=255)
    emergency_contact_id: str | None = Field(default=None, max_length=40)


class GesturePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)
    intent: str | None = Field(default=None, max_length=40)
    message: str | None = Field(default=None, min_length=1, max_length=500)
    enabled: bool | None = None
    require_confirmation: bool | None = None
    cooldown_s: int | None = Field(default=None, ge=0, le=3600)
    emergency_contact_id: str | None = Field(default=None, max_length=40)


class AssistIn(BaseModel):
    contact_id: str | None = Field(default=None, max_length=40)
    location: LocationIn | None = None
    include_video: bool = False
    sharing_consent: bool = False


class GestureReportIn(BaseModel):
    contact_id: str | None = Field(default=None, max_length=40)
    location_text: str | None = Field(default=None, max_length=300)


class PreferencesIn(BaseModel):
    communication_needs: list[str] = Field(default_factory=list, max_length=12)
    preferred_language: str = Field(default="en", max_length=10)


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)


class NoteIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


COMMUNICATION_NEEDS = [
    "Prefers text communication", "Prefers Indian Sign Language", "Please do not call; message instead",
    "Please face me while talking", "May need an ISL interpreter", "Uses a hearing aid", "Can lip-read a little",
]


def _gesture(g: CustomGesture) -> dict:
    return {"id": g.id, "name": g.name, "description": g.description, "intent": g.intent, "category": g.category,
            "category_label": category_label(g.category), "message": g.message, "enabled": g.enabled,
            "require_confirmation": bool(g.activation.get("require_confirmation", True)),
            "cooldown_s": int(g.activation.get("cooldown_s", 0)), "image_ref": g.image_ref,
            "last_triggered_at": _iso(g.last_triggered_at),
            "has_video": bool(g.video_storage_key), "video_mime_type": g.video_mime_type,
            "video_duration_seconds": g.video_duration_seconds, "recording_consent": bool(g.recording_consent),
            "recording_consent_at": _iso(g.recording_consent_at), "emergency_contact_id": g.emergency_contact_id,
            "emergency_contact": gr.find_contact(g.emergency_contact_id), "created_at": _iso(g.created_at),
            "updated_at": _iso(g.updated_at)}


def _media() -> LocalMediaStore:
    try:
        return get_media_store()
    except MediaStorageError as exc:
        raise HTTPException(503, str(exc)) from exc


def _video_response(storage_key: str, mime: str | None) -> Response:
    try:
        data = _media().read(storage_key)
    except MediaStorageError as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(content=data, media_type=mime or "application/octet-stream", headers={"Cache-Control": "no-store"})


def _check_contact(contact_id: str | None) -> None:
    if contact_id is not None and gr.find_contact(contact_id) is None:
        raise HTTPException(422, "unknown emergency contact")


MAX_UPLOAD_READ = 25 * 1024 * 1024 + 1  # one byte more than allowed, so too-large files are detected


def build_router() -> APIRouter:
    r = APIRouter(prefix="/api/v1/emergency", tags=["emergency"])
    buttons = json.loads((CONFIG_DIR / "emergency_buttons.json").read_text(encoding="utf-8"))

    # -------------------------------------------------------------- public config
    @r.get("/config")
    def config() -> dict:
        s, rules = get_settings(), rules_config()
        return {
            "demo_mode": s.demo_mode, "recognition_provider": get_provider().name,
            "recognition_simulated": get_provider().simulated, "dispatch_provider": s.dispatch_provider,
            "dispatch_destination": s.dispatch_destination, "notice": NOTICE,
            "categories": [{"id": c, "label": rules["category_labels"][c]} for c in rules["categories"] if c != "UNKNOWN"],
            "intents": [{"id": i, "category": c, "category_label": rules["category_labels"][c]}
                        for i, c in rules["intent_categories"].items()],
            "buttons": buttons, "communication_needs": COMMUNICATION_NEEDS,
            "vocabulary": scenario_config()["vocabulary"] if s.demo_mode else [],
            "scenarios": [{"id": x["id"], "title": x["title"], "description": x["description"], "hint": x.get("hint")}
                          for x in scenario_config()["scenarios"]] if s.demo_mode else [],
        }

    # -------------------------------------------------------------- sessions
    @r.post("/sessions")
    async def create_session(body: SessionIn, request: Request, key: DeviceKey,
                             db: DB) -> dict:
        _rate(f"session:{key}")
        _rate(f"session-ip:{_client(request)}", 60)
        settings = get_settings()
        if body.scenario_id and (not settings.demo_mode or scenario(body.scenario_id) is None):
            raise HTTPException(422, "unknown demo scenario")
        token, token_hash = new_session_token()
        session = EmergencySession(device_key=key, access_token_hash=token_hash, mode=body.mode, demo=settings.demo_mode,
                                   scenario_id=body.scenario_id)
        db.add(session)
        db.flush()
        db.add(StatusHistory(session_id=session.id, from_status=None, to_status="SESSION_CREATED", actor_type="USER",
                             reason="session started"))
        await get_provider().start_session(session.id)
        track(db, "SESSION_STARTED", session.id, mode=body.mode, scenario_id=body.scenario_id)
        db.commit()
        return {"session": session_view(db, session), "access_token": token}

    @r.get("/sessions/{session_id}")
    def get_session(session: EmSession, db: DB) -> dict:
        return session_view(db, session)

    @r.post("/sessions/{session_id}/mock-sign")
    async def mock_sign(body: MockSignIn, session: EmSession,
                        db: DB) -> dict:
        if not get_settings().demo_mode or not get_provider().simulated:
            raise HTTPException(403, "manual signs are only available with the demo (mock) provider")
        with errors():
            wf.ingest(db, session, await get_provider().recognize(session.id, body.model_dump()))
        return _commit_view(db, session)

    @r.post("/sessions/{session_id}/scenario")
    async def scenario_control(body: ScenarioIn, session: EmSession,
                               db: DB) -> dict:
        sc = scenario(session.scenario_id) if session.scenario_id else None
        if sc is None:
            raise HTTPException(409, "this session has no demo scenario")
        with errors():
            if body.action == "pause":
                session.scenario_paused = True
            elif body.action == "resume":
                session.scenario_paused = False
            elif body.action == "restart":
                wf.require_editable(session)
                SignSequenceManager(db, session).reset()
                session.scenario_cursor, session.scenario_paused = 0, False
                session.user_category = session.user_summary = None
                wf.update_context(db, session)
            elif "gesture" in sc:  # scenario E: the user's configured gesture is triggered once
                if session.scenario_cursor == 0:
                    gesture = wf.ensure_demo_gesture(db, session.device_key, sc["gesture"])
                    wf.trigger_gesture(db, session, gesture)
                    session.scenario_cursor = 1
            else:
                if session.scenario_paused:
                    raise wf.WorkflowError("playback is paused")
                steps = len(sc["events"]) - session.scenario_cursor if body.action == "all" else 1
                for _ in range(max(steps, 0)):
                    events = await get_provider().recognize(session.id, {"scenario_id": sc["id"], "cursor": session.scenario_cursor})
                    if not events:
                        break
                    wf.ingest(db, session, events)
                    session.scenario_cursor += 1
        return _commit_view(db, session)

    @r.post("/sessions/{session_id}/selections")
    def select_need(body: SelectionIn, session: EmSession,
                    db: DB) -> dict:
        need = next((n for n in buttons["needs"] if n["id"] == body.need), None)
        if need is None:
            raise HTTPException(422, "unknown option")
        signals = list(need["signals"])
        if body.detail:
            detail = next((d for d in buttons["details"].get(body.need, []) if d["id"] == body.detail), None)
            if detail is None:
                raise HTTPException(422, "unknown detail option")
            signals += detail["signals"]
        now = datetime.now(timezone.utc)
        with errors():
            wf.ingest(db, session, [RecognitionEvent(event_type="BUTTON_SELECTED", session_id=session.id, sign=s,
                                                     confidence=None, start_time=now, end_time=now, source="BUTTONS")
                                    for s in signals])
        return _commit_view(db, session)

    @r.post("/sessions/{session_id}/gestures/{gesture_id}/trigger")
    def trigger(gesture_id: str, session: EmSession, db: DB) -> dict:
        gesture = db.get(CustomGesture, gesture_id)
        if gesture is None or gesture.device_key != session.device_key:
            raise HTTPException(404, "gesture not found")
        with errors():
            wf.trigger_gesture(db, session, gesture)
        return _commit_view(db, session)

    @r.post("/sessions/{session_id}/provider-events")
    async def provider_events(payload: dict, session: EmSession,
                              db: DB) -> dict:
        """Real-model adapter input (a /stream `sign_ended` message). Mock provider: {"sign", "confidence"}."""
        with errors():
            events = await get_provider().recognize(session.id, payload)
            wf.ingest(db, session, events)
        return _commit_view(db, session)

    # -------------------------------------------------------------- sequence editing
    def _edit(session: EmergencySession, db: Session, action) -> dict:
        with errors():
            wf.require_editable(session)
            action(SignSequenceManager(db, session))
            track(db, "SIGN_CORRECTED", session.id)
            wf.update_context(db, session)
        return _commit_view(db, session)

    @r.post("/sessions/{session_id}/sequence/correct")
    def correct(body: CorrectIn, session: EmSession, db: DB) -> dict:
        return _edit(session, db, lambda m: m.correct(body.event_id, body.sign))

    @r.post("/sessions/{session_id}/sequence/remove")
    def remove(body: RemoveIn, session: EmSession, db: DB) -> dict:
        return _edit(session, db, lambda m: m.remove(body.event_id))

    @r.post("/sessions/{session_id}/sequence/undo")
    def undo(session: EmSession, db: DB) -> dict:
        return _edit(session, db, lambda m: m.undo())

    @r.post("/sessions/{session_id}/sequence/reset")
    def reset(session: EmSession, db: DB) -> dict:
        return _edit(session, db, lambda m: m.reset())

    # -------------------------------------------------------------- questions
    def _own_question(db: Session, session: EmergencySession, question_id: str) -> Question:
        q = db.get(Question, question_id)
        if q is None or q.session_id != session.id:
            raise HTTPException(404, "question not found")
        return q

    @r.post("/sessions/{session_id}/questions/{question_id}/answer")
    def answer(question_id: str, body: AnswerIn, session: EmSession,
               db: DB) -> dict:
        with errors():
            wf.require_editable(session)
            qm.answer(db, _own_question(db, session, question_id), body.value)
            db.flush()
            wf.update_context(db, session)
        return _commit_view(db, session)

    @r.post("/sessions/{session_id}/questions/{question_id}/skip")
    def skip(question_id: str, session: EmSession, db: DB) -> dict:
        with errors():
            wf.require_editable(session)
            qm.skip(db, _own_question(db, session, question_id))
            db.flush()
            wf.update_context(db, session)
        return _commit_view(db, session)

    # -------------------------------------------------------------- location
    @r.put("/sessions/{session_id}/location")
    def set_location(body: LocationIn, session: EmSession,
                     db: DB) -> dict:
        with errors():
            wf.require_editable(session)
            session.location = to_record(body, get_settings().demo_mode)
            audit(db, "SESSION", session.id, "LOCATION_SHARED", "USER", source=body.source)
            wf.update_context(db, session)
        return _commit_view(db, session)

    @r.delete("/sessions/{session_id}/location")
    def clear_location(session: EmSession, db: DB) -> dict:
        with errors():
            wf.require_editable(session)
            session.location = None
            audit(db, "SESSION", session.id, "LOCATION_REMOVED", "USER")
            wf.update_context(db, session)
        return _commit_view(db, session)

    # -------------------------------------------------------------- review, confirm, share
    @r.post("/sessions/{session_id}/review")
    def review(session: EmSession, db: DB) -> dict:
        with errors():
            wf.review(db, session)
        return _commit_view(db, session)

    @r.post("/sessions/{session_id}/correction")
    def correct_report(body: ReportCorrectionIn, session: EmSession,
                       db: DB) -> dict:
        with errors():
            wf.correct_report(db, session, body.category, body.summary)
        return _commit_view(db, session)

    @r.post("/sessions/{session_id}/confirm")
    def confirm(session: EmSession, db: DB) -> dict:
        with errors():
            wf.confirm(db, session)
        return _commit_view(db, session)

    @r.post("/sessions/{session_id}/share")
    def share(session: EmSession, db: DB) -> dict:
        _rate(f"share:{session.id}", 10)
        with errors():
            wf.share(db, session)
        return _commit_view(db, session)

    @r.post("/sessions/{session_id}/cancel")
    def cancel(session: EmSession, db: DB) -> dict:
        with errors():
            wf.cancel(db, session)
        return _commit_view(db, session)

    # -------------------------------------------------------------- gestures & preferences (per device)
    @r.get("/gestures")
    def list_gestures(key: DeviceKey, db: DB) -> list[dict]:
        rows = db.scalars(select(CustomGesture).where(CustomGesture.device_key == key).order_by(CustomGesture.created_at))
        return [_gesture(g) for g in rows]

    def _intent_category(intent: str) -> str:
        cat = rules_config()["intent_categories"].get(intent)
        if cat is None:
            raise HTTPException(422, "unknown intent")
        return cat

    @r.post("/gestures")
    def create_gesture(body: GestureIn, key: DeviceKey, db: DB) -> dict:
        count = db.scalar(select(func.count()).select_from(CustomGesture).where(CustomGesture.device_key == key))
        if count >= 20:
            raise HTTPException(409, "at most 20 gestures per device")
        _check_contact(body.emergency_contact_id)
        g = CustomGesture(device_key=key, name=body.name.strip(), description=body.description, intent=body.intent,
                          category=_intent_category(body.intent), message=body.message.strip(), enabled=body.enabled,
                          image_ref=body.image_ref, emergency_contact_id=body.emergency_contact_id,
                          activation={"require_confirmation": body.require_confirmation, "cooldown_s": body.cooldown_s})
        db.add(g)
        db.commit()
        return _gesture(g)

    def _own_gesture(db: Session, key: str, gesture_id: str) -> CustomGesture:
        g = db.get(CustomGesture, gesture_id)
        if g is None or g.device_key != key:
            raise HTTPException(404, "gesture not found")
        return g

    @r.patch("/gestures/{gesture_id}")
    def update_gesture(gesture_id: str, body: GesturePatch, key: DeviceKey, db: DB) -> dict:
        g = _own_gesture(db, key, gesture_id)
        data = body.model_dump(exclude_none=True)
        if "intent" in data:
            g.intent, g.category = data["intent"], _intent_category(data["intent"])
        if "emergency_contact_id" in data:
            _check_contact(data["emergency_contact_id"])
        for f in ("name", "description", "message", "enabled", "emergency_contact_id"):
            if f in data:
                setattr(g, f, data[f])
        activation = dict(g.activation)
        for f in ("require_confirmation", "cooldown_s"):
            if f in data:
                activation[f] = data[f]
        g.activation = activation
        db.commit()
        return _gesture(g)

    @r.delete("/gestures/{gesture_id}")
    def delete_gesture(gesture_id: str, key: DeviceKey, db: DB) -> dict:
        g = _own_gesture(db, key, gesture_id)
        video = g.video_storage_key
        db.delete(g)
        db.commit()
        _media().delete(video)  # the recording goes with its gesture
        return {"deleted": gesture_id, "video_deleted": bool(video)}

    # -- demonstration video (owner only; never logged, never used for training)
    @r.put("/gestures/{gesture_id}/video")
    async def put_gesture_video(gesture_id: str, key: DeviceKey, db: DB, file: UploadFile = File(...),
                                duration_s: float = Form(...), recording_consent: bool = Form(...)) -> dict:
        g = _own_gesture(db, key, gesture_id)
        if not recording_consent:
            raise HTTPException(422, "recording consent is required to store a video")
        data = await file.read(MAX_UPLOAD_READ)
        try:
            mime = validate_video(data, file.content_type, duration_s)
            new_key = _media().save(data, mime)
        except MediaError as exc:
            raise HTTPException(422, str(exc)) from exc
        except MediaStorageError as exc:
            raise HTTPException(503, str(exc)) from exc
        old = g.video_storage_key
        g.video_storage_key, g.video_mime_type = new_key, mime
        g.video_duration_seconds, g.video_size_bytes = round(float(duration_s), 2), len(data)
        g.recording_consent, g.recording_consent_at = True, datetime.now(timezone.utc)
        db.commit()
        _media().delete(old)  # retake: the previous recording is removed
        track(db, "GESTURE_VIDEO_SAVED", None, mime=mime)
        db.commit()
        return _gesture(g)

    @r.get("/gestures/{gesture_id}/video")
    def get_gesture_video(gesture_id: str, key: DeviceKey, db: DB) -> Response:
        g = _own_gesture(db, key, gesture_id)
        if not g.video_storage_key:
            raise HTTPException(404, "this gesture has no video")
        return _video_response(g.video_storage_key, g.video_mime_type)

    @r.delete("/gestures/{gesture_id}/video")
    def delete_gesture_video(gesture_id: str, key: DeviceKey, db: DB) -> dict:
        g = _own_gesture(db, key, gesture_id)
        old = g.video_storage_key
        g.video_storage_key = g.video_mime_type = g.video_duration_seconds = g.video_size_bytes = None
        db.commit()
        _media().delete(old)
        return _gesture(g)

    # -- emergency contacts and the report prepared from a gesture (nothing is sent or dialled here)
    @r.get("/contacts")
    def contacts(region: str | None = None) -> dict:
        try:
            reg, items = gr.region_contacts(region)
        except KeyError as exc:
            raise HTTPException(404, "unknown region") from exc
        cfg = gr.contacts_config()
        return {"region": reg, "regions": {k: v["label"] for k, v in cfg["regions"].items()},
                "default_contact_id": cfg["default_contact_id"], "contacts": items, "note": cfg["_note"]}

    @r.post("/gestures/{gesture_id}/report")
    def gesture_report(gesture_id: str, body: GestureReportIn, key: DeviceKey, db: DB) -> dict:
        g = _own_gesture(db, key, gesture_id)
        contact_id = body.contact_id or g.emergency_contact_id
        contact = gr.find_contact(contact_id) if contact_id else gr.default_contact()
        if contact is None:
            raise HTTPException(422, "unknown emergency contact")
        p = db.scalar(select(UserPreference).where(UserPreference.device_key == key))
        report = gr.build_report(g, contact, p.communication_needs if p else [], body.location_text)
        track(db, "GESTURE_REPORT_PREPARED", None, category=g.category)
        db.commit()
        return {"gesture": _gesture(g), **report}

    @r.post("/gestures/{gesture_id}/request-assistance")
    async def request_assistance(gesture_id: str, body: AssistIn, request: Request, key: DeviceKey, db: DB) -> dict:
        """Prepare a report from a gesture through the normal flow (session -> gesture -> review -> confirm) up to
        READY_TO_SHARE. Nothing is shared: the user shares it with POST /sessions/{id}/share, after which the
        status shows delivered / acknowledged / resolved or delivery failed."""
        g = _own_gesture(db, key, gesture_id)
        if not g.enabled:
            raise HTTPException(409, "this gesture is turned off")
        contact_id = body.contact_id or g.emergency_contact_id
        contact = gr.find_contact(contact_id) if contact_id else gr.default_contact()
        if contact is None:
            raise HTTPException(422, "unknown emergency contact")
        if body.include_video and not g.video_storage_key:
            raise HTTPException(422, "this gesture has no video")
        if body.include_video and not body.sharing_consent:
            raise HTTPException(422, "sharing the video needs your consent")
        _rate(f"session:{key}")
        _rate(f"session-ip:{_client(request)}", 60)
        settings = get_settings()
        token, token_hash = new_session_token()
        session = EmergencySession(device_key=key, access_token_hash=token_hash, mode="GESTURE", demo=settings.demo_mode)
        db.add(session)
        db.flush()
        db.add(StatusHistory(session_id=session.id, from_status=None, to_status="SESSION_CREATED", actor_type="USER",
                             reason="session started from a personal gesture"))
        await get_provider().start_session(session.id)
        track(db, "SESSION_STARTED", session.id, mode="GESTURE", scenario_id=None)
        location_text = None
        with errors():
            wf.trigger_gesture(db, session, g)
            if body.location is not None:
                session.location = to_record(body.location, settings.demo_mode)
                audit(db, "SESSION", session.id, "LOCATION_SHARED", "USER", source=body.location.source)
                loc = body.location
                location_text = loc.text or (f"{loc.lat:.5f}, {loc.lon:.5f}" + (f" (±{loc.accuracy_m:.0f} m)" if loc.accuracy_m else ""))
            _, ctx = wf.context_of(db, session)
            category, summary = wf.effective(session, ctx)
            if not category or not summary:  # the gesture's own category and message, as saved by the user
                wf.correct_report(db, session, None if category else g.category, None if summary else g.message)
            wf.review(db, session)
            incident = wf.confirm(db, session)
        p = db.scalar(select(UserPreference).where(UserPreference.device_key == key))
        report = gr.build_report(g, contact, p.communication_needs if p else [], location_text)
        incident.gesture_id, incident.emergency_contact_id, incident.emergency_message = g.id, contact["id"], report["text"]
        if body.include_video:
            try:
                incident.video_storage_key = _media().copy(g.video_storage_key)
            except MediaStorageError as exc:
                db.rollback()
                raise HTTPException(503, str(exc)) from exc
            incident.video_mime_type, incident.video_duration_seconds = g.video_mime_type, g.video_duration_seconds
            incident.sharing_consent, incident.sharing_consent_at = True, datetime.now(timezone.utc)
            incident.video_sharing_status = "ATTACHED"
            audit(db, "INCIDENT", incident.id, "VIDEO_ATTACHED_WITH_CONSENT", "USER")
        db.commit()
        return {"session": session_view(db, session), "access_token": token, "report": report}

    @r.post("/gestures/{gesture_id}/test")
    def test_gesture(gesture_id: str, key: DeviceKey, db: DB) -> dict:
        """What this gesture would add to a report. Nothing is sent and no session is created."""
        g = _own_gesture(db, key, gesture_id)
        track(db, "GESTURE_TESTED", None, category=g.category)
        db.commit()
        return {"gesture": _gesture(g), "would_report": {"category": g.category, "category_label": category_label(g.category),
                                                         "message": g.message, "provenance": "USER_REPORTED"},
                "sent": False}

    @r.get("/preferences")
    def get_preferences(key: DeviceKey, db: DB) -> dict:
        p = db.scalar(select(UserPreference).where(UserPreference.device_key == key))
        return {"communication_needs": p.communication_needs if p else [], "preferred_language": p.preferred_language if p else "en"}

    @r.put("/preferences")
    def put_preferences(body: PreferencesIn, key: DeviceKey, db: DB) -> dict:
        p = db.scalar(select(UserPreference).where(UserPreference.device_key == key))
        if p is None:
            p = UserPreference(device_key=key)
            db.add(p)
        p.communication_needs = [n.strip()[:120] for n in body.communication_needs if n.strip()]
        p.preferred_language = body.preferred_language
        db.commit()
        return {"communication_needs": p.communication_needs, "preferred_language": p.preferred_language}

    # -------------------------------------------------------------- responders
    @r.post("/responder/login")
    def login(body: LoginIn, request: Request, db: DB) -> dict:
        _rate(f"login:{_client(request)}", 10)
        user = db.scalar(select(User).where(User.username == body.username.strip().lower()))
        if user is None or not user.is_active or not verify_password(body.password, user.password_hash):
            audit(db, "USER", body.username[:40], "LOGIN_FAILED", "RESPONDER")
            db.commit()
            raise HTTPException(401, "wrong username or password")
        audit(db, "USER", user.id, "LOGIN", "RESPONDER", user.id)
        db.commit()
        return {"token": issue_token(user.id), "expires_in": get_settings().token_ttl_s,
                "user": {"id": user.id, "username": user.username, "display_name": user.display_name, "role": user.role}}

    @r.get("/responder/me")
    def me(user: Responder) -> dict:
        return {"id": user.id, "username": user.username, "display_name": user.display_name, "role": user.role}

    def _visible(db: Session, incident_id: str) -> EmergencyIncident:
        inc = db.get(EmergencyIncident, incident_id)
        # responders only see reports that were actually delivered to their desk
        if inc is None or inc.sharing_status != "DELIVERED":
            raise HTTPException(404, "incident not found")
        return inc

    @r.get("/responder/incidents")
    def incidents(user: Responder, db: DB, status: str | None = None) -> list[dict]:
        q = select(EmergencyIncident).where(EmergencyIncident.sharing_status == "DELIVERED")
        if status:
            q = q.where(EmergencyIncident.status == status)
        rows = db.scalars(q.order_by(EmergencyIncident.created_at.desc()).limit(200))
        return [{"id": i.id, "category": i.category, "category_label": category_label(i.category), "summary": i.summary,
                 "status": i.status, "status_label": sm.LABELS[i.status], "priority": i.priority, "demo": i.demo,
                 "has_location": bool(i.location), "communication_needs": i.communication_needs,
                 "created_at": _iso(i.created_at)} for i in rows]

    @r.get("/responder/incidents/{incident_id}")
    def incident(incident_id: str, user: Responder, db: DB) -> dict:
        inc = _visible(db, incident_id)
        db.add(ResponderAction(incident_id=inc.id, user_id=user.id, action="VIEW"))
        audit(db, "INCIDENT", inc.id, "VIEWED", "RESPONDER", user.id)
        db.commit()
        return _incident(db, inc, for_responder=True)

    def _responder_status(incident_id: str, target: str, user: User, db: Session, event: str) -> dict:
        inc = _visible(db, incident_id)
        if target not in sm.RESPONDER_SETTABLE:
            raise HTTPException(403, "responders cannot set this status")
        with errors():
            wf.set_status(db, inc.session, target, "RESPONDER", user.id, reason=f"{target.lower()} by responder")
        db.add(ResponderAction(incident_id=inc.id, user_id=user.id, action="ACKNOWLEDGE" if target == "ACKNOWLEDGED" else "RESOLVE"))
        track(db, event, inc.session_id, category=inc.category)
        db.commit()
        return _incident(db, inc, for_responder=True)

    @r.get("/responder/incidents/{incident_id}/video")
    def incident_video(incident_id: str, user: Responder, db: DB) -> Response:
        """The video the user chose to share with this report (delivered reports only)."""
        inc = _visible(db, incident_id)
        if not inc.video_storage_key or not inc.sharing_consent:
            raise HTTPException(404, "no video was shared with this report")
        response = _video_response(inc.video_storage_key, inc.video_mime_type)
        db.add(ResponderAction(incident_id=inc.id, user_id=user.id, action="VIEW_VIDEO"))
        audit(db, "INCIDENT", inc.id, "VIDEO_VIEWED", "RESPONDER", user.id)
        db.commit()
        return response

    @r.post("/responder/incidents/{incident_id}/acknowledge")
    def acknowledge(incident_id: str, user: Responder, db: DB) -> dict:
        return _responder_status(incident_id, "ACKNOWLEDGED", user, db, "RESPONDER_ACKNOWLEDGED")

    @r.post("/responder/incidents/{incident_id}/resolve")
    def resolve(incident_id: str, user: Responder, db: DB) -> dict:
        return _responder_status(incident_id, "RESOLVED", user, db, "INCIDENT_RESOLVED")

    @r.post("/responder/incidents/{incident_id}/notes")
    def add_note(incident_id: str, body: NoteIn, user: Responder, db: DB) -> dict:
        inc = _visible(db, incident_id)
        db.add(ResponderNote(incident_id=inc.id, user_id=user.id, text=body.text.strip()))
        db.add(ResponderAction(incident_id=inc.id, user_id=user.id, action="NOTE"))
        audit(db, "INCIDENT", inc.id, "NOTE_ADDED", "RESPONDER", user.id)
        db.commit()
        return _incident(db, inc, for_responder=True)

    @r.get("/responder/analytics")
    def analytics(user: Responder, db: DB, days: int = 30) -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=max(1, min(days, 365)))
        counts = dict(db.execute(select(AnalyticsEvent.event_type, func.count()).where(AnalyticsEvent.created_at >= since)
                                 .group_by(AnalyticsEvent.event_type)).all())
        by_category = dict(db.execute(select(EmergencyIncident.category, func.count()).where(EmergencyIncident.created_at >= since)
                                      .group_by(EmergencyIncident.category)).all())
        return {"since": since.isoformat(), "events": counts,
                "incidents_by_category": {category_label(k): v for k, v in by_category.items()},
                "note": "Aggregated counts only. Analytics never contain signs, messages, summaries or locations."}

    return r
