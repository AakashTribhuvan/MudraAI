"""Emergency session workflow: ties events, context, questions, confirmation, sharing and status together.

The API layer only calls these functions; all status changes go through `set_status`, which checks the
state machine and writes status history and an audit event.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import (
    ContextEvent,
    CustomGesture,
    DispatchAttempt,
    EmergencyIncident,
    EmergencySession,
    StatusHistory,
)
from . import question_manager as qm
from . import state_machine as sm
from .audit import audit, track
from .context_engine import category_label, evaluate, rules_config
from .dispatch import get_dispatch_provider
from .profile_builder import build_incident
from .recognition import RecognitionEvent
from .recognition.mock_provider import scenario
from .sequence_manager import SignSequenceManager

EDITABLE = {"SESSION_CREATED", "COMMUNICATING", "POSSIBLE_EMERGENCY_DETECTED", "AWAITING_USER_CONFIRMATION"}


class WorkflowError(ValueError):
    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


def set_status(db: Session, session: EmergencySession, target: str, actor_type: str = "SYSTEM",
               actor_id: str | None = None, reason: str = "") -> None:
    if session.status == target:
        return
    sm.check(session.status, target)
    incident = session.incident
    db.add(StatusHistory(session_id=session.id, incident_id=incident.id if incident else None,
                         from_status=session.status, to_status=target, actor_type=actor_type, actor_id=actor_id, reason=reason))
    audit(db, "SESSION", session.id, f"STATUS_{target}", actor_type, actor_id, previous=session.status)
    session.status = target
    if incident:
        incident.status = target
    if target in sm.TERMINAL:
        session.ended_at = datetime.now(timezone.utc)


def require_editable(session: EmergencySession) -> None:
    if session.status not in EDITABLE:
        raise WorkflowError(f"the report can no longer be changed (status {sm.LABELS[session.status].lower()})")


def gesture_messages(db: Session, sequence: list[dict]) -> dict[str, str]:
    ids = [e["gesture_id"] for e in sequence if e.get("gesture_id")]
    if not ids:
        return {}
    rows = db.scalars(select(CustomGesture).where(CustomGesture.id.in_(ids)))
    return {g.intent: g.message for g in rows}


def context_of(db: Session, session: EmergencySession) -> tuple[list[dict], dict]:
    sequence = SignSequenceManager(db, session).normalized()
    ctx = evaluate(sequence, qm.answers_of(db, session.id), gesture_messages(db, sequence))
    return sequence, ctx


def effective(session: EmergencySession, ctx: dict) -> tuple[str | None, str | None]:
    """(category, summary): the user's own correction always wins over the inference."""
    category = session.user_category or ctx["top_category"]
    summary = session.user_summary or (ctx["summary"] if not session.user_category or session.user_category == ctx["top_category"]
                                       else f"You reported: {category_label(session.user_category).lower()}.")
    return category, summary


def update_context(db: Session, session: EmergencySession) -> dict:
    sequence, ctx = context_of(db, session)
    db.add(ContextEvent(session_id=session.id, interpretations=ctx["interpretations"],
                        top_category=ctx["top_category"], signals=ctx["signals"]))
    category, _ = effective(session, ctx)
    session.current_category = category
    session.general_message = ctx["general_message"]
    if session.status == "SESSION_CREATED" and sequence:
        set_status(db, session, "COMMUNICATING")
    if session.status == "COMMUNICATING" and category:
        set_status(db, session, "POSSIBLE_EMERGENCY_DETECTED", reason="rule match" if not session.user_category else "user choice")
    elif session.status == "POSSIBLE_EMERGENCY_DETECTED" and not category:
        set_status(db, session, "COMMUNICATING", reason="no emergency in the current sequence")
    qm.refresh(db, session, ctx)
    track(db, "CONTEXT_UPDATED", session.id, category=category)
    db.flush()
    return ctx


def ingest(db: Session, session: EmergencySession, events: list[RecognitionEvent]) -> dict | None:
    require_editable(session)
    if not events:
        return None
    manager = SignSequenceManager(db, session)
    for event in events:
        manager.add(event)
        track(db, "SIGN_EVENT_RECEIVED", session.id, source=event.source, kind=event.event_type)
    return update_context(db, session)


def ensure_demo_gesture(db: Session, device_key: str, spec: dict) -> CustomGesture:
    gesture = db.scalar(select(CustomGesture).where(CustomGesture.device_key == device_key, CustomGesture.name == spec["name"]))
    if gesture is None:
        cat = rules_config()["intent_categories"][spec["intent"]]
        gesture = CustomGesture(device_key=device_key, name=spec["name"], description="Demo gesture (scenario E)",
                                intent=spec["intent"], category=cat, message=spec["message"],
                                activation={"require_confirmation": True, "cooldown_s": 0})
        db.add(gesture)
        db.flush()
    return gesture


def gesture_event(session: EmergencySession, gesture: CustomGesture) -> RecognitionEvent:
    now = datetime.now(timezone.utc)
    return RecognitionEvent(event_type="GESTURE_DETECTED", session_id=session.id, sign=f"INTENT:{gesture.intent}",
                            confidence=None, start_time=now, end_time=now, source="CUSTOM_GESTURE", gesture_id=gesture.id)


def trigger_gesture(db: Session, session: EmergencySession, gesture: CustomGesture) -> dict | None:
    if not gesture.enabled:
        raise WorkflowError("this gesture is turned off")
    cooldown = int((gesture.activation or {}).get("cooldown_s", 0) or 0)
    now = datetime.now(timezone.utc)
    last = gesture.last_triggered_at
    if cooldown and last and (now - last).total_seconds() < cooldown:
        raise WorkflowError(f"wait {cooldown} s between uses of this gesture", 429)
    gesture.last_triggered_at = now
    return ingest(db, session, [gesture_event(session, gesture)])


def review(db: Session, session: EmergencySession) -> None:
    require_editable(session)
    if session.status == "AWAITING_USER_CONFIRMATION":
        return
    if session.status == "SESSION_CREATED":
        set_status(db, session, "COMMUNICATING", "USER")
    set_status(db, session, "AWAITING_USER_CONFIRMATION", "USER", reason="user opened the review")


def correct_report(db: Session, session: EmergencySession, category: str | None, summary: str | None) -> None:
    require_editable(session)
    if category is not None and category not in rules_config()["categories"]:
        raise WorkflowError("unknown category", 422)
    session.user_category = category or session.user_category
    if summary is not None:
        session.user_summary = summary.strip() or None
    audit(db, "SESSION", session.id, "USER_CORRECTED_REPORT", "USER", category=category)
    track(db, "USER_CORRECTED", session.id, category=category)
    update_context(db, session)


def confirm(db: Session, session: EmergencySession) -> EmergencyIncident:
    if session.status != "AWAITING_USER_CONFIRMATION":
        raise WorkflowError("open the review screen before confirming")
    sequence, ctx = context_of(db, session)
    category, summary = effective(session, ctx)
    if not category or not summary:
        raise WorkflowError("choose what kind of help you need before confirming", 422)
    set_status(db, session, "USER_CONFIRMED", "USER", reason="user confirmed the report")
    incident = build_incident(db, session, sequence, ctx, category, summary,
                              gesture_messages(db, sequence))
    track(db, "USER_CONFIRMED", session.id, category=category)
    track(db, "INCIDENT_PROFILE_CREATED", session.id, category=category)
    set_status(db, session, "READY_TO_SHARE", reason="incident profile prepared")
    return incident


def share(db: Session, session: EmergencySession) -> DispatchAttempt:
    incident = session.incident
    if incident is None:
        raise WorkflowError("confirm the report before sharing it")
    if session.status == "DELIVERY_FAILED":
        set_status(db, session, "READY_TO_SHARE", "USER", reason="retry")
    if session.status != "READY_TO_SHARE":
        raise WorkflowError(f"cannot share now (status {sm.LABELS[session.status].lower()})")
    settings = get_settings()
    provider = get_dispatch_provider()
    previous = db.scalar(select(DispatchAttempt).where(DispatchAttempt.incident_id == incident.id).limit(1))
    sc = scenario(session.scenario_id) if session.scenario_id else None
    # Scenario I: the first attempt fails (simulated) so the failure and retry paths can be shown
    simulate_failure = bool(sc and sc.get("simulate_dispatch_failure") and previous is None)
    track(db, "SHARE_ATTEMPTED", session.id, provider=provider.name)
    incident.sharing_status = "SENDING"
    result = provider.send({"incident_id": incident.id}, settings.dispatch_destination, simulate_failure)
    attempt = DispatchAttempt(incident_id=incident.id, provider=provider.name, destination=settings.dispatch_destination,
                              status=result.status, receipt_id=result.receipt_id, error=result.error, simulated=result.simulated)
    db.add(attempt)
    if result.status == "DELIVERED":
        incident.sharing_status = "DELIVERED"
        if incident.video_storage_key:
            incident.video_sharing_status = "DELIVERED"  # same in-app destination as the report
        set_status(db, session, "SENT_TO_CONFIGURED_DESTINATION", reason=f"delivered to {settings.dispatch_destination}")
        track(db, "SHARE_SUCCEEDED", session.id, provider=provider.name)
    else:
        incident.sharing_status = "FAILED"
        if incident.video_storage_key:
            incident.video_sharing_status = "FAILED"
        set_status(db, session, "DELIVERY_FAILED", reason=result.error or "delivery failed")
        track(db, "SHARE_FAILED", session.id, provider=provider.name)
    db.flush()
    return attempt


def cancel(db: Session, session: EmergencySession) -> None:
    if "CANCELLED" not in sm.TRANSITIONS.get(session.status, set()):
        raise WorkflowError("this report has already been shared and cannot be cancelled here")
    set_status(db, session, "CANCELLED", "USER", reason="cancelled by the user")
    track(db, "SESSION_CANCELLED", session.id)
