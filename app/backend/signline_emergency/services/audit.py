"""Audit trail and privacy-first analytics.

Analytics keep only the event type, time, session id and a few non-sensitive fields (category, counts,
source). Never signs, messages, summaries, notes or locations.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import AnalyticsEvent, AuditEvent

log = logging.getLogger("signline_emergency")

ANALYTICS_TYPES = {
    "SESSION_STARTED", "SESSION_ENDED", "SIGN_EVENT_RECEIVED", "SIGN_CORRECTED", "CONTEXT_UPDATED",
    "QUESTION_CREATED", "QUESTION_ANSWERED", "QUESTION_SKIPPED", "USER_CONFIRMED", "USER_CORRECTED",
    "INCIDENT_PROFILE_CREATED", "SHARE_ATTEMPTED", "SHARE_SUCCEEDED", "SHARE_FAILED",
    "RESPONDER_ACKNOWLEDGED", "INCIDENT_RESOLVED", "GESTURE_TESTED", "SESSION_CANCELLED",
    "GESTURE_VIDEO_SAVED", "GESTURE_REPORT_PREPARED",
}
SAFE_META_KEYS = {"category", "source", "count", "mode", "answer_type", "status", "provider", "scenario_id", "kind"}


def audit(db: Session, entity_type: str, entity_id: str, action: str, actor_type: str,
          actor_id: str | None = None, **detail) -> None:
    db.add(AuditEvent(entity_type=entity_type, entity_id=entity_id, action=action, actor_type=actor_type,
                      actor_id=actor_id, detail=detail))


def track(db: Session, event_type: str, session_id: str | None = None, **meta) -> None:
    if event_type not in ANALYTICS_TYPES:
        raise ValueError(f"unknown analytics event {event_type}")
    safe = {k: v for k, v in meta.items() if k in SAFE_META_KEYS}
    db.add(AnalyticsEvent(event_type=event_type, session_id=session_id, meta=safe, demo=get_settings().demo_mode))
    log.info("analytics %s", event_type)  # routine logs carry no sign sequence, message or location
