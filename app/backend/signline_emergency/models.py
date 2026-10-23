"""SQLAlchemy models. Provenance is a column, not prose: every fact says where it came from.

Provenance values: USER_REPORTED (communicated by the user), AI_INFERRED (system interpretation),
USER_CONFIRMED (explicitly confirmed by the user), SYSTEM_GENERATED (ids, times, workflow statuses).
No raw video or camera images are stored anywhere.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every database (SQLite returns naive values otherwise)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class User(Base):
    """Responder / admin accounts (emergency users stay anonymous: they are identified per device)."""

    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("usr"))
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(20))  # RESPONDER | ADMIN
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class UserPreference(Base):
    """Communication needs the user configured themselves (never assumed)."""

    __tablename__ = "user_preferences"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    communication_needs: Mapped[list] = mapped_column(JSON, default=list)
    preferred_language: Mapped[str] = mapped_column(String(10), default="en")
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class EmergencySession(Base):
    __tablename__ = "emergency_sessions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ses"))
    device_key: Mapped[str] = mapped_column(String(64), index=True)
    access_token_hash: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(40), default="SESSION_CREATED", index=True)
    mode: Mapped[str] = mapped_column(String(20), default="SIGN")  # SIGN | GESTURE | BUTTONS
    demo: Mapped[bool] = mapped_column(Boolean, default=True)
    # mock-provider playback state (deterministic scenarios)
    scenario_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    scenario_cursor: Mapped[int] = mapped_column(Integer, default=0)
    scenario_paused: Mapped[bool] = mapped_column(Boolean, default=False)
    location: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {lat, lon, accuracy_m, text, source, permission, simulated}
    current_category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # the user's own correction always wins over the inferred category / summary
    user_category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    user_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    general_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    events: Mapped[list[SignEvent]] = relationship(back_populates="session", order_by="SignEvent.seq_index", cascade="all, delete-orphan")
    incident: Mapped[EmergencyIncident | None] = relationship(back_populates="session", uselist=False,
                                                              cascade="all, delete-orphan")  # retention deletes both


class SignEvent(Base):
    """One recognized sign, gesture or button selection, exactly as received (the raw sequence)."""

    __tablename__ = "sign_events"
    __table_args__ = (UniqueConstraint("session_id", "seq_index", name="uq_sign_event_order"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("emergency_sessions.id", ondelete="CASCADE"), index=True)
    seq_index: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(30))  # SIGN_DETECTED | UNKNOWN_SIGN | GESTURE_DETECTED | BUTTON_SELECTED
    sign: Mapped[str] = mapped_column(String(80))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_time: Mapped[datetime] = mapped_column(UTCDateTime)
    end_time: Mapped[datetime] = mapped_column(UTCDateTime)
    source: Mapped[str] = mapped_column(String(30))  # MOCK_PROVIDER | REAL_MODEL | BUTTONS | CUSTOM_GESTURE
    gesture_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    session: Mapped[EmergencySession] = relationship(back_populates="events")


class SequenceRevision(Base):
    """Audit trail of the normalized sequence: every add, correction, removal, undo and reset."""

    __tablename__ = "sequence_revisions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("emergency_sessions.id", ondelete="CASCADE"), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(20))  # ADD | CORRECT | REMOVE | UNDO | RESET
    event_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    normalized: Mapped[list] = mapped_column(JSON)  # the full normalized sequence after this revision
    actor: Mapped[str] = mapped_column(String(20))  # USER | SYSTEM
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class CustomGesture(Base):
    """A personal gesture mapped to an intent or phrase. A mapping, not a trained recognizer."""

    __tablename__ = "custom_gestures"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("gst"))
    device_key: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(Text, default="")
    image_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    intent: Mapped[str] = mapped_column(String(40))
    category: Mapped[str] = mapped_column(String(40))
    message: Mapped[str] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    activation: Mapped[dict] = mapped_column(JSON, default=dict)  # {require_confirmation, cooldown_s}
    last_triggered_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    # optional demonstration video (services/media_store.py: a random storage key, never the user's file name)
    video_storage_key: Mapped[str | None] = mapped_column(String(80), nullable=True)
    video_mime_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    video_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recording_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    recording_consent_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    emergency_contact_id: Mapped[str | None] = mapped_column(String(40), nullable=True)  # config/emergency_contacts.json
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class ContextEvent(Base):
    """One evaluation of the context engine (all interpretations, ranked)."""

    __tablename__ = "context_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("emergency_sessions.id", ondelete="CASCADE"), index=True)
    interpretations: Mapped[list] = mapped_column(JSON)
    top_category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    signals: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (Index("ix_questions_session_key", "session_id", "key"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("q"))
    session_id: Mapped[str] = mapped_column(ForeignKey("emergency_sessions.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(40))
    text: Mapped[str] = mapped_column(Text)
    answer_type: Mapped[str] = mapped_column(String(30))
    options: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")  # PENDING | ANSWERED | SKIPPED | SUPERSEDED
    priority: Mapped[int] = mapped_column(Integer, default=50)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class QuestionAnswer(Base):
    __tablename__ = "question_answers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("emergency_sessions.id", ondelete="CASCADE"), index=True)
    value: Mapped[object] = mapped_column(JSON)
    provenance: Mapped[str] = mapped_column(String(20), default="USER_REPORTED")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class EmergencyIncident(Base):
    __tablename__ = "emergency_incidents"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("inc"))
    session_id: Mapped[str] = mapped_column(ForeignKey("emergency_sessions.id", ondelete="CASCADE"), unique=True)
    category: Mapped[str] = mapped_column(String(40), index=True)
    summary: Mapped[str] = mapped_column(Text)
    user_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(40), index=True)
    sharing_status: Mapped[str] = mapped_column(String(20), default="NOT_SHARED")  # NOT_SHARED | SENDING | DELIVERED | FAILED
    communication_needs: Mapped[list] = mapped_column(JSON, default=list)
    location: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    priority: Mapped[str | None] = mapped_column(String(10), nullable=True)  # set only by explicit rules
    demo: Mapped[bool] = mapped_column(Boolean, default=True)
    # prepared from a personal gesture (POST /gestures/{id}/request-assistance)
    gesture_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    emergency_contact_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    emergency_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # the incident's own encrypted copy of the gesture video, only with the user's sharing consent
    video_storage_key: Mapped[str | None] = mapped_column(String(80), nullable=True)
    video_mime_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    video_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharing_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    sharing_consent_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    video_sharing_status: Mapped[str] = mapped_column(String(20), default="NOT_ATTACHED")  # NOT_ATTACHED | ATTACHED | DELIVERED | FAILED
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    session: Mapped[EmergencySession] = relationship(back_populates="incident")
    facts: Mapped[list[IncidentFact]] = relationship(order_by="IncidentFact.id", cascade="all, delete-orphan")


class IncidentFact(Base):
    __tablename__ = "incident_facts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("emergency_incidents.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(30))  # SIGN | SELECTION | ANSWER | SUMMARY | CATEGORY | MESSAGE | UNCERTAINTY | LOCATION
    key: Mapped[str] = mapped_column(String(80))
    value: Mapped[str] = mapped_column(Text)
    provenance: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class StatusHistory(Base):
    __tablename__ = "incident_status_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("emergency_sessions.id", ondelete="CASCADE"), index=True)
    incident_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    from_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    to_status: Mapped[str] = mapped_column(String(40))
    actor_type: Mapped[str] = mapped_column(String(20))  # USER | RESPONDER | SYSTEM
    actor_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class DispatchAttempt(Base):
    __tablename__ = "dispatch_attempts"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("dsp"))
    incident_id: Mapped[str] = mapped_column(ForeignKey("emergency_incidents.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(40))
    destination: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20))  # PREPARED | SENT | DELIVERED | FAILED
    receipt_id: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    simulated: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class ResponderNote(Base):
    __tablename__ = "responder_notes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("emergency_incidents.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class ResponderAction(Base):
    __tablename__ = "responder_actions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("emergency_incidents.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(30))  # VIEW | ACKNOWLEDGE | NOTE | STATUS | RESOLVE
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_entity", "entity_type", "entity_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(30))
    entity_id: Mapped[str] = mapped_column(String(40))
    action: Mapped[str] = mapped_column(String(40))
    actor_type: Mapped[str] = mapped_column(String(20))
    actor_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)  # non-sensitive only
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class AnalyticsEvent(Base):
    """Privacy-first usage events: type, time, session id and non-sensitive metadata only."""

    __tablename__ = "analytics_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    session_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    demo: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
