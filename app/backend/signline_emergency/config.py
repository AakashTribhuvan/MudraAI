"""Configuration from environment variables (see app/backend/.env.example)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
VAR_DIR = BACKEND_DIR / "var"  # local database + generated demo credentials (git-ignored)
CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class EmergencySettings:
    # SQLite for the prototype; any SQLAlchemy URL works (e.g. postgresql+psycopg://...)
    database_url: str = field(default_factory=lambda: os.getenv(
        "SIGNLINE_EM_DATABASE_URL", f"sqlite:///{(VAR_DIR / 'signline_emergency.db').as_posix()}"))
    # Signs responder tokens. Unset: a random key per server start (tokens end on restart).
    secret_key: str = field(default_factory=lambda: os.getenv("SIGNLINE_EM_SECRET_KEY") or secrets.token_urlsafe(32))
    token_ttl_s: int = field(default_factory=lambda: int(os.getenv("SIGNLINE_EM_TOKEN_TTL_S", "28800")))
    # Demo mode: mock recognition, mock location and mock dispatch are labelled as such everywhere
    demo_mode: bool = field(default_factory=lambda: _bool("SIGNLINE_EM_DEMO_MODE", True))
    recognition_provider: str = field(default_factory=lambda: os.getenv("SIGNLINE_EM_RECOGNITION_PROVIDER", "mock"))
    dispatch_provider: str = field(default_factory=lambda: os.getenv("SIGNLINE_EM_DISPATCH_PROVIDER", "mock"))
    # Where the mock dispatch provider "delivers" (a test desk inside this app, never a real service)
    dispatch_destination: str = field(default_factory=lambda: os.getenv("SIGNLINE_EM_DISPATCH_DESTINATION", "signline-demo-responder-desk"))
    # Days after which ended sessions and incidents may be purged (`python -m signline_emergency.retention`)
    retention_days: int = field(default_factory=lambda: int(os.getenv("SIGNLINE_EM_RETENTION_DAYS", "30")))
    # Requests per minute for sensitive endpoints (login, session creation, sharing)
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("SIGNLINE_EM_RATE_LIMIT_PER_MIN", "30")))
    run_migrations: bool = field(default_factory=lambda: _bool("SIGNLINE_EM_RUN_MIGRATIONS", True))
    # Gesture demonstration videos (services/media_store.py); git-ignored like the database
    media_dir: str = field(default_factory=lambda: os.getenv("SIGNLINE_EM_MEDIA_DIR", str(VAR_DIR / "emergency_media")))
    # AES-256-GCM key for those videos: base64 in SIGNLINE_EM_MEDIA_KEY, else generated once into this file
    media_key: str | None = field(default_factory=lambda: os.getenv("SIGNLINE_EM_MEDIA_KEY") or None)
    media_key_file: str = field(default_factory=lambda: os.getenv("SIGNLINE_EM_MEDIA_KEY_FILE", str(VAR_DIR / "emergency_media.key")))
    # A gesture's video is deleted this many days after it was recorded (0 = keep until the user deletes it)
    gesture_video_retention_days: int = field(default_factory=lambda: int(os.getenv("SIGNLINE_EM_GESTURE_VIDEO_RETENTION_DAYS", "365")))
    # The retention job runs at start-up and then every this many hours (0 = only when run by hand)
    retention_interval_h: float = field(default_factory=lambda: float(os.getenv("SIGNLINE_EM_RETENTION_INTERVAL_H", "24")))


_settings: EmergencySettings | None = None


def get_settings() -> EmergencySettings:
    global _settings
    if _settings is None:
        _settings = EmergencySettings()
    return _settings


def set_settings(settings: EmergencySettings) -> None:
    """Tests: replace the settings (call before the app is created)."""
    global _settings
    _settings = settings
