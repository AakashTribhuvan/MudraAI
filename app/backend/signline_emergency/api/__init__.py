"""Mounting: `app.include_router(emergency_router())` (see mudra_api/app.py)."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..config import get_settings
from ..db import run_migrations, session_factory
from .router import build_router

log = logging.getLogger("signline_emergency")


def emergency_router() -> APIRouter:
    """Migrate the database, make sure the demo responder exists (demo mode) and return the router."""
    settings = get_settings()
    if settings.run_migrations:
        run_migrations()
    if settings.demo_mode:
        from ..seed import ensure_demo_responder

        with session_factory()() as db:
            ensure_demo_responder(db)
    from ..retention import start_scheduler

    start_scheduler(settings.retention_interval_h)  # retention now, then every N hours
    log.info("SignLine Emergency mounted (demo mode: %s, recognition: %s)", settings.demo_mode, settings.recognition_provider)
    return build_router()
