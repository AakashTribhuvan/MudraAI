"""Data retention.

- Ended sessions (and their incidents, events, answers, the incident's video copy) older than the retention
  period (``SIGNLINE_EM_RETENTION_DAYS``) and old analytics events are deleted.
- A personal gesture's demonstration video is deleted ``SIGNLINE_EM_GESTURE_VIDEO_RETENTION_DAYS`` after it was
  recorded (the gesture itself stays; 0 keeps videos until the user deletes them).
- Video files that no gesture or incident refers to (e.g. left by a crash) are deleted after an hour.

Runs automatically at server start and every ``SIGNLINE_EM_RETENTION_INTERVAL_H`` hours (``start_scheduler``),
or by hand: python -m signline_emergency.retention [--days N] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import session_factory
from .models import AnalyticsEvent, CustomGesture, EmergencyIncident, EmergencySession
from .services.media_store import get_media_store

log = logging.getLogger("signline_emergency")
ORPHAN_GRACE_S = 3600


def purge(db: Session, days: int, dry_run: bool = False, gesture_video_days: int | None = None, store=None) -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    store = store or get_media_store()
    old = list(db.scalars(select(EmergencySession).where(EmergencySession.ended_at.is_not(None),
                                                         EmergencySession.ended_at < cutoff)))
    analytics = db.scalars(select(AnalyticsEvent.id).where(AnalyticsEvent.created_at < cutoff)).all()
    incident_videos = [s.incident.video_storage_key for s in old if s.incident and s.incident.video_storage_key]

    gesture_video_days = get_settings().gesture_video_retention_days if gesture_video_days is None else gesture_video_days
    expired = []
    if gesture_video_days > 0:
        video_cutoff = now - timedelta(days=gesture_video_days)
        expired = list(db.scalars(select(CustomGesture).where(CustomGesture.video_storage_key.is_not(None),
                                                              CustomGesture.recording_consent_at < video_cutoff)))
    if not dry_run:
        for s in old:
            db.delete(s)  # foreign keys cascade to events, questions, incident, facts, history
        db.execute(delete(AnalyticsEvent).where(AnalyticsEvent.created_at < cutoff))
        for g in expired:
            store.delete(g.video_storage_key)
            g.video_storage_key = g.video_mime_type = g.video_duration_seconds = g.video_size_bytes = None
        db.commit()
        for key in incident_videos:
            store.delete(key)

    referenced = set(db.scalars(select(CustomGesture.video_storage_key).where(CustomGesture.video_storage_key.is_not(None))))
    referenced |= set(db.scalars(select(EmergencyIncident.video_storage_key).where(EmergencyIncident.video_storage_key.is_not(None))))
    orphans = []
    for key in store.keys():
        if key in referenced:
            continue
        try:
            age = time.time() - (store.root / key).stat().st_mtime
        except OSError:
            continue
        if age > ORPHAN_GRACE_S:
            orphans.append(key)
            if not dry_run:
                store.delete(key)
    return {"sessions": len(old), "analytics_events": len(analytics), "incident_videos": len(incident_videos),
            "gesture_videos": len(expired), "orphan_videos": len(orphans), "cutoff": cutoff.isoformat(), "dry_run": dry_run}


_scheduler: threading.Thread | None = None
_stop = threading.Event()


def start_scheduler(interval_h: float) -> None:
    """Run ``purge`` now and then every ``interval_h`` hours in a daemon thread (once per process)."""
    global _scheduler
    if interval_h <= 0 or (_scheduler and _scheduler.is_alive()):
        return

    def loop() -> None:
        while not _stop.is_set():
            try:
                with session_factory()() as db:
                    counts = purge(db, get_settings().retention_days)
                log.info("Retention: %s", {k: v for k, v in counts.items() if k != "cutoff"})  # counts only, no content
            except Exception:  # noqa: BLE001 - keep the scheduler alive; the next run tries again
                log.exception("Retention run failed")
            _stop.wait(interval_h * 3600)

    _stop.clear()
    _scheduler = threading.Thread(target=loop, name="signline-retention", daemon=True)
    _scheduler.start()


def stop_scheduler() -> None:
    _stop.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=get_settings().retention_days)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with session_factory()() as db:
        print(purge(db, args.days, args.dry_run))


if __name__ == "__main__":
    main()
