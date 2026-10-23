"""Gesture reports through the responder desk, video encryption at rest, and retention of videos."""

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from signline_emergency.config import get_settings
from signline_emergency.db import session_factory
from signline_emergency.models import CustomGesture, EmergencySession
from signline_emergency.retention import purge
from signline_emergency.services import media_store
from signline_emergency.services.media_store import get_media_store

from test_gesture_video import API, WEBM, make_gesture, stored_files, upload


def assist(client, device, gid, **body):
    return client.post(f"{API}/gestures/{gid}/request-assistance", headers=device, json=body)


def test_report_is_prepared_not_shared_then_delivered_and_acknowledged(client, device, responder_headers):
    g = make_gesture(client, device, emergency_contact_id="in_108")
    r = assist(client, device, g["id"], location={"source": "MANUAL", "text": "Platform 2, Pune station"})
    assert r.status_code == 200, r.text
    view, token = r.json()["session"], r.json()["access_token"]
    assert view["status"] == "READY_TO_SHARE"  # prepared and confirmed, nothing shared yet
    inc = view["incident"]
    assert inc["report_sharing_status"] == "NOT_SHARED" and inc["delivery_receipt"] is None and inc["video"] is None
    assert inc["emergency_contact"]["number"] == "108" and "Platform 2, Pune station" in inc["emergency_message"]
    assert client.get(f"{API}/responder/incidents/{inc['id']}", headers=responder_headers).status_code == 404  # not shared
    h = {"X-Session-Token": token}
    shared = client.post(f"{API}/sessions/{view['id']}/share", headers=h).json()
    assert shared["status"] == "SENT_TO_CONFIGURED_DESTINATION"
    assert shared["incident"]["report_sharing_status"] == "DELIVERED" and shared["incident"]["delivery_receipt"].startswith("SIM-")
    seen = client.get(f"{API}/responder/incidents/{inc['id']}", headers=responder_headers).json()
    assert seen["gesture_id"] == g["id"] and seen["video"] is None
    client.post(f"{API}/responder/incidents/{inc['id']}/acknowledge", headers=responder_headers)
    assert client.get(f"{API}/sessions/{view['id']}", headers=h).json()["status"] == "ACKNOWLEDGED"


def test_video_goes_to_the_responder_only_with_sharing_consent(client, device, responder_headers):
    g = make_gesture(client, device)
    assert assist(client, device, g["id"], include_video=True, sharing_consent=True).status_code == 422  # no video yet
    upload(client, device, g["id"])
    assert assist(client, device, g["id"], include_video=True).status_code == 422  # consent missing
    r = assist(client, device, g["id"], include_video=True, sharing_consent=True).json()
    inc = r["session"]["incident"]
    assert inc["video_sharing_status"] == "ATTACHED" and inc["video"]["sharing_consent"] and inc["video"]["sharing_consent_at"]
    assert client.get(f"{API}/responder/incidents/{inc['id']}/video", headers=responder_headers).status_code == 404  # not delivered
    shared = client.post(f"{API}/sessions/{r['session']['id']}/share", headers={"X-Session-Token": r["access_token"]}).json()
    assert shared["incident"]["video_sharing_status"] == "DELIVERED"
    v = client.get(f"{API}/responder/incidents/{inc['id']}/video", headers=responder_headers)
    assert v.status_code == 200 and v.content == WEBM and v.headers["cache-control"] == "no-store"
    assert client.get(f"{API}/responder/incidents/{inc['id']}/video").status_code == 401
    # the incident keeps its own copy: deleting the gesture does not remove what was shared
    client.delete(f"{API}/gestures/{g['id']}", headers=device)
    assert client.get(f"{API}/responder/incidents/{inc['id']}/video", headers=responder_headers).content == WEBM


def test_report_without_video_consent_has_no_video_for_responders(client, device, responder_headers):
    g = make_gesture(client, device)
    upload(client, device, g["id"])
    r = assist(client, device, g["id"]).json()
    client.post(f"{API}/sessions/{r['session']['id']}/share", headers={"X-Session-Token": r["access_token"]})
    inc_id = r["session"]["incident"]["id"]
    assert client.get(f"{API}/responder/incidents/{inc_id}", headers=responder_headers).json()["video"] is None
    assert client.get(f"{API}/responder/incidents/{inc_id}/video", headers=responder_headers).status_code == 404


def test_failed_delivery_is_reported_as_failed(client, device, monkeypatch):
    from signline_emergency.services import dispatch

    g = make_gesture(client, device)
    upload(client, device, g["id"])
    r = assist(client, device, g["id"], include_video=True, sharing_consent=True).json()
    real = dispatch.get_dispatch_provider()

    class Down:
        name = real.name

        def send(self, payload, destination, simulate_failure=False):
            return real.send(payload, destination, True)
    monkeypatch.setattr("signline_emergency.services.workflow.get_dispatch_provider", lambda: Down())
    out = client.post(f"{API}/sessions/{r['session']['id']}/share", headers={"X-Session-Token": r["access_token"]}).json()
    assert out["status"] == "DELIVERY_FAILED"
    assert out["incident"]["report_sharing_status"] == "FAILED" and out["incident"]["video_sharing_status"] == "FAILED"
    assert out["incident"]["delivery_receipt"] is None


def test_turned_off_gesture_and_unknown_contact_are_refused(client, device):
    g = make_gesture(client, device, enabled=False)
    assert assist(client, device, g["id"]).status_code == 409
    g2 = make_gesture(client, device)
    assert assist(client, device, g2["id"], contact_id="nope").status_code == 422


def test_videos_are_encrypted_at_rest_and_tampering_is_detected(client, device):
    g = make_gesture(client, device)
    upload(client, device, g["id"])
    store = get_media_store()
    raw = (store.root / stored_files()[0]).read_bytes()
    assert raw.startswith(media_store.MAGIC) and WEBM[:64] not in raw  # not readable on disk
    assert client.get(f"{API}/gestures/{g['id']}/video", headers=device).content == WEBM
    damaged = bytearray(raw)
    damaged[-1] ^= 1
    (store.root / stored_files()[0]).write_bytes(bytes(damaged))
    assert client.get(f"{API}/gestures/{g['id']}/video", headers=device).status_code == 503
    other = media_store.LocalMediaStore(store.root, b"\x01" * 32)  # a different key cannot read it
    with pytest.raises(media_store.MediaStorageError):
        other.read(stored_files()[0])


def test_media_key_from_environment_or_file(tmp_path):
    k = media_store.load_key(None, tmp_path / "k.key")
    assert len(k) == 32 and media_store.load_key(None, tmp_path / "k.key") == k  # created once, then reused
    import base64
    env = base64.urlsafe_b64encode(b"\x02" * 32).decode()
    assert media_store.load_key(env, tmp_path / "unused.key") == b"\x02" * 32
    with pytest.raises(media_store.MediaStorageError):
        media_store.load_key(base64.urlsafe_b64encode(b"short").decode(), tmp_path / "x.key")


def test_retention_deletes_expired_videos_old_incident_copies_and_orphans(client, device):
    g = make_gesture(client, device)
    upload(client, device, g["id"])
    r = assist(client, device, g["id"], include_video=True, sharing_consent=True).json()
    assert len(stored_files()) == 2  # the gesture video and the incident's copy
    store = get_media_store()
    orphan = store.save(WEBM, "video/webm")
    old = time.time() - 2 * 3600
    os.utime(store.root / orphan, (old, old))
    with session_factory()() as db:
        gesture = db.get(CustomGesture, g["id"])
        gesture.recording_consent_at = datetime.now(timezone.utc) - timedelta(days=400)
        session = db.get(EmergencySession, r["session"]["id"])
        session.ended_at = datetime.now(timezone.utc) - timedelta(days=60)
        db.commit()
        dry = purge(db, days=30, dry_run=True, gesture_video_days=365)
        assert (dry["sessions"], dry["incident_videos"], dry["gesture_videos"], dry["orphan_videos"]) == (1, 1, 1, 1)
        assert len(stored_files()) == 3  # dry run deletes nothing
        out = purge(db, days=30, gesture_video_days=365)
    assert (out["incident_videos"], out["gesture_videos"], out["orphan_videos"]) == (1, 1, 1)
    assert stored_files() == []
    left = client.get(f"{API}/gestures", headers=device).json()[0]
    assert left["id"] == g["id"] and left["has_video"] is False  # the gesture stays, its video is gone


def test_retention_keeps_recent_videos(client, device):
    g = make_gesture(client, device)
    upload(client, device, g["id"])
    with session_factory()() as db:
        out = purge(db, days=30, gesture_video_days=365)
    assert out["gesture_videos"] == 0 and out["orphan_videos"] == 0 and len(stored_files()) == 1
    assert get_settings().retention_interval_h == 0  # tests never start the background scheduler
