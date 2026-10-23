"""Gesture demonstration videos, emergency contacts and the gesture report (nothing is dialled or sent)."""

import pytest

from signline_emergency.config import get_settings
from signline_emergency.services import media_store
from signline_emergency.services.gesture_report import tel_link

API = "/api/v1/emergency"
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 2048  # EBML header: a WebM container
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 2048


def make_gesture(client, device, **extra):
    r = client.post(f"{API}/gestures", headers=device, json={
        "name": "My Medical SOS", "description": "Fist on the chest, twice", "intent": "REQUEST_MEDICAL_HELP",
        "message": "I need urgent medical help.", **extra})
    assert r.status_code == 200, r.text
    return r.json()


def upload(client, device, gid, data=WEBM, mime="video/webm", duration="4.2", consent="true"):
    return client.put(f"{API}/gestures/{gid}/video", headers=device,
                      files={"file": ("clip.webm", data, mime)}, data={"duration_s": duration, "recording_consent": consent})


def stored_files():
    root = media_store.Path(get_settings().media_dir)
    return sorted(p.name for p in root.glob("*")) if root.exists() else []


def test_gesture_without_video_then_with_video_retake_and_delete(client, device):
    g = make_gesture(client, device)
    assert g["has_video"] is False and g["recording_consent"] is False and g["created_at"]
    r = upload(client, device, g["id"])
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["has_video"] and v["video_mime_type"] == "video/webm" and v["video_duration_seconds"] == 4.2
    assert v["recording_consent"] and v["recording_consent_at"]
    first = stored_files()
    assert len(first) == 1 and first[0].startswith("gv_") and "clip" not in first[0]  # random key, not the file name
    got = client.get(f"{API}/gestures/{g['id']}/video", headers=device)
    assert got.status_code == 200 and got.content == WEBM and got.headers["content-type"].startswith("video/webm")
    # retake replaces the file
    assert upload(client, device, g["id"], data=MP4, mime="video/mp4").status_code == 200
    assert len(stored_files()) == 1 and stored_files() != first
    # deleting the gesture deletes its recording
    d = client.delete(f"{API}/gestures/{g['id']}", headers=device).json()
    assert d["video_deleted"] and stored_files() == []


def test_video_needs_consent_and_a_real_short_video(client, device):
    gid = make_gesture(client, device)["id"]
    assert upload(client, device, gid, consent="false").status_code == 422
    assert upload(client, device, gid, mime="image/png").status_code == 422
    assert upload(client, device, gid, data=b"not a video at all").status_code == 422  # declared webm, content is not
    assert upload(client, device, gid, data=MP4, mime="video/webm").status_code == 422  # type mismatch
    assert upload(client, device, gid, duration="45").status_code == 422
    assert upload(client, device, gid, duration="0.1").status_code == 422
    big = b"\x1a\x45\xdf\xa3" + b"\x00" * media_store.MAX_VIDEO_BYTES
    assert upload(client, device, gid, data=big).status_code == 422
    assert stored_files() == []


def test_only_the_owner_can_see_or_change_the_video(client, device):
    gid = make_gesture(client, device)["id"]
    assert upload(client, device, gid).status_code == 200
    other = {"X-Device-Key": "test-device-someone-else"}
    assert client.get(f"{API}/gestures/{gid}/video", headers=other).status_code == 404
    assert client.get(f"{API}/gestures/{gid}/video").status_code == 401
    assert upload(client, other, gid).status_code == 404
    assert client.delete(f"{API}/gestures/{gid}/video", headers=other).status_code == 404
    assert client.delete(f"{API}/gestures/{gid}/video", headers=device).json()["has_video"] is False
    assert client.get(f"{API}/gestures/{gid}/video", headers=device).status_code == 404


def test_storage_failure_is_reported_not_hidden(client, device, monkeypatch):
    gid = make_gesture(client, device)["id"]

    def broken(self, data, mime):
        raise media_store.MediaStorageError("the video could not be stored")
    monkeypatch.setattr(media_store.LocalMediaStore, "save", broken)
    r = upload(client, device, gid)
    assert r.status_code == 503
    assert client.get(f"{API}/gestures", headers=device).json()[0]["has_video"] is False


def test_contacts_directory_defaults_to_112_and_nothing_is_marked_verified(client):
    c = client.get(f"{API}/contacts").json()
    numbers = {x["number"] for x in c["contacts"]}
    assert c["region"] == "IN" and c["default_contact_id"] == "in_112"
    assert {"112", "100", "101", "102", "108", "103", "1073", "1033", "181", "1098", "1930", "1070", "139"} == numbers
    assert not any(x["verified"] for x in c["contacts"])  # until checked against an authoritative source
    assert client.get(f"{API}/contacts", params={"region": "XX"}).status_code == 404


def test_report_uses_saved_details_only_and_changes_destination(client, device):
    g = make_gesture(client, device, emergency_contact_id="in_108")
    assert client.patch(f"{API}/gestures/{g['id']}", headers=device, json={"emergency_contact_id": "nope"}).status_code == 422
    r = client.post(f"{API}/gestures/{g['id']}/report", headers=device, json={}).json()
    assert r["contact"]["number"] == "108" and r["dial_link"] == "tel:108" and r["sent"] is False
    text = r["text"]
    assert text.startswith("EMERGENCY ASSISTANCE REQUESTED\n\n")
    assert "Emergency type: Medical help" in text and "User-reported message: I need urgent medical help." in text
    assert "Gesture: My Medical SOS" in text and "Location" not in text and "Video" not in text  # nothing invented
    prov = {line["label"]: line["provenance"] for line in r["lines"]}
    assert prov["User-reported message"] == "USER_REPORTED" and prov["Communication requirement"] == "SYSTEM_GENERATED"
    # the user changes the destination and adds a place; a video appears only when one is saved
    upload(client, device, g["id"])
    r2 = client.post(f"{API}/gestures/{g['id']}/report", headers=device,
                     json={"contact_id": "in_112", "location_text": "Platform 2, Pune station"}).json()
    assert r2["dial_link"] == "tel:112" and r2["has_video"]
    assert "Location: Platform 2, Pune station" in r2["text"] and "Video: A recorded gesture" in r2["text"]
    # without a saved destination the report goes to 112
    plain = make_gesture(client, device)
    assert client.post(f"{API}/gestures/{plain['id']}/report", headers=device, json={}).json()["dial_link"] == "tel:112"


def test_communication_needs_are_user_reported(client, device):
    client.put(f"{API}/preferences", headers=device, json={"communication_needs": ["Prefers text communication"]})
    g = make_gesture(client, device)
    r = client.post(f"{API}/gestures/{g['id']}/report", headers=device, json={}).json()
    line = next(x for x in r["lines"] if x["label"] == "Communication requirement")
    assert line["text"] == "Prefers text communication" and line["provenance"] == "USER_REPORTED"


@pytest.mark.parametrize("raw, link", [("112", "tel:112"), (" 1 0 8 ", "tel:108"), ("+91-1800-11-0000", "tel:+911800110000")])
def test_dial_links_contain_digits_only(raw, link):
    assert tel_link(raw) == link


@pytest.mark.parametrize("raw", ["", "abc", "1", "112;evil", "javascript:alert(1)"])
def test_dial_links_reject_non_numbers(raw):
    if any(ch.isdigit() for ch in raw) and len([c for c in raw if c.isdigit()]) >= 2:
        assert tel_link(raw).startswith("tel:") and ";" not in tel_link(raw)  # junk around digits is stripped
    else:
        with pytest.raises(ValueError):
            tel_link(raw)


def test_video_details_never_reach_analytics(client, device):
    from sqlalchemy import select

    from signline_emergency.db import session_factory
    from signline_emergency.models import AnalyticsEvent
    gid = make_gesture(client, device)["id"]
    upload(client, device, gid)
    client.post(f"{API}/gestures/{gid}/report", headers=device, json={"location_text": "Secret street 5"})
    with session_factory()() as db:
        rows = db.scalars(select(AnalyticsEvent)).all()
        blob = repr([(e.event_type, e.meta) for e in rows])
    assert "gv_" not in blob and "Secret street" not in blob and "urgent medical" not in blob
