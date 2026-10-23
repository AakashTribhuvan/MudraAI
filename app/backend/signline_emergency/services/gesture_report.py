"""Emergency contacts (config/emergency_contacts.json) and the report prepared from a personal gesture.

The report repeats only what the user saved or typed; every line says where it came from (USER_REPORTED,
SYSTEM_GENERATED). Nothing here sends, dials or shares anything: the browser opens the phone's dialer
(``tel:``) or share sheet only after the user confirms, and the app cannot know whether a call connected.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from ..config import CONFIG_DIR
from .context_engine import category_label

DEFAULT_COMMUNICATION = "Please communicate using text or visual methods."


@lru_cache(maxsize=1)
def contacts_config() -> dict:
    return json.loads((CONFIG_DIR / "emergency_contacts.json").read_text(encoding="utf-8"))


def region_contacts(region: str | None = None) -> tuple[str, list[dict]]:
    cfg = contacts_config()
    region = region or cfg["default_region"]
    if region not in cfg["regions"]:
        raise KeyError(region)
    return region, cfg["regions"][region]["contacts"]


def find_contact(contact_id: str | None) -> dict | None:
    for region in contacts_config()["regions"].values():
        for c in region["contacts"]:
            if c["id"] == contact_id:
                return c
    return None


def default_contact() -> dict:
    return find_contact(contacts_config()["default_contact_id"])


def tel_link(number: str) -> str:
    """A dialer link with digits only (and a leading +), so nothing else can be injected into it."""
    digits = re.sub(r"[^\d+]", "", number or "")
    digits = digits[0] + digits[1:].replace("+", "") if digits else ""
    if not re.fullmatch(r"\+?\d{2,15}", digits):
        raise ValueError("not a phone number")
    return f"tel:{digits}"


def build_report(gesture, contact: dict, communication_needs: list[str], location_text: str | None) -> dict:
    """The message for the dial / share screen, line by line with provenance, plus the plain text."""
    lines = [
        {"label": None, "text": "EMERGENCY ASSISTANCE REQUESTED", "provenance": "SYSTEM_GENERATED"},
        {"label": "Emergency type", "text": category_label(gesture.category), "provenance": "USER_REPORTED"},
        {"label": "User-reported message", "text": gesture.message, "provenance": "USER_REPORTED"},
    ]
    if communication_needs:
        lines.append({"label": "Communication requirement", "text": "; ".join(communication_needs), "provenance": "USER_REPORTED"})
    else:
        lines.append({"label": "Communication requirement", "text": DEFAULT_COMMUNICATION, "provenance": "SYSTEM_GENERATED"})
    if location_text and location_text.strip():
        lines.append({"label": "Location", "text": location_text.strip()[:300], "provenance": "USER_REPORTED"})
    lines.append({"label": "Gesture", "text": gesture.name, "provenance": "USER_REPORTED"})
    if gesture.video_storage_key:
        lines.append({"label": "Video", "text": "A recorded gesture demonstration is available.", "provenance": "SYSTEM_GENERATED"})
    text = "\n".join(line["text"] if line["label"] is None else f"{line['label']}: {line['text']}" for line in lines)
    return {
        "lines": lines,
        "text": lines[0]["text"] + "\n\n" + text.split("\n", 1)[1],
        "contact": contact,
        "dial_link": tel_link(contact["number"]),
        "has_video": bool(gesture.video_storage_key),
        "notice": "Your phone places the call, not SignLine. SignLine cannot tell whether the call connected, and a "
                  "voice call does not carry this message or the video: share them separately.",
        "sent": False,
    }
