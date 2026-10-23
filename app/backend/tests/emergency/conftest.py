import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from signline_emergency.config import EmergencySettings, set_settings
from signline_emergency.db import reset_engine
from signline_emergency.services.auth import limiter
from signline_emergency.services.recognition import set_provider

PASSWORD = "test-only-password"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A fresh database and an app with only the emergency router (the sign model is not needed)."""
    monkeypatch.setenv("SIGNLINE_EM_DEMO_RESPONDER_PASSWORD", PASSWORD)
    set_settings(EmergencySettings(database_url=f"sqlite:///{(tmp_path / 'em.db').as_posix()}",
                                   secret_key="test-secret", demo_mode=True, media_dir=str(tmp_path / "media"),
                                   media_key_file=str(tmp_path / "media.key"), retention_interval_h=0))
    reset_engine()
    limiter.reset()
    set_provider(None)
    from signline_emergency.api import emergency_router

    app = FastAPI()
    app.include_router(emergency_router())
    with TestClient(app) as c:
        yield c
    reset_engine()
    set_settings(None)


@pytest.fixture
def device():
    return {"X-Device-Key": f"test-device-{uuid.uuid4().hex}"}


class Em:
    """Small helper around one emergency session."""

    def __init__(self, client, device, scenario_id=None, mode="SIGN"):
        self.c, self.device = client, device
        r = client.post("/api/v1/emergency/sessions", json={"mode": mode, "scenario_id": scenario_id}, headers=device)
        assert r.status_code == 200, r.text
        self.id = r.json()["session"]["id"]
        self.h = {"X-Session-Token": r.json()["access_token"]}
        self.view = r.json()["session"]

    def post(self, path, json=None, expect=200):
        r = self.c.post(f"/api/v1/emergency/sessions/{self.id}{path}", json=json, headers=self.h)
        assert r.status_code == expect, r.text
        if expect == 200:
            self.view = r.json()
        return r

    def put(self, path, json, expect=200):
        r = self.c.put(f"/api/v1/emergency/sessions/{self.id}{path}", json=json, headers=self.h)
        assert r.status_code == expect, r.text
        if expect == 200:
            self.view = r.json()
        return r

    def play(self):
        return self.post("/scenario", {"action": "all"})

    def question(self, key):
        return next(q for q in self.view["questions"] if q["key"] == key)

    def pending(self):
        return [q["key"] for q in self.view["questions"] if q["status"] == "PENDING"]

    def answer(self, key, value):
        return self.post(f"/questions/{self.question(key)['id']}/answer", {"value": value})

    def confirm_and_share(self):
        self.post("/review")
        self.post("/confirm")
        return self.post("/share")


@pytest.fixture
def em(client, device):
    return lambda scenario_id=None, mode="SIGN": Em(client, device, scenario_id, mode)


@pytest.fixture
def responder_headers(client):
    r = client.post("/api/v1/emergency/responder/login", json={"username": "demo.responder", "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}
