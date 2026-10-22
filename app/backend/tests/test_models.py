"""Selectable word models: GET /models, /labels?model=, and the stream's per-session model choice."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from mudra_api.app import create_app
from mudra_api.settings import Settings

STATIC = BACKEND / "models" / "mudra_intro_static_pose_medoids_v1.npz"
pytestmark = pytest.mark.skipif(not STATIC.exists(), reason="static-pose model not present")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(Settings(device="cpu")))


def test_models_lists_v9_merged_and_auto(client):
    models = client.get("/models").json()["models"]
    assert [m["id"] for m in models][:1] == ["sequence"] and models[0]["default"] and not models[0]["experimental"]
    ids = {m["id"] for m in models}
    assert {"merged", "auto"} <= ids and not any(m["default"] for m in models if m["id"] != "sequence")


def test_labels_per_model(client):
    static = client.get("/labels", params={"model": "intro_static"}).json()
    assert static["count"] == 10 and {"HELP", "YOU"} <= {s["sign"] for s in static["labels"]}
    assert client.get("/labels", params={"model": "nope"}).status_code == 400


def test_stream_uses_the_chosen_model(client):
    frame = {"type": "frame", "t": 0.0, "hands": [], "pose": None}
    with client.websocket_connect("/stream") as ws:
        ws.send_json({"type": "start", "width": 640, "height": 480, "model": "missing_model"})
        assert ws.receive_json()["event"] == "error"
    with client.websocket_connect("/stream") as ws:
        ws.send_json({"type": "start", "width": 640, "height": 480, "model": "intro_static"})
        for i in range(5):  # no hands: the static recogniser stays idle and sends nothing
            ws.send_json({**frame, "t": i / 10})
        ws.send_json({"type": "start", "width": 640, "height": 480})  # default model again
        ws.send_json(frame)


def test_safe_joblib_refuses_code_and_loads_the_teammate_svm(tmp_path):
    import pickle

    from mudra_api import safe_joblib

    class Evil:
        def __reduce__(self):
            return (print, ("this must never run",))

    bad = tmp_path / "bad.joblib"
    bad.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError):
        safe_joblib.load(bad)
    svm = BACKEND / "models" / "isl500_video_pilot" / "model.joblib"
    if svm.exists():
        payload = safe_joblib.load(svm)
        assert len(payload["classes"]) == 14 and hasattr(payload["estimator"], "predict_proba")


COMMON = BACKEND / "models" / "common_word_video_baseline" / "model.pt"


@pytest.mark.skipif(not COMMON.exists(), reason="teammate models not present")
def test_merged_mode_adds_only_words_our_model_lacks(client):
    models = {m["id"]: m for m in client.get("/models").json()["models"]}
    ours = {s["sign"] for s in client.get("/labels").json()["labels"]}
    merged = client.get("/labels", params={"model": "merged"}).json()["labels"]
    added = {s["sign"] for s in merged} - ours
    assert added == set(models["merged"]["words"]) and "MYSELF" in added and not added & ours
    by_model = models["merged"]["words_by_model"]
    assert not set(by_model["isl500_video"]) & set(by_model.get("common_word_video", []))  # one model per word
    excluded = set(models["merged"]["excluded_words"])  # static models: below the accuracy bar, only words we lack
    assert "HOW" in excluded and not excluded & ours
    assert models["auto"]["letters"] and models["auto"]["words"] == models["merged"]["words"]


def test_merge_routes_each_word_to_the_most_accurate_model():
    from mudra_api.extra_models import MergedFallback, WordSource

    def fixed(sign, conf):
        return lambda *a: (sign, conf)

    sources = [WordSource("weak", ["WATER", "TEA"], fixed("TEA", 0.99), 0.6),
               WordSource("strong", ["WATER", "HELLO"], fixed("WATER", 0.9), 0.9),
               WordSource("bad", ["NAME"], fixed("NAME", 0.99), 0.2)]
    m = MergedFallback(sources, known={"HELLO"})
    assert m.routing == {"WATER": "strong", "TEA": "weak"} and m.excluded == ["NAME"]
    assert m(None, 1, 1, {"top_k": []})["sign"] == "WATER"
    low = MergedFallback([WordSource("s", ["WATER"], fixed("WATER", 0.6), 0.9)], known=set())
    assert low(None, 1, 1, {}) is None  # below 0.7 confidence
    known = MergedFallback([WordSource("s", ["HELLO"], fixed("HELLO", 0.95), 0.9)], known={"HELLO"})
    assert known(None, 1, 1, {}) is None  # our model knows HELLO: never answered by another model


def test_auto_mode_letters_need_a_held_raised_hand():
    import numpy as np
    from mudra_api.auto_mode import AutoRecognizer
    from mudra_ml.streaming import Frame

    class Words:  # stands in for the word recognizer: never signing, no events
        class segmenter:
            state = "idle"

        def push(self, frame):
            return None

    class Letters:
        def predict_frames(self, frames, w, h):
            return {"status": "ok", "letter": "A", "confidence": 0.9, "top_k": []}

    pose = np.zeros((33, 4), np.float32)
    pose[11, :2], pose[12, :2] = (0.6, 0.4), (0.4, 0.4)
    hands = np.zeros((2, 21, 3), np.float32)

    def frames(wrist_y, n=20):
        hands[1, :, 1] = wrist_y
        return [Frame(t=i * 0.1, hands=hands.copy(), present=np.array([False, True]), pose=pose, pose_present=True)
                for i in range(n)]

    rec = AutoRecognizer(Words(), Letters(), 640, 480)
    letters = [ev for f in frames(0.45) for ev in rec.push(f) if ev["event"] == "letter"]
    assert [ev["letter"] for ev in letters] == ["A"]  # held: accepted once, not repeated
    rec = AutoRecognizer(Words(), Letters(), 640, 480)
    assert not [ev for f in frames(0.9) for ev in rec.push(f)]  # hand down by the hips: no letters
