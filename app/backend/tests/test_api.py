"""API tests against the exported model and real (held-out) INCLUDE test clips."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from mudra_ml.config import PROJECT_ROOT
from mudra_ml.preprocessing.sequence import load_sequence

from mudra_api.app import create_app
from mudra_api.settings import Settings

MODEL_DIR = Settings().model_dir  # the deployed word model (settings.py / MUDRA_MODEL_DIR)
TEST_SPLIT = PROJECT_ROOT / "data" / "metadata" / "splits" / "include" / "test.csv"

pytestmark = pytest.mark.skipif(not (MODEL_DIR / "model.pt").exists(), reason="exported model not present")
# words the deployed model knows (its model card; classes starting with "_" are internal, e.g. _BACKGROUND_)
NUM_SIGNS = (sum(not c.startswith("_") for c in json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))["classes"])
             if (MODEL_DIR / "model_card.json").exists() else 0)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(Settings(model_dir=MODEL_DIR, device="cpu")))


def clip_to_request(npz_path: Path, top_k: int = 3) -> dict:
    """Turn a stored clip into the JSON the browser sends: detected hands + pose per frame."""
    seq = load_sequence(npz_path)
    frames = []
    for t in range(seq.num_frames):
        hands = [
            {"landmarks": seq.hands[t, h].round(5).tolist(), "score": 1.0}
            for h in range(2)
            if seq.hand_present[t, h]
        ]
        pose = seq.pose[t].round(5).tolist() if seq.pose_present[t] else None
        frames.append({"hands": hands, "pose": pose})
    return {"frames": frames, "width": seq.meta["width"], "height": seq.meta["height"], "top_k": top_k}


def held_out_clips(n: int) -> list[dict]:
    if not TEST_SPLIT.exists():
        pytest.skip("INCLUDE splits not generated")
    with TEST_SPLIT.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if float(r["any_hand_rate"]) >= 0.9]
    rng = np.random.default_rng(0)
    return [rows[i] for i in rng.choice(len(rows), n, replace=False)]


def test_health_and_labels(client):
    health = client.get("/health").json()
    assert health["model_loaded"] and health["num_signs"] == NUM_SIGNS
    labels = client.get("/labels").json()
    assert labels["count"] == NUM_SIGNS
    entry = next(x for x in labels["labels"] if x["sign"] == "GOOD_MORNING")
    assert entry["label"] == "Good morning" and isinstance(entry["has_reference"], bool)
    card = client.get("/model").json()
    assert "feature_mean" not in card and card["active_threshold"] > 0
    models = client.get("/models").json()["models"]
    intro = next(model for model in models if model["id"] == "intro_static")
    assert (intro["trained_words"], intro["live_words"], intro["experimental"]) == (32, 10, True)
    intro_labels = client.get("/labels", params={"model": "intro_static"}).json()["labels"]
    assert len(intro_labels) == 32
    assert sum(label["live"] for label in intro_labels) == 10


def test_intro_static_stream_uses_selected_model(client, monkeypatch):
    instances = []

    class StubStaticRecognizer:
        def __init__(self, model, width, height):
            instances.append(self)

        def push(self, frame):
            return {
                "event": "sign_ended",
                "state": "idle",
                "result": {
                    "status": "uncertain",
                    "sign": None,
                    "label": None,
                    "confidence": 0.5,
                    "top_k": [],
                    "model": "intro_static",
                    "score_type": "static_hold",
                },
                "stats": {},
                "t0": frame.t,
                "t1": frame.t,
            }

    monkeypatch.setattr("mudra_api.app.StaticPoseStreamingRecognizer", StubStaticRecognizer)
    with client.websocket_connect("/stream") as ws:
        ws.send_json({"type": "start", "width": 640, "height": 480, "model": "intro_static"})
        ws.send_json({"type": "frame", "t": 1.0, "hands": [], "pose": None})
        event = ws.receive_json()

    assert len(instances) == 1
    assert event["event"] == "sign_ended"
    assert event["result"]["model"] == "intro_static"
    assert event["result"]["score_type"] == "static_hold"


def test_predicts_held_out_test_clips(client):
    clips = held_out_clips(25)
    correct_top1 = correct_top3 = 0
    for row in clips:
        body = client.post("/predict", json=clip_to_request(PROJECT_ROOT / row["landmarks_path"])).json()
        top = [c["sign"] for c in body["top_k"]]
        correct_top1 += top[0] == row["label"]
        correct_top3 += row["label"] in top
        assert body["inference_ms"] < 1000
    # Offline test accuracy is 97%; the API path must stay in the same range.
    assert correct_top1 >= 22, f"top-1 {correct_top1}/25"
    assert correct_top3 >= 24, f"top-3 {correct_top3}/25"


def test_no_hands_is_not_a_guess(client):
    frames = [{"hands": [], "pose": None} for _ in range(24)]
    body = client.post("/predict", json={"frames": frames, "width": 640, "height": 480}).json()
    assert body["status"] == "no_hands"
    assert body["sign"] is None and body["label"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"frames": [{"hands": []}] * 3, "width": 640, "height": 480},  # too few frames
        {"frames": [{"hands": [{"landmarks": [[0, 0, 0]] * 20}]}] * 10, "width": 640, "height": 480},  # 20 points
        {"frames": [{"pose": [[0, 0, 0, 1]] * 10}] * 10, "width": 640, "height": 480},  # short pose
        {"frames": [{"hands": []}] * 10, "width": 0, "height": 480},  # bad size
    ],
)
def test_invalid_payloads_are_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_model_gives_clear_503(tmp_path):
    c = TestClient(create_app(Settings(model_dir=tmp_path / "nope", device="cpu")))
    assert c.get("/health").json()["model_loaded"] is False
    r = c.post("/predict", json={"frames": [{"hands": []}] * 10, "width": 640, "height": 480})
    assert r.status_code == 503 and "export_model.py" in r.json()["detail"]


COMBINED_DIR = MODEL_DIR
COMBINED_TEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "combined_words_v3" / "test.csv"


@pytest.mark.skipif(not (COMBINED_DIR / "model.pt").exists() or not COMBINED_TEST.exists(), reason="combined model/data absent")
def test_combined_model_waits_for_the_whole_sign():
    """The start of a sign (rest + first moments) must give no_sign/uncertain, never a wrong word."""
    c = TestClient(create_app(Settings(model_dir=COMBINED_DIR, device="cpu")))
    assert c.get("/health").json()["num_signs"] == NUM_SIGNS
    assert all(not item["sign"].startswith("_") for item in c.get("/labels").json()["labels"])
    with COMBINED_TEST.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == "islwords"][:12]
    waiting = 0
    for r in rows:
        full = clip_to_request(PROJECT_ROOT / r["landmarks_path"])
        opening = {**full, "frames": full["frames"][: max(8, len(full["frames"]) // 6)]}
        body = c.post("/predict", json=opening).json()
        assert body["status"] != "ok" or body["sign"] == r["label"], f"wrong word committed early: {body['sign']}"
        waiting += body["status"] in ("no_sign", "uncertain", "no_hands")
        assert all(not cand["sign"].startswith("_") for cand in body["top_k"])
    assert waiting >= 8


@pytest.mark.skipif(not (COMBINED_DIR / "model.pt").exists() or not COMBINED_TEST.exists(), reason="combined model/data absent")
def test_stream_segments_and_recognises_a_sign():
    c = TestClient(create_app(Settings(model_dir=COMBINED_DIR, device="cpu")))
    with COMBINED_TEST.open(newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["source"] == "islwords" and float(r["any_hand_rate"]) > 0.9)
    req = clip_to_request(PROJECT_ROOT / row["landmarks_path"])
    rest = [{"hands": [], "pose": req["frames"][0]["pose"]}] * 30  # 1 s of resting before and after
    frames = rest + req["frames"] + rest
    # Expected events straight from the shared Python implementation
    from mudra_ml.config import project_path as pp
    from mudra_ml.preprocessing.sequence import assign_hands
    from mudra_ml.streaming import (
        Frame,
        ModelBundle,
        SegmenterConfig,
        StreamingRecognizer,
    )

    local = StreamingRecognizer(ModelBundle(COMBINED_DIR), SegmenterConfig.load(pp("training/configs/streaming.json")),
                                req["width"], req["height"])
    expected = []
    for i, fr in enumerate(frames):
        pose = np.asarray(fr["pose"], np.float32) if fr["pose"] else None
        hands, present, _ = assign_hands([np.asarray(h["landmarks"], np.float32) for h in fr["hands"]],
                                         ["" for _ in fr["hands"]], [1.0 for _ in fr["hands"]], pose, False)
        ev = local.push(Frame(i / 30, hands, present, pose if pose is not None else np.zeros((33, 4), np.float32), pose is not None))
        if ev:
            expected.append(ev)
    assert [e["event"] for e in expected][-1] == "sign_ended"
    assert expected[-1]["result"]["status"] == "ok" and expected[-1]["result"]["sign"] == row["label"]

    with c.websocket_connect("/stream") as ws:
        ws.send_json({"type": "start", "width": req["width"], "height": req["height"]})
        for i, fr in enumerate(frames):
            ws.send_json({"type": "frame", "t": i / 30, **fr})
        got = [ws.receive_json() for _ in expected]
    assert [g["event"] for g in got] == [e["event"] for e in expected]
    assert got[-1]["result"]["sign"] == row["label"] and got[-1]["result"]["label"]


def test_segmenter_states():
    from mudra_ml.streaming import Frame, SegmenterConfig, SignSegmenter

    seg = SignSegmenter(SegmenterConfig(on_s=0.1, off_s=0.2, min_sign_s=0.2, pad_pre_s=0, pad_post_s=0), 640, 480)
    pose = np.zeros((33, 4), np.float32)
    pose[11], pose[12] = [0.6, 0.5, 0, 1], [0.4, 0.5, 0, 1]
    hand_up = np.zeros((2, 21, 3), np.float32)
    hand_up[1, :, :2] = [0.5, 0.45]  # wrist above the shoulder line
    events = []
    for i in range(90):
        t = i / 30
        signing = 30 <= i < 60
        present = np.array([False, signing])
        ev, segment = seg.push(Frame(t, hand_up if signing else np.zeros_like(hand_up), present, pose, True))
        if ev:
            events.append((ev, round(t, 2), None if segment is None else segment.hand_present.shape[0]))
    assert [e[0] for e in events] == ["sign_started", "sign_ended"]
    assert 1.0 <= events[0][1] <= 1.2 and events[1][1] <= 2.3


def test_segmenter_parked_hands_end_a_sign_and_need_movement_to_start():
    """Hands resting in view (a desk in front of a close webcam) always count as raised; with
    ``still_s`` they end the sign once still, and only movement starts the next one."""
    from mudra_ml.streaming import Frame, SegmenterConfig, SignSegmenter

    cfg = SegmenterConfig(on_s=0.05, off_s=0.3, min_sign_s=0.2, pad_pre_s=0, pad_post_s=0, max_sign_s=6.0,
                          still_s=0.5, still_speed=0.5, still_below_y=0.0)
    pose = np.zeros((33, 4), np.float32)
    pose[11], pose[12] = [0.6, 0.5, 0, 1], [0.4, 0.5, 0, 1]
    rng = np.random.default_rng(0)

    def hand_at(x, y):
        h = np.zeros((2, 21, 3), np.float32)
        h[1, :, :2] = [x, y]
        h[1, :, :2] += rng.normal(0, 0.002, (21, 2))  # tracking jitter
        return h

    for still_s in (0.0, 0.5):
        seg = SignSegmenter(SegmenterConfig(**{**cfg.to_dict(), "still_s": still_s}), 640, 480)
        events = []
        for i in range(150):  # 5 s at 30 fps: rest in view, a 1 s movement, rest in view again
            t = i / 30
            moving = 30 <= i < 60
            x = 0.5 + 0.1 * np.sin((i - 30) / 30 * 2 * np.pi) if moving else 0.5
            ev, _ = seg.push(Frame(t, hand_at(x, 0.65 if not moving else 0.55), np.array([False, True]), pose, True))
            if ev:
                events.append((ev, round(t, 2)))
        if still_s == 0:  # without parking the resting hands keep the first sign open (6 s cap)
            assert [e[0] for e in events] == ["sign_started"], events
    # with parking: hands appearing at rest give at most one ignored too_short; the movement is one
    # sign that ends on stillness; resting afterwards starts nothing (tracking jitter is not movement)
    names = [e[0] for e in events]
    assert names in (["sign_started", "too_short", "sign_started", "sign_ended"], ["sign_started", "sign_ended"]), events
    assert 0.9 <= events[-2][1] <= 1.2 and events[-1][1] <= 3.2 and seg.last_end["reason"] == "still"


def test_live_session_is_saved_only_when_enabled(tmp_path):
    from mudra_ml.preprocessing.sequence import load_sequence

    hand = [[0.5, 0.4, 0.0]] * 21
    pose = [[0.5, 0.5, 0.0, 1.0]] * 33
    for enabled in (False, True):
        out = tmp_path / ("on" if enabled else "off")
        c = TestClient(create_app(Settings(model_dir=MODEL_DIR, device="cpu", session_dir=out if enabled else None)))
        with c.websocket_connect("/stream") as ws:
            ws.send_json({"type": "start", "width": 640, "height": 480})
            ws.send_json({"type": "diag", "camera_fps": 7, "tracking_fps": 7, "inference_ms": 30, "delegate": "GPU"})
            for i in range(40):
                ws.send_json({"type": "frame", "t": i / 10, "hands": [{"landmarks": hand, "score": 0.9}], "pose": pose})
        saved = sorted(out.glob("session_*.npz")) if out.exists() else []
        assert len(saved) == (1 if enabled else 0)
    seq = load_sequence(saved[0])
    assert seq.num_frames == 40 and seq.meta["client"]["camera_fps"] == 7 and seq.meta["width"] == 640


@pytest.mark.skipif(not (COMBINED_DIR / "model.pt").exists() or not COMBINED_TEST.exists(), reason="combined model/data absent")
def test_recordings_store_and_custom_words(tmp_path):
    c = TestClient(create_app(Settings(model_dir=COMBINED_DIR, device="cpu", recordings_dir=tmp_path)))
    with COMBINED_TEST.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == "islwords" and float(r["any_hand_rate"]) > 0.9][:3]
    ids = []
    for i, r in enumerate(rows):
        req = clip_to_request(PROJECT_ROOT / r["landmarks_path"])
        frames = [{"t": k / 30, **fr} for k, fr in enumerate(req["frames"])]
        body = c.post("/recordings", json={"label": "my new word", "signer": "Test User", "width": req["width"],
                                           "height": req["height"], "frames": frames}).json()
        assert body["label"] == "MY_NEW_WORD" and body["signer"] == "test_user" and body["quality_flag"] == ""
        assert body["custom_active"] and not body["in_model"] and body["label_count"] == i + 1
        ids.append(body["sample_id"])
    summary = c.get("/recordings").json()
    assert summary["total"] == 3 and summary["custom_words"][0] == {"sign": "MY_NEW_WORD", "label": "My new word", "examples": 3}
    assert c.get("/health").json()["custom_words"] == 1
    q = summary["signs"][0]["quality"]  # barrier report: how well the takes agree and the word's boundary
    assert q["takes"] == 3 and q["used"] == 3 and 0.0 < q["consistency"] <= 1.0 and q["boundary"] >= 0.0
    manifest = (tmp_path / "metadata" / "manifest_app.csv").read_text(encoding="utf-8")
    assert manifest.count("app:session:test_user:") == 3 and manifest.count("app:test_user") == 3
    # a recording of a word the model knows is stored for retraining but is not a custom word
    req = clip_to_request(PROJECT_ROOT / rows[0]["landmarks_path"])
    known = c.post("/recordings", json={"label": rows[0]["label"], "width": req["width"], "height": req["height"],
                                        "frames": [{"t": k / 30, **fr} for k, fr in enumerate(req["frames"])]}).json()
    assert known["in_model"] and not known["custom_active"]  # one take of a known word: not yet personal
    for _ in range(2):  # 3 good takes of a known word: the signer's own examples become active
        known = c.post("/recordings", json={"label": rows[0]["label"], "width": req["width"], "height": req["height"],
                                            "frames": [{"t": k / 30, **fr} for k, fr in enumerate(req["frames"])]}).json()
    assert known["in_model"] and known["custom_active"] and known["label_count"] == 3
    assert known["quality"]["consistency"] > 0.99  # the same clip three times
    assert known["quality"]["looks_like"] is None  # the model reads it as the word itself
    for sid in ids:
        assert c.delete(f"/recordings/{sid}").status_code == 200
    assert c.get("/health").json()["custom_words"] == 1  # the known word's personal examples remain
    assert c.delete("/recordings/missing").status_code == 404
    assert c.post("/recordings", json={"label": "x", "width": 640, "height": 480, "frames": [{"t": 0}] * 3}).status_code == 422


def test_custom_word_bank_decision():
    from mudra_ml.fewshot import CustomWordBank, FewShotConfig

    bank = CustomWordBank(FewShotConfig(min_similarity=0.8, max_model_confidence=0.8, margin=0.05))
    bank.set_examples({"A": np.array([[1, 0, 0], [0.9, 0.1, 0]]), "B": np.array([[0, 1, 0]])})
    unsure = {"status": "uncertain", "sign": None, "confidence": 0.4, "top_k": [{"sign": "X", "confidence": 0.4}]}
    sure = {"status": "ok", "sign": "X", "confidence": 0.95, "top_k": [{"sign": "X", "confidence": 0.95}]}
    assert bank.decide(np.array([1.0, 0.05, 0]), unsure)["sign"] == "A"
    assert bank.decide(np.array([1.0, 0.05, 0]), sure) is None  # the model is confident about a known word
    assert bank.decide(np.array([0, 0, 1.0]), unsure) is None  # unlike every prototype
    assert bank.decide(np.array([1.0, 1.0, 0]), unsure) is None  # A and B equally close


@pytest.mark.skipif(not (COMBINED_DIR / "model.pt").exists() or not COMBINED_TEST.exists(), reason="combined model/data absent")
def test_reference_clips():
    c = TestClient(create_app(Settings(model_dir=COMBINED_DIR, device="cpu")))
    labels = c.get("/labels").json()["labels"]
    with_ref = [x for x in labels if x["has_reference"]]
    assert len(with_ref) > 200
    ref = c.get(f"/reference/{with_ref[0]['sign']}").json()
    assert ref["sign"] == with_ref[0]["sign"] and ref["frames"] and ref["width"] > 0
    first = next(f for f in ref["frames"] if f["pose"])
    assert len(first["pose"]) == 33 and len(first["pose"][0]) == 4
    assert c.get("/reference/NOT_A_SIGN").status_code == 404


def test_client_diagnostics_are_logged_once_a_minute(tmp_path):
    """Browser fps / delegate reports reach the events log (first report, then at most once a minute)."""
    log = tmp_path / "events.jsonl"
    c = TestClient(create_app(Settings(model_dir=MODEL_DIR, device="cpu", events_log=log)))
    diag = {"type": "diag", "camera_fps": 30, "tracking_fps": 22, "inference_ms": 18, "delegate": "GPU",
            "camera": "Integrated Webcam", "width": 640, "height": 480, "landmarks": [[0.1, 0.2]]}
    with c.websocket_connect("/stream") as ws:
        ws.send_json({"type": "start", "width": 640, "height": 480})
        ws.send_json(diag)
        ws.send_json({**diag, "tracking_fps": 21})  # within the minute: not logged again
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [x["event"] for x in lines] == ["client_diag"]
    assert lines[0]["tracking_fps"] == 22 and lines[0]["delegate"] == "GPU" and "landmarks" not in lines[0]


def test_segmenter_joins_a_short_piece_with_the_rest_of_the_sign():
    """A sign with a pause: a short piece ended by stillness is held back and joined with the next
    movement (join_short_s), instead of being classified alone (live COMPUTER takes, 2026-10-15)."""
    from mudra_ml.streaming import Frame, SegmenterConfig, SignSegmenter

    pose = np.zeros((33, 4), np.float32)
    pose[11], pose[12] = [0.6, 0.5, 0, 1], [0.4, 0.5, 0, 1]

    def frame(t, x):
        h = np.zeros((2, 21, 3), np.float32)
        h[1, :, :2] = [x, 0.55]
        return Frame(t, h, np.array([False, True]), pose, True)

    def run(join):
        cfg = SegmenterConfig(on_s=0.05, off_s=0.3, min_sign_s=0.2, pad_pre_s=0, pad_post_s=0, still_s=0.4,
                              still_speed=0.5, still_below_y=0.0, join_short_s=join, join_gap_s=1.0)
        seg, events = SignSegmenter(cfg, 640, 480), []
        x = 0.5
        for i in range(150):  # rest, 0.2 s move, 1.0 s pause, 1.0 s move, then still (pieces of 0.33-0.44 s were seen live)
            t = i / 30
            if 30 <= i < 36 or 66 <= i < 96:  # about 3.5 shoulder widths / s: clearly moving
                x = 0.5 + 0.08 * np.sin(2 * np.pi * 2 * t)
            ev, segment = seg.push(frame(t, x))
            if ev:
                events.append((ev, None if segment is None else round(float(segment.timestamps_ms[0]) / 1000, 2)))
        return events

    assert [e[0] for e in run(0.0)].count("sign_ended") == 2  # without joining: two pieces
    joined = run(0.5)
    ended = [e for e in joined if e[0] == "sign_ended"]
    assert len(ended) == 1 and ended[0][1] <= 1.1, joined  # one sign, starting at the first piece


def test_events_endpoint_returns_latest_lines_newest_first(tmp_path):
    log = tmp_path / "events.jsonl"
    lines = [json.dumps({"event": "sign_ended", "n": i}) for i in range(5)] + ["not json"]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    c = TestClient(create_app(Settings(model_dir=MODEL_DIR, device="cpu", events_log=log)))
    body = c.get("/events?limit=3").json()
    assert body["total"] == 6 and [e["n"] for e in body["events"]] == [4, 3, 2]
    empty = TestClient(create_app(Settings(model_dir=MODEL_DIR, device="cpu", events_log=tmp_path / "none.jsonl")))
    assert empty.get("/events").json() == {"total": 0, "events": []}
