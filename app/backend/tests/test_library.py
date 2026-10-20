"""Letters and phrases: feature properties, model files and the /library, /letters, /phrases endpoints."""

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from mudra_ml.letters import (
    FEATURE_DIM,
    LETTERS,
    LetterModel,
    letter_features,
    mirror,
)

from mudra_api.app import create_app
from mudra_api.settings import Settings

LETTER_DIR = BACKEND / "models" / "isl_letters_v1"
PHRASE_DIR = BACKEND / "models" / "isl_phrases_v1_bilstm"
LETTER_DATA = BACKEND.parents[1] / "data" / "processed" / "landmarks" / "letters"


def _hand(cx: float, cy: float, size: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, size / 3, (21, 3)).astype(np.float32)
    pts[0] = 0
    pts[9] = [0, -size, 0]  # middle-finger base above the wrist: palm size = size
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def test_features_ignore_hand_order_and_image_scale():
    a, b = _hand(0.3, 0.6, seed=1), _hand(0.7, 0.6, seed=2)
    f1 = letter_features([a, b])
    assert f1.shape == (FEATURE_DIM,)
    assert np.allclose(f1, letter_features([b, a]))  # MediaPipe's hand order does not matter
    small = [(h - 0.5) * 0.5 + 0.5 for h in (a, b)]  # the same shape further from the camera
    assert np.allclose(f1, letter_features(small), atol=1e-4)
    assert f1[129] == 1 and f1[130] == 1
    one = letter_features([a])
    assert one[129] == 1 and one[130] == 0 and not one[63:126].any()
    assert not letter_features([]).any()


def test_mirror_flips_x_only():
    h = _hand(0.3, 0.6)
    m = mirror([h])[0]
    assert np.allclose(m[:, 0], 1 - h[:, 0]) and np.allclose(m[:, 1:], h[:, 1:])


@pytest.mark.skipif(not (LETTER_DIR / "weights.npz").exists(), reason="letter model not exported")
def test_letter_model_card_and_forward():
    model = LetterModel(LETTER_DIR)
    assert model.classes == LETTERS
    probs = model.probabilities(np.zeros((2, FEATURE_DIM), np.float32))
    assert probs.shape == (2, 26) and np.allclose(probs.sum(1), 1)
    assert model.predict_frames([[], []], 640, 480)["status"] == "no_hands"
    assert {"val", "test"} <= set(model.card["metrics"])


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(Settings(device="cpu")))


@pytest.mark.skipif(not (LETTER_DIR / "weights.npz").exists() or not (PHRASE_DIR / "model.pt").exists(),
                    reason="letter or phrase model not exported")
def test_library_listing(client):
    body = client.get("/library").json()
    assert [x["label"] for x in body["letters"]] == LETTERS
    assert len(body["phrases"]) >= 90
    assert "letters" in body["models"] and "phrases" in body["models"]


@pytest.mark.skipif(not (LETTER_DATA / "A.npz").exists(), reason="letter landmarks not extracted (local data)")
def test_letter_reference_and_prediction_on_training_photo(client):
    ref = client.get("/reference/LETTER_A")
    assert ref.status_code == 200
    clip = ref.json()
    assert clip["kind"] == "letter" and len(clip["frames"]) == 15 and clip["frames"][0]["pose"] is None
    hands = [{"landmarks": h} for h in clip["frames"][0]["hands"] if h]
    res = client.post("/letters/predict", json={"frames": [{"hands": hands}], "width": 100, "height": 100}).json()
    assert res["top_k"][0]["sign"].startswith("LETTER_") and len(res["top_k"]) == 3
    assert client.post("/letters/predict", json={"frames": [{"hands": []}], "width": 100, "height": 100}).json()["status"] == "no_hands"
    assert client.post("/letters/predict", json={"frames": [], "width": 100, "height": 100}).status_code == 422


@pytest.mark.skipif(not (PHRASE_DIR / "model.pt").exists(), reason="phrase model not exported")
def test_phrase_prediction_shape(client):
    ref = client.get("/reference/ARE_YOU_FREE_TODAY")
    if ref.status_code != 200:
        pytest.skip("phrase landmarks not present (local data)")
    clip = ref.json()
    frames = [{"hands": [{"landmarks": h} for h in f["hands"] if h], "pose": f["pose"]} for f in clip["frames"]][:256]
    res = client.post("/phrases/predict", json={"frames": frames, "width": clip["width"], "height": clip["height"]}).json()
    assert res["model"] == "isl_phrases_v1_bilstm" and len(res["top_k"]) == 5
    assert any(t["sign"] == "ARE_YOU_FREE_TODAY" for t in res["top_k"])  # a training clip: must be among the guesses


def test_chirality_flips_with_a_mirror_image():
    from mudra_api.library import chirality
    rng = np.random.default_rng(0)
    hand = rng.normal(size=(21, 3))
    mirrored = hand * np.array([-1.0, 1.0, 1.0])
    assert chirality(hand) == pytest.approx(-chirality(mirrored))


@pytest.mark.skipif(not LETTER_DATA.exists(), reason="letter landmarks are local data")
def test_letter_hands_sit_in_their_own_slots():
    from mudra_api.library import LetterReferences, chirality
    refs = LetterReferences(LETTER_DATA)
    for letter in "CLOUV":  # one-handed letters: always on the signer's right
        left, right = refs.get(f"LETTER_{letter}")["frames"][0]["hands"]
        assert left is None and right is not None
    for sign in refs.available:
        left, right = refs.get(sign)["frames"][0]["hands"]
        if left is not None and right is not None:
            assert chirality(left) < chirality(right)
