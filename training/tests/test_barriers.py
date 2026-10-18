"""Barriers between custom words (fewshot.py) and the cosine-margin head (sequence_model.py)."""

import numpy as np
import torch

from mudra_ml.fewshot import CustomWordBank, FewShotConfig, leave_one_out, normalise
from mudra_ml.models.sequence_model import ModelConfig, build_model

RNG = np.random.default_rng(0)
DIM = 32


def _word(center: np.ndarray, n: int, spread: float) -> np.ndarray:
    return normalise(center + RNG.normal(0, spread, (n, DIM)))


def _model_says_nothing():
    return {"status": "uncertain", "top_k": [{"sign": "X", "confidence": 0.1}]}


def test_margin_rejects_a_sign_between_two_close_words():
    a = normalise(RNG.normal(size=DIM))
    b = normalise(a + 0.15 * RNG.normal(size=DIM))  # a very similar word
    examples = {"A": _word(a, 5, 0.02), "B": _word(b, 5, 0.02)}
    between = normalise(a + b)
    open_bank = CustomWordBank(FewShotConfig(min_similarity=0.5, max_model_confidence=0.95, margin=0.0))
    open_bank.set_examples(examples)
    assert open_bank.decide(between, _model_says_nothing()) is not None  # no barrier: the nearest word wins
    barrier = CustomWordBank(FewShotConfig(min_similarity=0.5, max_model_confidence=0.95, margin=0.1))
    barrier.set_examples(examples)
    assert barrier.decide(between, _model_says_nothing()) is None  # inside the barrier: not sure
    clear_a = examples["A"].mean(axis=0)  # a sign right at word A
    assert barrier.decide(clear_a, _model_says_nothing())["sign"] == "A"


def test_word_boundary_rejects_a_far_sign_that_is_still_nearest():
    a = normalise(RNG.normal(size=DIM))
    examples = {"A": _word(a, 6, 0.03)}
    far = normalise(a + 0.6 * normalise(RNG.normal(size=DIM)))  # nearest to A, but outside A's own spread
    loose = CustomWordBank(FewShotConfig(min_similarity=0.5, max_model_confidence=0.95, margin=0.0))
    loose.set_examples(examples)
    tight = CustomWordBank(FewShotConfig(min_similarity=0.5, max_model_confidence=0.95, margin=0.0, radius_k=2.0))
    tight.set_examples(examples)
    assert tight.floors[0] > loose.floors[0] == 0.5
    sim = float(tight.protos[0] @ far)
    assert loose.floors[0] <= sim < tight.floors[0]
    assert loose.decide(far, _model_says_nothing()) is not None
    assert tight.decide(far, _model_says_nothing()) is None  # "I don't know this sign"


def test_outlier_take_is_left_out_but_never_below_three():
    a = normalise(RNG.normal(size=DIM))
    takes = np.vstack([_word(a, 5, 0.02), normalise(RNG.normal(size=(1, DIM)))])  # one wrong take
    bank = CustomWordBank(FewShotConfig(outlier_z=2.0))
    bank.set_examples({"A": takes})
    assert bank.dropped["A"] == [5] and bank.used == [5]
    three = CustomWordBank(FewShotConfig(outlier_z=0.1))
    three.set_examples({"A": takes})
    assert three.used[0] >= 3
    assert leave_one_out(normalise(takes)).shape == (6,)


def test_report_flags_words_that_can_be_mixed_up():
    a = normalise(RNG.normal(size=DIM))
    c = normalise(RNG.normal(size=DIM))
    bank = CustomWordBank(FewShotConfig(min_similarity=0.6, margin=0.05))
    bank.set_examples({"A": _word(a, 4, 0.02), "B": _word(a, 4, 0.02), "C": _word(c, 4, 0.02)})
    rep = bank.report()
    assert rep["A"]["nearest"] == "B" and rep["A"]["too_close"]
    assert not rep["C"]["too_close"]
    assert rep["A"]["consistency"] > 0.9


def test_cosine_head_margin_only_in_training_and_unit_embeddings():
    model = build_model(ModelConfig(input_dim=10, num_classes=4, head="cosine", cos_margin=0.3, cos_scale=20.0)).eval()
    x, y = torch.randn(3, 8, 10), torch.tensor([0, 1, 2])
    plain, margined = model(x), model(x, y)
    assert torch.allclose((plain - margined)[torch.arange(3), y], torch.full((3,), 0.3 * 20.0), atol=1e-4)
    emb = model.embedding_from(model.encode(x))
    assert torch.allclose(emb.norm(dim=1), torch.ones(3), atol=1e-5)
    linear = build_model(ModelConfig(input_dim=10, num_classes=4)).eval()
    assert torch.equal(linear(x), linear(x, y))  # labels do nothing for the old head
