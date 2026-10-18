"""Few-shot custom words: recognise a new sign from 1-5 recorded examples, without retraining.

Each example is passed through the trained model up to its pooled sequence representation
(``SequenceClassifier.encode``). A custom word's prototype is the mean of its examples'
normalised embeddings. A new sign matches a custom word when its cosine similarity to that
prototype is high enough AND the trained model itself is not confident about a known word;
both limits are chosen on validation data by ``training/scripts/eval_fewshot.py``.

When a custom word gets enough recordings it should be added to the training data and the
model retrained; prototypes are the bridge until then.

Barriers between words (2026-10-18), all optional and tuned on validation by eval_fewshot.py:
- ``margin``: the best word must beat the runner-up by this much, else no custom word is chosen;
- ``radius_k``: each word gets its own boundary from how much its takes vary (leave-one-out similarity
  mean - radius_k x std); a sign outside it is not that word, even when it is the nearest one;
- ``outlier_z``: a take far less similar to the word's other takes than they are to each other (a wrong or
  half-done sign) is left out of the prototype (only with 4+ takes, never below 3).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class FewShotConfig:
    min_similarity: float = 0.8  # cosine similarity to a prototype needed to accept a custom word
    max_model_confidence: float = 0.8  # the custom word wins only if the model's best known word is below this
    margin: float = 0.05  # best custom prototype must beat the second best by this much
    radius_k: float | None = None  # per-word boundary (None = only min_similarity)
    outlier_z: float | None = None  # leave inconsistent takes out of a prototype (None = keep all)

    @classmethod
    def load(cls, path: Path | None) -> FewShotConfig:
        if path and Path(path).exists():
            return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)


def normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def leave_one_out(e: np.ndarray) -> np.ndarray:
    """Similarity of each (unit) take to the mean of the word's other takes."""
    total = e.sum(axis=0)
    return np.array([float(normalise(total - e[i]) @ e[i]) for i in range(len(e))])


class CustomWordBank:
    """Prototypes of custom words, built from embeddings of their recorded examples."""

    def __init__(self, cfg: FewShotConfig | None = None):
        self.cfg = cfg or FewShotConfig()
        self.labels: list[str] = []
        self.protos = np.zeros((0, 0), np.float32)
        self.counts: list[int] = []
        self.used: list[int] = []
        self.dropped: dict[str, list[int]] = {}  # label -> indices of takes left out of the prototype
        self.floors = np.zeros(0, np.float32)  # per-word acceptance boundary
        self.consistency: list[float | None] = []  # mean leave-one-out similarity of the used takes

    def __len__(self) -> int:
        return len(self.labels)

    def _keep(self, e: np.ndarray) -> np.ndarray:
        z = self.cfg.outlier_z
        if z is None or len(e) < 4:
            return np.arange(len(e))
        loo = leave_one_out(e)
        keep = np.flatnonzero(loo >= loo.mean() - z * max(loo.std(), 1e-6))
        return keep if len(keep) >= 3 else np.argsort(-loo)[:3]

    def set_examples(self, examples: dict[str, np.ndarray]) -> None:
        """examples: label -> [n, D] embeddings (any n >= 1)."""
        items = [(k, normalise(np.asarray(v, np.float32))) for k, v in sorted(examples.items()) if len(v)]
        self.labels = [k for k, _ in items]
        self.counts = [len(v) for _, v in items]
        protos, floors, used, cons = [], [], [], []
        self.dropped = {}
        for label, e in items:
            keep = self._keep(e)
            self.dropped[label] = sorted(set(range(len(e))) - set(keep.tolist()))
            kept = e[keep]
            protos.append(normalise(kept.mean(axis=0)))
            used.append(len(kept))
            loo = leave_one_out(kept) if len(kept) >= 2 else None
            cons.append(round(float(loo.mean()), 4) if loo is not None else None)
            floor = self.cfg.min_similarity
            if self.cfg.radius_k is not None and loo is not None:
                floor = max(floor, float(loo.mean() - self.cfg.radius_k * loo.std()))
            floors.append(min(floor, 0.995))
        self.protos = np.stack(protos) if protos else np.zeros((0, 0), np.float32)
        self.floors = np.array(floors, np.float32)
        self.used, self.consistency = used, cons

    def match(self, emb: np.ndarray) -> tuple[str | None, float, float]:
        """(best label, its similarity, runner-up similarity) for one embedding."""
        if not self.labels:
            return None, 0.0, 0.0
        sims = self.protos @ normalise(np.asarray(emb, np.float32))
        order = np.argsort(-sims)
        second = float(sims[order[1]]) if len(order) > 1 else -1.0
        return self.labels[int(order[0])], float(sims[order[0]]), second

    def decide(self, emb: np.ndarray, model_result: dict) -> dict | None:
        """A custom-word result that overrides ``model_result``, or None to keep the model's answer."""
        label, sim, second = self.match(emb)
        if label is None or sim < self.floors[self.labels.index(label)] or sim - second < self.cfg.margin:
            return None
        known = model_result["top_k"][0]["confidence"] if model_result.get("top_k") else 0.0
        if model_result.get("status") == "ok" and known >= self.cfg.max_model_confidence:
            return None
        top = [{"sign": label, "confidence": sim, "custom": True}] + list(model_result.get("top_k", []))[:2]
        return {"status": "ok", "sign": label, "confidence": sim, "custom": True, "top_k": top}

    def report(self) -> dict[str, dict]:
        """Per word: takes used / left out, consistency, its boundary, and the nearest other word.

        ``too_close``: the nearest word's prototype lies inside this word's boundary (or the margin), so
        the two can be mixed up; record clearer or more varied takes, or keep only one of them."""
        out: dict[str, dict] = {}
        sims = self.protos @ self.protos.T if len(self.labels) else np.zeros((0, 0))
        for i, label in enumerate(self.labels):
            entry = {"takes": self.counts[i], "used": self.used[i], "left_out": self.dropped.get(label, []),
                     "consistency": self.consistency[i], "boundary": round(float(self.floors[i]), 4),
                     "nearest": None, "nearest_similarity": None, "too_close": False}
            if len(self.labels) > 1:
                row = sims[i].copy()
                row[i] = -np.inf
                j = int(np.argmax(row))
                near = float(row[j])
                entry.update(nearest=self.labels[j], nearest_similarity=round(near, 4),
                             too_close=bool(near >= min(self.floors[i], self.floors[j]) or 1 - near < self.cfg.margin))
            out[label] = entry
        return out


class KnownWordBoundaries:
    """Open-set check for the trained model's own words (the "I don't know this sign" barrier).

    Each known word's training clips give a centroid and a spread of their embeddings; a sign the model
    labels as word W is only accepted when its embedding lies within W's boundary
    (similarity to W's centroid >= mean - k x std of W's training clips). Signs never taught to the
    model otherwise get forced onto the nearest known word.
    """

    def __init__(self, labels: np.ndarray, embs: np.ndarray, k: float):
        embs = normalise(np.asarray(embs, np.float32))
        self.classes = sorted(set(labels.tolist()))
        self.centroids = np.stack([normalise(embs[labels == c].mean(axis=0)) for c in self.classes])
        self.mean = np.zeros(len(self.classes), np.float32)
        self.std = np.zeros(len(self.classes), np.float32)
        for i, c in enumerate(self.classes):
            sims = embs[labels == c] @ self.centroids[i]
            self.mean[i], self.std[i] = float(sims.mean()), float(sims.std()) if len(sims) > 1 else 1.0
        self.index = {c: i for i, c in enumerate(self.classes)}
        self.set_k(k)

    def set_k(self, k: float) -> None:
        self.k = k
        self.floors = self.mean - k * self.std

    def inside(self, sign: str, emb: np.ndarray) -> bool:
        i = self.index.get(sign)
        return True if i is None else float(self.centroids[i] @ normalise(np.asarray(emb, np.float32))) >= self.floors[i]

    def save(self, path) -> None:
        np.savez(path, classes=np.array(self.classes), centroids=self.centroids, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path, k: float) -> KnownWordBoundaries:
        z = np.load(path)
        obj = cls.__new__(cls)
        obj.classes = z["classes"].tolist()
        obj.centroids, obj.mean, obj.std = z["centroids"], z["mean"], z["std"]
        obj.index = {c: i for i, c in enumerate(obj.classes)}
        obj.set_k(k)
        return obj
