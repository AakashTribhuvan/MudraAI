"""The teammate's packaged word models (AakashTribhuvan/MitIndia, work/static-landmark-tracking, 8105c5e) merged
with ours: every word is answered by the most accurate model that knows it.

Measured on OUR test split (training/scripts/compare_teammate_models.py), shared words, teammate vs our v9:
common_word_video 23/41 vs 38/41, isl500_video 4/43 vs 43/43, signline_intro 0/92 vs 89/92, static_top10 0/22
vs 22/22, intro_static 0/22 vs 22/22. So v9 keeps every word it knows. A word v9 does not know goes to the
teammate model with the best test accuracy that has it (their own held-out tests: common 88.2%, ISL500 80.5%,
SignLine-26 62.3%); models below MIN_MODEL_ACCURACY (static poses 17.8%, static top-10 unreported, 0 right on
ours) give no words. The others are asked only when v9 is uncertain and accepted at >= their min confidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from mudra_ml.streaming import ModelBundle, SegmenterConfig

from mudra_api.isl500_video import ISL500WordModel, temporal_features

log = logging.getLogger("mudra_api")

TO_OURS = {"THANKYOU": "THANK_YOU", "THANK": "THANK_YOU", "I_ME_MINE_MY": "I"}  # their gloss -> ours
FALLBACK_MIN_CONFIDENCE = 0.7
MIN_MODEL_ACCURACY = 0.5
# whole-model held-out accuracy on the teammate's own test (per-word scores exist only for the common-word model)
TEAMMATE_TEST_ACCURACY = {"common_word_video": 0.882, "isl500_video": 0.805, "signline_intro_video": 0.623,
                          "intro_static": 0.178, "static_top10": None}


def ours(sign: str) -> str:
    return TO_OURS.get(sign, sign)


def own_threshold(cfg: SegmenterConfig) -> SegmenterConfig:
    """Segmenter settings for another model: its own card threshold, no word boundaries built for ours."""
    return replace(cfg, commit_threshold=None, known_boundary_k=None)


def evaluation(model_dir: Path) -> dict | None:
    path = model_dir / "evaluation_on_our_test.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@dataclass
class WordSource:
    """One model in the merge: its id, every word it knows, and a whole-segment predictor -> (sign, confidence)."""
    model_id: str
    vocabulary: list[str]
    predict: Callable
    accuracy: float | None
    words: list[str] | None = None  # the words this model answers in the merge (set by MergedFallback)
    min_confidence: float = FALLBACK_MIN_CONFIDENCE


def bundle_source(model_id: str, bundle: ModelBundle) -> WordSource:
    def predict(segment, w, h):
        p = bundle.probs([segment], w, h)[0]
        best = int(np.argmax(p))
        return ours(bundle.classes[best]), float(p[best])
    return WordSource(model_id, [ours(c) for c in bundle.classes if not c.startswith("_")], predict,
                      TEAMMATE_TEST_ACCURACY.get(model_id))


def svm_source(model_id: str, model: ISL500WordModel) -> WordSource:
    def predict(segment, w, h):
        try:
            vec, _ = temporal_features(segment, w, h, model.descriptor)
        except ValueError:
            return None
        top = model.predict(vec)["top_k"][0]
        return ours(top["sign"]), float(top["confidence"])
    return WordSource(model_id, [ours(c) for c in model.classes], predict, TEAMMATE_TEST_ACCURACY.get(model_id))


class MergedFallback:
    """Words our model does not know, each from the most accurate other model that has it; asked only when our
    model is uncertain (StreamingRecognizer(fallback=...))."""

    def __init__(self, sources: list[WordSource], known: set[str]):
        usable = [s for s in sources if (s.accuracy or 0) >= MIN_MODEL_ACCURACY]
        usable.sort(key=lambda s: -(s.accuracy or 0))
        taken = set(known)
        self.sources = []
        for s in usable:
            words = sorted(set(s.vocabulary) - taken)
            if words:
                taken |= set(words)
                self.sources.append(replace(s, words=words))
        self.excluded = sorted({w for s in sources if s not in usable for w in s.vocabulary} - taken)
        self.routing = {w: s.model_id for s in self.sources for w in s.words}

    @property
    def extra(self) -> list[str]:
        return sorted(self.routing)

    def __call__(self, segment, width: int, height: int, result: dict) -> dict | None:
        for s in self.sources:
            got = s.predict(segment, width, height)
            if got and got[0] in s.words and got[1] >= s.min_confidence:
                return {"status": "ok", "sign": got[0], "confidence": got[1],
                        "top_k": [{"sign": got[0], "confidence": got[1]}], "model": s.model_id,
                        "word_model_top_k": result.get("top_k", [])}
        return None


class ExtraModels:
    def __init__(self, common_word_dir: Path, isl500_path: Path, signline_path: Path | None = None, device: str = "cpu"):
        self.common_word: ModelBundle | None = None
        self.isl500: ISL500WordModel | None = None
        self.signline: ISL500WordModel | None = None
        try:
            self.common_word = ModelBundle(common_word_dir, device)
            log.info("Loaded teammate common-word model (%d words)", len(self.common_word.classes))
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            log.info("Common-word model is unavailable: %s", exc)
        for attr, path, label in (("isl500", isl500_path, "ISL500 video"), ("signline", signline_path, "SignLine-26 video")):
            if path is None:
                continue
            try:
                setattr(self, attr, ISL500WordModel(path))
                log.info("Loaded teammate %s model (%d words)", label, len(getattr(self, attr).classes))
            except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
                log.info("%s model is unavailable: %s", label, exc)

    def common_words(self) -> list[str]:
        return [ours(c) for c in self.common_word.classes if not c.startswith("_")] if self.common_word else []

    def isl500_words(self) -> list[str]:
        return [ours(c) for c in self.isl500.classes] if self.isl500 else []

    def sources(self, extra_sources: list[WordSource] = ()) -> list[WordSource]:
        out = []
        if self.common_word is not None:
            out.append(bundle_source("common_word_video", self.common_word))
        if self.isl500 is not None:
            out.append(svm_source("isl500_video", self.isl500))
        if self.signline is not None:
            out.append(svm_source("signline_intro_video", self.signline))
        return out + list(extra_sources)
