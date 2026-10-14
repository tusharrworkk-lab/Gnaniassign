"""Versioned decision taxonomy and a compact multilingual text classifier."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DecisionCatalog:
    """The stable label contract for one voice-AI decision point."""

    name: str
    version: str
    labels: tuple[str, ...]
    fallback_label: str = "other"

    @classmethod
    def from_json(cls, path: str | Path) -> "DecisionCatalog":
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        labels = tuple(body["labels"])
        catalog = cls(body["name"], body["version"], labels, body.get("fallback_label", "other"))
        if not catalog.labels or catalog.fallback_label not in catalog.labels:
            raise ValueError("Catalog must contain at least one label and its fallback_label")
        if len(set(catalog.labels)) != len(catalog.labels):
            raise ValueError("Catalog labels must be unique")
        return catalog


@dataclass(frozen=True)
class DecisionResult:
    label: str
    confidence: float
    scores: dict[str, float]
    model_name: str
    model_version: str
    latency_ms: float


class TfidfDecisionModel:
    """A persisted word + character n-gram classifier with abstention.

    Imports of scikit-learn and joblib stay lazy so normal API use does not
    require the optional training dependency.
    """

    def __init__(self, pipeline: Any, catalog: DecisionCatalog, threshold: float = 0.65) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.pipeline = pipeline
        self.catalog = catalog
        self.threshold = threshold

    @classmethod
    def train(
        cls, texts: list[str], labels: list[str], catalog: DecisionCatalog, threshold: float = 0.65
    ) -> "TfidfDecisionModel":
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts and labels must be non-empty and have equal length")
        invalid = set(labels) - set(catalog.labels)
        if invalid:
            raise ValueError(f"Training labels are absent from catalog: {sorted(invalid)}")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import FeatureUnion, Pipeline
        except ImportError as error:
            raise RuntimeError("Install training support with: uv sync --extra decision-model") from error
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)),
            ])),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial")),
        ])
        pipeline.fit(texts, labels)
        return cls(pipeline, catalog, threshold)

    def decide(self, text: str) -> DecisionResult:
        text = text.strip()
        if not text:
            return DecisionResult(self.catalog.fallback_label, 0.0, {}, self.catalog.name, self.catalog.version, 0.0)
        started = time.perf_counter()
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        scores = {str(label): float(score) for label, score in zip(classes, probabilities, strict=True)}
        chosen = max(scores, key=scores.get)
        confidence = scores[chosen]
        if confidence < self.threshold:
            chosen = self.catalog.fallback_label
        return DecisionResult(chosen, confidence, scores, self.catalog.name, self.catalog.version, (time.perf_counter() - started) * 1000)

    def save(self, path: str | Path) -> None:
        try:
            import joblib
        except ImportError as error:
            raise RuntimeError("Install training support with: uv sync --extra decision-model") from error
        joblib.dump({"pipeline": self.pipeline, "catalog": self.catalog, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfDecisionModel":
        try:
            import joblib
        except ImportError as error:
            raise RuntimeError("Install decision-model support with: uv sync --extra decision-model") from error
        payload = joblib.load(path)
        return cls(payload["pipeline"], payload["catalog"], payload["threshold"])
