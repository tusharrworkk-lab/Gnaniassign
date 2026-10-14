"""Local adapter for Hari5115's 13-label Hinglish retail BERT classifier."""

from __future__ import annotations

import os
import time
from typing import Any

from gnaniassignment.llm.prompts.retail_intents import RETAIL_INTENT_KEYS


RETAIL_MODEL_ID = "Hari5115/hinglish-retail-intent-classifier"


class HinglishRetailRouter:
    """Run the published retail classifier locally with a stable response shape."""

    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        self.model_id = model_id or os.getenv("RETAIL_INTENT_MODEL", RETAIL_MODEL_ID)
        self.device = device or os.getenv("RETAIL_INTENT_DEVICE", "cuda")
        self.max_length = int(os.getenv("RETAIL_INTENT_MAX_LENGTH", "128"))
        self.last_ms = 0.0
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install the retail classifier with: uv sync --extra retail-classifier") from error
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("RETAIL_INTENT_DEVICE requests CUDA but PyTorch cannot access an NVIDIA GPU")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.to(self.device)
        if self.device.startswith("cuda"):
            self._model.half()
        self._model.eval()
        labels = tuple(self._model.config.id2label[index] for index in range(self._model.config.num_labels))
        expected = (*RETAIL_INTENT_KEYS[:5], "other", *RETAIL_INTENT_KEYS[5:])
        if labels != expected:
            raise RuntimeError(f"Retail classifier label contract changed: {labels!r}")
        self._torch = torch

    def warmup(self) -> None:
        """Load the tokenizer and model before the first customer utterance."""
        self._load()

    def predict(self, state: str | dict[str, Any], questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        if model is not None and model != self.model_id:
            raise ValueError("The local retail classifier serves one fixed checkpoint")
        query = state.get("current_query", "") if isinstance(state, dict) else state
        if not isinstance(query, str) or not query.strip():
            return {"model": self.model_id, "answers": {"intent": {"type": "choice", "choice": "other", "confidence": 0.0}}}
        started = time.perf_counter()
        try:
            self._load()
            encoded = self._tokenizer(query, return_tensors="pt", truncation=True, max_length=self.max_length)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                probabilities = self._torch.softmax(self._model(**encoded).logits.float(), dim=-1)[0]
            scores = {self._model.config.id2label[index]: float(value) for index, value in enumerate(probabilities.tolist())}
            choice = max(scores, key=scores.get)
            return {"model": self.model_id, "answers": {"intent": {"type": "choice", "choice": choice, "confidence": scores[choice], "probabilities": scores}}}
        finally:
            self.last_ms = (time.perf_counter() - started) * 1000
