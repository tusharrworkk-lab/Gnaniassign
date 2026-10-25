"""Local dual-encoder router for the trained nine-label retail MiniLM model."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


RETAIL_MINILM_KEYS = (
    "track_order",
    "return_item",
    "cancel_order",
    "refund_status",
    "wrong_item",
    "size_issue",
    "payment_issue",
    "delivery_address",
    "other",
)
RETAIL_INSTRUCTION = "What is the customer's intent?"
DEFAULT_MODEL_PATH = Path("artifacts/models/systemone-retail-v1")


class RetailMiniLMRouter:
    """Score retail label text against a query with the fine-tuned dual encoder."""

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self.model_path = model_path or os.getenv("RETAIL_MINILM_MODEL", str(DEFAULT_MODEL_PATH))
        requested = device or os.getenv("RETAIL_MINILM_DEVICE", "auto")
        self.requested_device = requested
        self.max_length = int(os.getenv("RETAIL_MINILM_MAX_LENGTH", "64"))
        self.device: Any | None = None
        self.last_ms = 0.0
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._F: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from torch.nn import functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install MiniLM support with: uv sync --extra transformers-experiment") from error
        if self.requested_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RETAIL_MINILM_DEVICE requests CUDA but PyTorch cannot access an NVIDIA GPU")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Retail MiniLM model not found: {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModel.from_pretrained(self.model_path).to(self.device).eval()
        self._torch, self._F = torch, F

    def warmup(self) -> None:
        self._load()

    def predict(self, state: str | dict[str, Any], questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        if model is not None and model != self.model_path:
            raise ValueError("The local retail MiniLM router serves one fixed checkpoint")
        query = state.get("current_query", "") if isinstance(state, dict) else state
        if not isinstance(query, str) or not query.strip():
            return {"model": self.model_path, "answers": {"intent": {"type": "choice", "choice": "other", "confidence": 0.0}}}
        started = time.perf_counter()
        try:
            self._load()
            state_inputs = self._tokenizer(query, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            option_inputs = self._tokenizer(
                [f"{RETAIL_INSTRUCTION}: {label}" for label in RETAIL_MINILM_KEYS],
                return_tensors="pt", padding=True, truncation=True, max_length=self.max_length,
            ).to(self.device)
            with self._torch.inference_mode():
                state_embedding = self._F.normalize(self._model(**state_inputs).last_hidden_state[:, 0], dim=-1)
                option_embeddings = self._F.normalize(self._model(**option_inputs).last_hidden_state[:, 0], dim=-1)
                probabilities = self._torch.softmax((state_embedding @ option_embeddings.T).squeeze(0) * 20, dim=0)
            scores = {label: float(probability) for label, probability in zip(RETAIL_MINILM_KEYS, probabilities.tolist())}
            choice = max(scores, key=scores.get)
            return {"model": self.model_path, "answers": {"intent": {"type": "choice", "choice": choice, "confidence": scores[choice], "probabilities": scores}}}
        finally:
            self.last_ms = (time.perf_counter() - started) * 1000
