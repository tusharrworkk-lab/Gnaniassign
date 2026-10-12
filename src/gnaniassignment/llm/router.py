"""Adapters for intent decision models."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"


class JevRouter:
    """OpenRouter Jev adapter implementing the domain IntentRouter contract."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.last_ms = 0.0
        self.http = requests.Session()

    def predict(self, state: str, questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = self.http.post(
                DECISIONS_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": JEV_MODEL, "state": state, "questions": questions},
                timeout=30,
            )
        finally:
            self.last_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        body = response.json()
        if "answers" not in body:
            raise RuntimeError(f"Jev response had no answers: {list(body)}")
        return body


def load_laya():
    """Load the local multilingual Laya classifier once."""
    import torch
    from laya import Router

    device = "cuda" if torch.cuda.is_available() else "cpu"
    router = Router(device=device, max_loaded=1, default="multilingual")
    router.preload(["multilingual"])
    if device == "cuda":
        router._agents["multilingual"].model.half()
    return router
