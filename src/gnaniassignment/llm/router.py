"""Vercel AI Gateway adapter for Laya intent decisions."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

LAYA_GATEWAY_URL = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
LAYA_MODEL = "convaiinnovations/laya"


class LayaGatewayRouter:
    """Route typed Choice questions to Laya through Vercel AI Gateway."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("AI_GATEWAY_API_KEY")
        if not self.api_key:
            raise ValueError("Set AI_GATEWAY_API_KEY in .env or your environment")
        self.gateway_url = os.getenv("LAYA_GATEWAY_URL") or LAYA_GATEWAY_URL
        self.model = os.getenv("LAYA_MODEL") or LAYA_MODEL
        self.last_ms = 0.0
        self.http = requests.Session()

    def predict(self, state: str | dict[str, Any], questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = self.http.post(
                self.gateway_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": model or self.model, "state": state, "questions": questions},
                timeout=30,
            )
        finally:
            self.last_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise RuntimeError("Laya response did not contain typed answers")
        return body
