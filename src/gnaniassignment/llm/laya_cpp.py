"""Adapter for a persistent local :mod:`laya.cpp` HTTP server."""

from __future__ import annotations

import os
import time
from typing import Any

import requests


DEFAULT_LAYA_CPP_URL = "http://127.0.0.1:8080"
DEFAULT_LAYA_CPP_MODEL = "laya"


class LayaCppRouter:
    """Route JEV-compatible typed decisions to a local native Laya server.

    ``laya.cpp`` keeps the checkpoint resident in one process.  This adapter is
    deliberately HTTP-only: starting its CLI for every turn would include model
    load time and make the router unusably slow.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("LAYA_CPP_URL") or DEFAULT_LAYA_CPP_URL).rstrip("/")
        self.model = os.getenv("LAYA_CPP_MODEL") or DEFAULT_LAYA_CPP_MODEL
        self.api_key = api_key if api_key is not None else os.getenv("LAYA_CPP_API_KEY")
        self.timeout_seconds = float(os.getenv("LAYA_CPP_TIMEOUT_SECONDS", "5"))
        self.last_ms = 0.0
        self.last_headers_ms: float | None = None
        self.http = requests.Session()

    def predict(
        self, state: str | dict[str, Any], questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        """Send one System One decision request to the loaded native checkpoint."""
        started = time.perf_counter()
        self.last_headers_ms = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self.http.post(
                f"{self.base_url}/v1/systemone",
                headers=headers,
                json={"model": model or self.model, "state": state, "questions": questions},
                timeout=self.timeout_seconds,
            )
            elapsed = getattr(response, "elapsed", None)
            if elapsed is not None:
                self.last_headers_ms = elapsed.total_seconds() * 1000
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
                raise RuntimeError("laya.cpp response did not contain typed answers")
            return body
        finally:
            self.last_ms = (time.perf_counter() - started) * 1000

    def health(self) -> dict[str, Any]:
        """Return the server readiness payload without submitting inference."""
        response = self.http.get(f"{self.base_url}/health", timeout=self.timeout_seconds)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("laya.cpp health response was not a JSON object")
        return body
