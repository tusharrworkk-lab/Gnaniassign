"""Low-latency OpenRouter chat completion adapter for subagents."""

from __future__ import annotations

import os
from typing import Any, Protocol

import requests

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenRouterChatModel:
    """Minimal HTTP client; one request per completed user turn."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.environ["SUBAGENT_MODEL"]
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.http = requests.Session()

    def complete(self, system: str, user: str) -> str:
        response = self.http.post(
            OPENROUTER_CHAT_URL,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0.2, "max_tokens": 220},
            timeout=20,
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body["choices"][0]["message"]["content"].strip()
