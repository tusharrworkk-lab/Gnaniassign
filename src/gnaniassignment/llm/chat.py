"""OpenRouter Python SDK adapter for intent-specific response agents."""

from __future__ import annotations

import os
from typing import Protocol

from openrouter import OpenRouter

DEFAULT_SUBAGENT_MODEL = "meta-llama/llama-3.1-8b-instruct"


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenRouterChatModel:
    """One reusable SDK client for all completed turns in a session."""

    def __init__(self, model: str | None = None, api_key: str | None = None, client: OpenRouter | None = None) -> None:
        self.model = model or os.getenv("SUBAGENT_MODEL") or DEFAULT_SUBAGENT_MODEL
        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise ValueError("Set OPENROUTER_API_KEY in .env or your environment")
            self.client = OpenRouter(api_key=key)

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.send(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=220,
            timeout_ms=20_000,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenRouter returned an empty text response")
        return content.strip()
