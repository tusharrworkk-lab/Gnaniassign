"""Vercel AI Gateway adapter for intent-specific response agents."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Protocol

from openai import OpenAI

DEFAULT_SUBAGENT_MODEL = "meta/llama-3.1-8b"
DEFAULT_GATEWAY_URL = "https://ai-gateway.vercel.sh/v1"
DEFAULT_VLLM_URL = "http://127.0.0.1:8000/v1"
DEFAULT_GEMMA_MODEL = "google/gemma-3-1b-it"


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...

    def stream_complete(self, system: str, user: str) -> Iterator[str]: ...


class VercelGatewayChatModel:
    """One reusable SDK client for all completed turns in a session."""

    def __init__(self, model: str | None = None, api_key: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("SUBAGENT_MODEL") or DEFAULT_SUBAGENT_MODEL
        self.last_first_token_ms: float | None = None
        self.last_stream_ms: float | None = None
        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv("AI_GATEWAY_API_KEY")
            if not key:
                raise ValueError("Set AI_GATEWAY_API_KEY in .env or your environment")
            self.client = OpenAI(
                api_key=key,
                base_url=os.getenv("SUBAGENT_GATEWAY_URL") or DEFAULT_GATEWAY_URL,
                timeout=20.0,
                max_retries=0,
            )

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Vercel AI Gateway returned an empty text response")
        return content.strip()

    def stream_complete(self, system: str, user: str) -> Iterator[str]:
        started = time.perf_counter()
        self.last_first_token_ms = None
        self.last_stream_ms = None
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=220,
                stream=True,
            )
            for event in stream:
                if event.choices:
                    content = event.choices[0].delta.content
                    if isinstance(content, str) and content:
                        if self.last_first_token_ms is None:
                            self.last_first_token_ms = (time.perf_counter() - started) * 1000
                        yield content
        finally:
            self.last_stream_ms = (time.perf_counter() - started) * 1000


class VllmGemmaChatModel:
    """Streaming client for the persistent, local vLLM Gemma server."""

    def __init__(self, model: str | None = None, api_key: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("VLLM_GEMMA_MODEL") or DEFAULT_GEMMA_MODEL
        self.last_first_token_ms: float | None = None
        self.last_stream_ms: float | None = None
        self.client = client or OpenAI(
            api_key=api_key or os.getenv("VLLM_API_KEY", "local-gemma"),
            base_url=os.getenv("VLLM_BASE_URL") or DEFAULT_VLLM_URL,
            timeout=float(os.getenv("VLLM_TIMEOUT_SECONDS", "10")),
            max_retries=0,
        )

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=60,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Local vLLM Gemma returned an empty text response")
        return content.strip()

    def stream_complete(self, system: str, user: str) -> Iterator[str]:
        started = time.perf_counter()
        self.last_first_token_ms = None
        self.last_stream_ms = None
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=60,
                stream=True,
            )
            for event in stream:
                if not event.choices:
                    continue
                content = event.choices[0].delta.content
                if isinstance(content, str) and content:
                    if self.last_first_token_ms is None:
                        self.last_first_token_ms = (time.perf_counter() - started) * 1000
                    yield content
        finally:
            self.last_stream_ms = (time.perf_counter() - started) * 1000
