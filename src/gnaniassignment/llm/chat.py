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
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_CEREBRAS_URL = "https://api.cerebras.ai/v1"
DEFAULT_CEREBRAS_MODEL = "llama-3.3-70b"
DEFAULT_GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_SIMPLISMART_URL = "https://api.simplismart.live"
DEFAULT_SIMPLISMART_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_BEDROCK_REGION = "ap-south-1"
DEFAULT_BEDROCK_MODEL = "google.gemma-3-4b-it"
DEFAULT_BEDROCK_GPT_OSS_MODEL = "openai.gpt-oss-120b-1:0"


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


class CerebrasChatModel:
    """Low-latency streamed Llama responses via Cerebras' OpenAI-compatible API."""

    def __init__(self, model: str | None = None, api_key: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("CEREBRAS_MODEL") or DEFAULT_CEREBRAS_MODEL
        self.last_first_token_ms: float | None = None
        self.last_stream_ms: float | None = None
        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv("CEREBRAS_API_KEY")
            if not key:
                raise ValueError("Set CEREBRAS_API_KEY in .env or your environment")
            self.client = OpenAI(
                api_key=key,
                base_url=os.getenv("CEREBRAS_BASE_URL") or DEFAULT_CEREBRAS_URL,
                timeout=float(os.getenv("CEREBRAS_TIMEOUT_SECONDS", "20")),
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
            max_tokens=int(os.getenv("CEREBRAS_MAX_OUTPUT_TOKENS", "32")),
            reasoning_effort=os.getenv("CEREBRAS_REASONING_EFFORT", "none"),
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Cerebras returned an empty text response")
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
                max_tokens=int(os.getenv("CEREBRAS_MAX_OUTPUT_TOKENS", "32")),
                reasoning_effort=os.getenv("CEREBRAS_REASONING_EFFORT", "none"),
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


class GroqChatModel:
    """Stream Llama responses through Groq's OpenAI-compatible API."""

    def __init__(self, model: str | None = None, api_key: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
        self.last_first_token_ms: float | None = None
        self.last_stream_ms: float | None = None
        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("Set GROQ_API_KEY in .env or your environment")
            self.client = OpenAI(
                api_key=key,
                base_url=os.getenv("GROQ_BASE_URL") or DEFAULT_GROQ_URL,
                timeout=float(os.getenv("GROQ_TIMEOUT_SECONDS", "20")),
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
            max_tokens=int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "32")),
            reasoning_effort=os.getenv("GROQ_REASONING_EFFORT", "low"),
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Groq returned an empty text response")
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
                max_tokens=int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "32")),
                reasoning_effort=os.getenv("GROQ_REASONING_EFFORT", "low"),
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


class SimplismartChatModel:
    """Stream Llama 3.1 8B through Simplismart's chat-completions API."""

    def __init__(self, model: str | None = None, api_key: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("SIMPLISMART_MODEL") or DEFAULT_SIMPLISMART_MODEL
        self.last_first_token_ms: float | None = None
        self.last_stream_ms: float | None = None
        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv("SIMPLISMART_API_KEY")
            if not key:
                raise ValueError("Set SIMPLISMART_API_KEY in .env or your environment")
            self.client = OpenAI(
                api_key=key,
                base_url=os.getenv("SIMPLISMART_BASE_URL") or DEFAULT_SIMPLISMART_URL,
                timeout=float(os.getenv("SIMPLISMART_TIMEOUT_SECONDS", "20")),
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
            max_tokens=int(os.getenv("SIMPLISMART_MAX_OUTPUT_TOKENS", "32")),
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Simplismart returned an empty text response")
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
                max_tokens=int(os.getenv("SIMPLISMART_MAX_OUTPUT_TOKENS", "32")),
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


class BedrockChatModel:
    """Stream Gemma from the in-region Amazon Bedrock Mumbai endpoint."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        self.model = model or os.getenv("BEDROCK_MODEL") or DEFAULT_BEDROCK_MODEL
        self.region = os.getenv("BEDROCK_REGION") or DEFAULT_BEDROCK_REGION
        self.auth_mode = os.getenv("BEDROCK_AUTH_MODE", "api_key").lower()
        self.last_first_token_ms: float | None = None
        self.last_stream_ms: float | None = None
        if client is not None:
            self.client = client
            return

        try:
            import boto3
        except ImportError as error:
            raise RuntimeError("Install the Bedrock provider with: uv sync --extra bedrock") from error

        if self.auth_mode == "api_key":
            # Boto3 recognizes this official environment variable and sends the
            # key as a Bedrock bearer token instead of SigV4 IAM credentials.
            bearer_token = api_key or os.getenv("BEDROCK_API_KEY")
            if bearer_token:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token
        elif self.auth_mode == "iam":
            # Explicitly prevent an inherited bearer token from overriding the
            # selected IAM/SSO/profile credential chain in this process.
            os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        else:
            raise ValueError("BEDROCK_AUTH_MODE must be 'api_key' or 'iam'.")
        profile = os.getenv("AWS_PROFILE")
        session = boto3.Session(profile_name=profile or None, region_name=self.region)
        self.client = session.client("bedrock-runtime")

    def _request(self, system: str, user: str) -> dict[str, object]:
        return {
            "modelId": self.model,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {
                "temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.5")),
                "maxTokens": int(os.getenv("BEDROCK_MAX_OUTPUT_TOKENS", "32")),
            },
        }

    def complete(self, system: str, user: str) -> str:
        response = self.client.converse(**self._request(system, user))
        content = response["output"]["message"]["content"]
        text = "".join(part.get("text", "") for part in content)
        if not text.strip():
            raise ValueError("Amazon Bedrock returned an empty text response")
        return text.strip()

    def stream_complete(self, system: str, user: str) -> Iterator[str]:
        started = time.perf_counter()
        self.last_first_token_ms = None
        self.last_stream_ms = None
        try:
            response = self.client.converse_stream(**self._request(system, user))
            for event in response["stream"]:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                content = delta.get("text")
                if isinstance(content, str) and content:
                    if self.last_first_token_ms is None:
                        self.last_first_token_ms = (time.perf_counter() - started) * 1000
                    yield content
        finally:
            self.last_stream_ms = (time.perf_counter() - started) * 1000


class BedrockGptOssChatModel:
    """GPT-OSS-specific Bedrock Converse adapter.

    GPT-OSS emits separate reasoning and final-text content blocks.  Reasoning
    is deliberately consumed but never surfaced to the caller or TTS.  This
    adapter is intentionally separate from :class:`BedrockChatModel`, which
    keeps the existing Gemma request and streaming behaviour unchanged.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        self.model = model or os.getenv("BEDROCK_GPT_OSS_MODEL") or DEFAULT_BEDROCK_GPT_OSS_MODEL
        self.region = os.getenv("BEDROCK_REGION") or DEFAULT_BEDROCK_REGION
        self.auth_mode = os.getenv("BEDROCK_AUTH_MODE", "api_key").lower()
        self.last_first_token_ms: float | None = None
        self.last_stream_ms: float | None = None
        if client is not None:
            self.client = client
            return

        try:
            import boto3
        except ImportError as error:
            raise RuntimeError("Install the Bedrock provider with: uv sync --extra bedrock") from error

        if self.auth_mode == "api_key":
            bearer_token = api_key or os.getenv("BEDROCK_API_KEY")
            if bearer_token:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token
        elif self.auth_mode == "iam":
            os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        else:
            raise ValueError("BEDROCK_AUTH_MODE must be 'api_key' or 'iam'.")
        profile = os.getenv("AWS_PROFILE")
        session = boto3.Session(profile_name=profile or None, region_name=self.region)
        self.client = session.client("bedrock-runtime")

    def _request(self, system: str, user: str) -> dict[str, object]:
        return {
            "modelId": self.model,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {
                "temperature": float(os.getenv("BEDROCK_GPT_OSS_TEMPERATURE", "0.2")),
                # Reasoning is included in this completion budget.  32 tokens,
                # suitable for Gemma, often ends before GPT-OSS emits final text.
                "maxTokens": int(os.getenv("BEDROCK_GPT_OSS_MAX_OUTPUT_TOKENS", "256")),
            },
            "additionalModelRequestFields": {
                "reasoning_effort": os.getenv("BEDROCK_GPT_OSS_REASONING_EFFORT", "low"),
            },
        }

    def complete(self, system: str, user: str) -> str:
        response = self.client.converse(**self._request(system, user))
        content = response["output"]["message"]["content"]
        text = "".join(part.get("text", "") for part in content)
        if not text.strip():
            stop_reason = response.get("stopReason", "unknown")
            raise ValueError(
                "GPT-OSS returned no final text "
                f"(stopReason={stop_reason!r}). Increase BEDROCK_GPT_OSS_MAX_OUTPUT_TOKENS if needed."
            )
        return text.strip()

    def stream_complete(self, system: str, user: str) -> Iterator[str]:
        started = time.perf_counter()
        self.last_first_token_ms = None
        self.last_stream_ms = None
        emitted_text = False
        stop_reason = "unknown"
        try:
            response = self.client.converse_stream(**self._request(system, user))
            for event in response["stream"]:
                message_stop = event.get("messageStop")
                if isinstance(message_stop, dict):
                    stop_reason = str(message_stop.get("stopReason", stop_reason))
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                # `reasoningContent` is intentionally ignored. It is internal
                # deliberation, not user-facing agent speech.
                content = delta.get("text")
                if isinstance(content, str) and content:
                    emitted_text = True
                    if self.last_first_token_ms is None:
                        self.last_first_token_ms = (time.perf_counter() - started) * 1000
                    yield content
            if not emitted_text:
                raise ValueError(
                    "GPT-OSS returned no final text "
                    f"(stopReason={stop_reason!r}). Increase BEDROCK_GPT_OSS_MAX_OUTPUT_TOKENS if needed."
                )
        finally:
            self.last_stream_ms = (time.perf_counter() - started) * 1000

    def stream_complete_with_memory_recall(self, system: str, user: str, recall_tool, on_recall_started) -> Iterator[str]:
        """Stream normally, letting GPT-OSS request private recall when needed.

        The first turn contains no prior-call text. If the model emits the
        sole allowed tool use, application code fetches the caller-scoped
        summaries and resumes the same Bedrock conversation.
        """
        tool_config = {
            "tools": [{"toolSpec": {"name": "memory_recall", "description": "Retrieve compact summaries from this caller's previous conversations only when the caller explicitly asks about a past conversation.", "inputSchema": {"json": {"type": "object", "properties": {}, "additionalProperties": False}}}}]
        }
        request = self._request(system, user)
        request["toolConfig"] = tool_config
        messages = request["messages"]
        response = self.client.converse_stream(**request)
        tool_blocks: dict[int, dict[str, object]] = {}
        stop_reason = "unknown"
        for event in response["stream"]:
            start = event.get("contentBlockStart", {})
            start_value = start.get("start", {}) if isinstance(start, dict) else {}
            tool = start_value.get("toolUse") if isinstance(start_value, dict) else None
            if isinstance(tool, dict):
                tool_blocks[int(start.get("contentBlockIndex", 0))] = {"toolUse": {"toolUseId": tool.get("toolUseId"), "name": tool.get("name"), "input": ""}}
            delta_event = event.get("contentBlockDelta", {})
            delta = delta_event.get("delta", {}) if isinstance(delta_event, dict) else {}
            if isinstance(delta, dict):
                text = delta.get("text")
                if isinstance(text, str) and text:
                    yield text
                tool_delta = delta.get("toolUse")
                index = int(delta_event.get("contentBlockIndex", 0))
                if isinstance(tool_delta, dict) and index in tool_blocks:
                    tool_blocks[index]["toolUse"]["input"] += str(tool_delta.get("input", ""))
            message_stop = event.get("messageStop")
            if isinstance(message_stop, dict):
                stop_reason = str(message_stop.get("stopReason", stop_reason))
        if stop_reason != "tool_use":
            return
        calls = [block["toolUse"] for _, block in sorted(tool_blocks.items())]
        if len(calls) != 1 or calls[0].get("name") != "memory_recall":
            raise ValueError("GPT-OSS requested an unsupported tool")
        # Let speech begin before the database lookup and continuation. The
        # callback stays outside this generator's text stream so the filler is
        # not accidentally included in the final LLM answer or its summary.
        on_recall_started()
        result = recall_tool()
        request["messages"] = [
            *messages,
            {"role": "assistant", "content": [{"toolUse": call} for call in calls]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": calls[0]["toolUseId"], "content": [{"json": result}], "status": "success"}}]},
        ]
        follow_up = self.client.converse_stream(**request)
        for event in follow_up["stream"]:
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            text = delta.get("text")
            if isinstance(text, str) and text:
                yield text


class VertexGeminiChatModel:
    """Gemini 2.5 Flash through Vertex AI, streaming text for TTS."""

    def __init__(self, model: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise RuntimeError("Install the Gemini provider with: uv sync --extra gemini") from error

        self._types = types
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.last_first_token_ms: float | None = None
        self.last_stream_ms: float | None = None
        auth_mode = os.getenv("GEMINI_AUTH_MODE", "vertex_api_key").lower()
        http_options = types.HttpOptions(api_version="v1")
        if auth_mode == "vertex_api_key":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Set GEMINI_API_KEY for Vertex AI Express Mode")
            self.client = genai.Client(vertexai=True, api_key=api_key, http_options=http_options)
        elif auth_mode == "vertex_adc":
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-south1")
            if not project:
                raise ValueError("Set GOOGLE_CLOUD_PROJECT for regional Vertex AI authentication")
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=http_options,
            )
        else:
            raise ValueError("GEMINI_AUTH_MODE must be 'vertex_api_key' or 'vertex_adc'.")

    def _config(self, system: str):
        thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "").upper()
        if thinking_level:
            try:
                thinking_config = self._types.ThinkingConfig(
                    thinking_level=self._types.ThinkingLevel[thinking_level],
                )
            except KeyError as error:
                raise ValueError("GEMINI_THINKING_LEVEL must be MINIMAL, LOW, MEDIUM, or HIGH") from error
        else:
            thinking_config = self._types.ThinkingConfig(
                thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "0")),
            )
        return self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "32")),
            thinking_config=thinking_config,
        )

    def complete(self, system: str, user: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=self._config(system),
        )
        content = response.text
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Vertex Gemini returned an empty text response")
        return content.strip()

    def stream_complete(self, system: str, user: str) -> Iterator[str]:
        started = time.perf_counter()
        self.last_first_token_ms = None
        self.last_stream_ms = None
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=user,
                config=self._config(system),
            )
            for chunk in stream:
                content = chunk.text
                if isinstance(content, str) and content:
                    if self.last_first_token_ms is None:
                        self.last_first_token_ms = (time.perf_counter() - started) * 1000
                    yield content
        finally:
            self.last_stream_ms = (time.perf_counter() - started) * 1000
