"""Gnani realtime TTS transport for low-latency voice replies."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any


GNANI_TTS_ENDPOINT = "wss://api.vachana.ai/api/v1/tts"
_INLINE_TAG = re.compile(r"<[^>]{1,80}>")
_POSITIVE_REPLY = re.compile(r"\b(shukriya|dhanyavaad|good news|acchi khabar|khushi|congratulations)\b", re.I)


def should_add_clear_throat(ttft_ms: float | int | None, threshold_ms: float = 700) -> bool:
    """Deterministic latency rule; no LLM or judge participates in this decision."""
    return isinstance(ttft_ms, (int, float)) and ttft_ms > threshold_ms


def decorate_tts_text(text: str, *, enabled: bool, model: str, add_clear_throat: bool = False, latency_giggle: bool = False) -> str:
    """Add only application-controlled Timbre v3 audio tags to spoken text.

    Stored transcripts stay unmodified. Any markup emitted by an LLM is removed
    so that only this allowlisted formatter controls voice delivery.
    """
    clean = _INLINE_TAG.sub("", text).strip()
    if not enabled or model != "timbre-v3.0":
        return clean
    # This cue is sent before any LLM text exists. The normal reply is
    # synthesized separately once generation completes.
    if latency_giggle:
        return "<giggle>"
    # A proactive latency cue deliberately has no accompanying answer text.
    # It is synthesized as a standalone request while the LLM is still
    # working, never prepended to a reply that has already arrived.
    if not clean:
        if add_clear_throat:
            return "<clears throat>"
        return clean
    tags: list[str] = []
    # The post-response latency cue is application-controlled, never LLM
    # markup. It remains separate from the proactive pre-token giggle cue.
    if add_clear_throat:
        tags.append("<clears throat>")
    # Keep the expressive catalog intentionally narrow for retail support.
    if _POSITIVE_REPLY.search(clean) and "<giggle>" not in tags:
        tags.append("<giggle>")
    return " ".join([*tags, clean])


@dataclass(frozen=True)
class GnaniAudioChunk:
    """One decoded audio block, ready for an audio output device."""

    audio: bytes
    chunk_index: int
    is_final: bool = False


class GnaniTtsError(RuntimeError):
    """A recoverable or terminal error returned by Gnani TTS."""


class GnaniRealtimeTts:
    """One persistent Gnani WebSocket which streams decoded TTS audio.

    The defaults generate 24 kHz, mono, signed-16-bit raw PCM. This avoids
    container buffering and lets a caller write each yielded block directly to
    its speaker or telephony output. ``synthesize`` calls must not overlap.
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        voice: str | None = None,
        model: str | None = None,
        language: str | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
        connect: Callable[[str, dict[str, str]], Awaitable[Any]] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GNANI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GNANI_API_KEY in .env or pass api_key explicitly")
        self.endpoint = endpoint or os.getenv("GNANI_TTS_ENDPOINT", GNANI_TTS_ENDPOINT)
        self.voice = voice or os.getenv("GNANI_TTS_VOICE", "Shivani")
        self.model = model or os.getenv("GNANI_TTS_MODEL", "timbre-v3.0")
        self.language = language or os.getenv("GNANI_TTS_LANGUAGE", "hi-en")
        self.speed = speed if speed is not None else float(os.getenv("GNANI_TTS_SPEED", "1.0"))
        self.audio_tags = os.getenv("GNANI_TTS_AUDIO_TAGS", "false").lower() in {"1", "true", "yes", "on"}
        self.sample_rate = sample_rate or int(os.getenv("GNANI_TTS_SAMPLE_RATE", "24000"))
        if self.model not in {"timbre-v2.5", "timbre-v3.0"}:
            raise ValueError("GNANI_TTS_MODEL must be 'timbre-v2.5' or 'timbre-v3.0'")
        if self.language not in {"auto", "hi-IN", "en-IN", "hi-en"}:
            raise ValueError("GNANI_TTS_LANGUAGE must be auto, hi-IN, en-IN, or hi-en")
        if not 0.85 <= self.speed <= 1.15:
            raise ValueError("GNANI_TTS_SPEED must be between 0.85 and 1.15")
        if self.sample_rate not in {8000, 16000, 22050, 24000, 44100, 48000}:
            raise ValueError("GNANI_TTS_SAMPLE_RATE is unsupported")
        self._connect = connect
        self._stream: Any | None = None
        self._request_lock = asyncio.Lock()
        self.last_first_audio_ms: float | None = None
        self.last_synthesis_ms: float | None = None
        self.last_audio_chunks = 0
        self.last_audio_bytes = 0

    async def _open_stream(self) -> None:
        """Open a fresh provider connection.

        A telephone call can sit silent for minutes between replies.  Gnani's
        endpoint may close an idle WebSocket in that interval, so this helper
        is also used by ``synthesize`` to recover before a new request.
        """
        if self._connect is not None:
            self._stream = await self._connect(self.endpoint, self.headers)
            return
        try:
            import websockets
        except ImportError as error:
            raise RuntimeError("Install voice WebSocket support with: uv sync --extra gnani-stt") from error
        header_name = "additional_headers" if "additional_headers" in inspect.signature(websockets.connect).parameters else "extra_headers"
        self._stream = await websockets.connect(self.endpoint, **{header_name: self.headers})

    async def _reconnect_if_closed(self) -> None:
        if self._stream is None or bool(getattr(self._stream, "closed", False)):
            if self._stream is not None:
                try:
                    await self._stream.close()
                except Exception:
                    # The old connection is already unusable; opening a new
                    # one is the recovery path.
                    pass
            await self._open_stream()

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key-ID": self.api_key, "Content-Type": "application/json"}

    def request(self, text: str, *, add_clear_throat: bool = False, latency_giggle: bool = False) -> dict[str, Any]:
        if not text.strip() and not (add_clear_throat or latency_giggle):
            raise ValueError("TTS text must not be empty")
        rendered = decorate_tts_text(text, enabled=self.audio_tags, model=self.model, add_clear_throat=add_clear_throat, latency_giggle=latency_giggle)
        if not rendered:
            raise ValueError("TTS latency cue requires GNANI_TTS_AUDIO_TAGS=true and timbre-v3.0")
        return {
            "text": rendered,
            "voice": self.voice,
            "model": self.model,
            "language": self.language,
            "speed": self.speed,
            "audio_config": {
                "sample_rate": self.sample_rate,
                "num_channels": 1,
                "sample_width": 2,
                "encoding": "linear_pcm",
                "container": "raw",
            },
        }

    async def __aenter__(self) -> "GnaniRealtimeTts":
        await self._open_stream()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._stream is not None:
            await self._stream.close()
        self._stream = None

    async def synthesize(self, text: str, *, add_clear_throat: bool = False, latency_giggle: bool = False) -> AsyncIterator[GnaniAudioChunk]:
        """Send one text request and yield decoded audio in provider order."""
        if self._stream is None:
            raise RuntimeError("Enter GnaniRealtimeTts with 'async with' before synthesis")
        async with self._request_lock:
            await self._reconnect_if_closed()
            started = time.perf_counter()
            self.last_first_audio_ms = None
            self.last_synthesis_ms = None
            self.last_audio_chunks = 0
            self.last_audio_bytes = 0
            data_complete_seen = False
            try:
                request = json.dumps(self.request(text, add_clear_throat=add_clear_throat, latency_giggle=latency_giggle))
                try:
                    await self._stream.send(request)
                except Exception as error:
                    # A close discovered at send time is the normal symptom of
                    # an idle provider socket.  No reply audio has been yielded
                    # yet, so a single reconnect-and-resend cannot duplicate it.
                    if not type(error).__name__.startswith("ConnectionClosed"):
                        raise
                    await self._open_stream()
                    await self._stream.send(request)
                while True:
                    try:
                        if data_complete_seen:
                            # Production currently emits a second envelope
                            # `complete` after its documented data-level
                            # `complete`. Drain it without hanging against a
                            # server implementation that sends only one.
                            raw_message = await asyncio.wait_for(self._stream.recv(), timeout=0.25)
                        else:
                            raw_message = await self._stream.recv()
                    except asyncio.TimeoutError:
                        return
                    except Exception as error:
                        if data_complete_seen and type(error).__name__.startswith("ConnectionClosed"):
                            return
                        raise
                    if isinstance(raw_message, bytes):
                        raw_message = raw_message.decode("utf-8")
                    try:
                        message = json.loads(raw_message)
                    except (TypeError, json.JSONDecodeError) as error:
                        raise GnaniTtsError("Gnani TTS returned a non-JSON message") from error
                    event_type = message.get("type")
                    if event_type == "start":
                        continue
                    if event_type == "error":
                        raise GnaniTtsError(str(message.get("message", "Unknown Gnani TTS error")))
                    if event_type not in {"audio", "complete"}:
                        continue
                    data = message.get("data")
                    # Gnani's terminal `complete` event may legitimately
                    # contain no audio/data object after the final audio
                    # frame. Treat it as the normal end of synthesis.
                    if event_type == "complete" and not isinstance(data, dict):
                        return
                    if not isinstance(data, dict):
                        raise GnaniTtsError("Gnani TTS audio message has no data object")
                    final = bool(data.get("is_final", False))
                    encoded_audio = data.get("audio", "")
                    if not isinstance(encoded_audio, str):
                        raise GnaniTtsError("Gnani TTS audio payload is invalid")
                    try:
                        audio = base64.b64decode(encoded_audio, validate=True) if encoded_audio else b""
                    except ValueError as error:
                        raise GnaniTtsError("Gnani TTS audio payload is not valid base64") from error
                    if audio:
                        if self.last_first_audio_ms is None:
                            self.last_first_audio_ms = (time.perf_counter() - started) * 1000
                        self.last_audio_chunks += 1
                        self.last_audio_bytes += len(audio)
                        yield GnaniAudioChunk(audio, int(data.get("chunk_index", 0)), final)
                    if event_type == "complete":
                        data_complete_seen = True
            finally:
                self.last_synthesis_ms = (time.perf_counter() - started) * 1000
