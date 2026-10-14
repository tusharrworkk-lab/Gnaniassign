"""Gnani realtime STT adapter and its safe handoff into a conversation turn."""

from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Protocol


PCM_FRAME_BYTES = 1024


@dataclass(frozen=True)
class GnaniProcessing:
    """Gnani VAD has closed a speech segment and transcription has begun."""

    timestamp: str | None = None


@dataclass(frozen=True)
class GnaniTranscript:
    """An immutable, final transcript produced for one Gnani speech segment."""

    text: str
    segment_id: str | None = None
    segment_index: int | None = None
    audio_duration_ms: int | None = None
    latency_ms: int | None = None


@dataclass(frozen=True)
class GnaniError:
    message: str
    timestamp: str | None = None


GnaniEvent = GnaniProcessing | GnaniTranscript | GnaniError


class _TurnSink(Protocol):
    def feed_final(self, phrase: str) -> dict[str, Any]: ...

    def feed_end(self, on_chunk: Callable[[str], None] | None = None) -> dict[str, Any]: ...


class GnaniTurnBridge:
    """Append deduplicated final STT segments to a ``TurnSession``.

    For a voice assistant, Gnani's final transcript is normally the fastest
    usable turn boundary, so ``auto_end_turn`` defaults to ``True``.  A caller
    can opt out to join several final segments into one request.
    """

    def __init__(self, session: _TurnSink, auto_end_turn: bool = True) -> None:
        self.session = session
        self.auto_end_turn = auto_end_turn
        self._seen_segment_ids: set[str] = set()

    def handle(
        self, event: GnaniEvent, on_chunk: Callable[[str], None] | None = None
    ) -> dict[str, Any] | None:
        if not isinstance(event, GnaniTranscript):
            return None
        if event.segment_id and event.segment_id in self._seen_segment_ids:
            return None
        if event.segment_id:
            self._seen_segment_ids.add(event.segment_id)
        pending = self.session.feed_final(event.text)
        return self.end_turn(on_chunk=on_chunk) if self.auto_end_turn else pending

    async def handle_async(
        self, event: GnaniEvent, on_chunk: Callable[[str], None] | None = None
    ) -> dict[str, Any] | None:
        """Handle a final event off the WebSocket event loop.

        The current LangGraph execution is synchronous.  Running it in a
        worker preserves receipt of subsequent STT events and audio I/O while
        the LLM request is in flight.
        """
        return await asyncio.to_thread(self.handle, event, on_chunk)

    def end_turn(self, on_chunk: Callable[[str], None] | None = None) -> dict[str, Any]:
        return self.session.feed_end(on_chunk=on_chunk)


class GnaniRealtimeStt:
    """Async audio stream backed by Gnani's official ``gnani-vachana`` SDK.

    Install it with ``uv sync --extra gnani-stt``.  The import is deliberately
    deferred so text-only deployments do not pay for or require the STT SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        language_code: str | None = None,
        sample_rate: int | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GNANI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GNANI_API_KEY in .env or pass api_key explicitly")
        self.language_code = language_code or os.getenv("GNANI_STT_LANGUAGE", "en-IN")
        self.sample_rate = sample_rate or int(os.getenv("GNANI_STT_SAMPLE_RATE", "16000"))
        if self.sample_rate not in {8000, 16000, 44100, 48000}:
            raise ValueError("GNANI_STT_SAMPLE_RATE must be 8000, 16000, 44100, or 48000")
        self._client: Any | None = None
        self._stream: Any | None = None

    async def __aenter__(self) -> "GnaniRealtimeStt":
        try:
            from gnani.stt import GnaniSTTStreamClient
        except ImportError as error:
            raise RuntimeError("Install Gnani STT support with: uv sync --extra gnani-stt") from error
        self._client = GnaniSTTStreamClient(
            api_key=self.api_key,
            language_code=self.language_code,
            sample_rate=self.sample_rate,
        )
        self._stream = await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, traceback)
        self._stream = None
        self._client = None

    async def send_pcm(self, frame: bytes) -> None:
        """Send precisely one signed-16-bit mono PCM frame to Gnani."""
        if len(frame) != PCM_FRAME_BYTES:
            raise ValueError(f"Gnani requires {PCM_FRAME_BYTES}-byte PCM frames; got {len(frame)} bytes")
        if self._stream is None:
            raise RuntimeError("Enter GnaniRealtimeStt with 'async with' before sending audio")
        await self._stream.send_audio(frame)

    async def events(self) -> AsyncIterator[GnaniEvent]:
        """Yield provider-neutral processing, transcript, and error events."""
        if self._stream is None:
            raise RuntimeError("Enter GnaniRealtimeStt with 'async with' before reading events")
        async for event in self._stream:
            raw = getattr(event, "raw", {}) or {}
            event_name = type(event).__name__
            if event_name == "StreamTranscriptEvent":
                yield GnaniTranscript(
                    text=str(getattr(event, "text", "")),
                    segment_id=raw.get("segment_id"),
                    segment_index=getattr(event, "segment_index", None),
                    audio_duration_ms=getattr(event, "audio_duration_ms", None),
                    latency_ms=getattr(event, "latency", None),
                )
            elif event_name == "StreamProcessingEvent":
                yield GnaniProcessing(timestamp=getattr(event, "timestamp", None))
            elif event_name == "StreamErrorEvent":
                yield GnaniError(message=str(getattr(event, "message", "Unknown Gnani STT error")), timestamp=getattr(event, "timestamp", None))
