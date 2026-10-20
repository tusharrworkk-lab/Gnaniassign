"""Gnani realtime STT adapter and its safe handoff into a conversation turn."""

from __future__ import annotations

import os
import asyncio
import inspect
import json
import time
from dataclasses import dataclass
from collections.abc import AsyncIterable, Awaitable
from typing import Any, AsyncIterator, Callable, Protocol


GNANI_STT_ENDPOINT = "wss://api.vachana.ai/stt/v3/stream"
PCM_FRAME_BYTES = 1024
PCM_SAMPLES_PER_FRAME = 512


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
        # VAD can occasionally close on initial room noise. A final event
        # without words must never create an empty intent/LLM request.
        if not event.text.strip():
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
    """Raw Gnani WebSocket STT stream with configurable VAD.

    The raw transport is intentional: Gnani's Python SDK currently does not
    expose VAD headers. Audio senders must provide signed 16-bit, mono,
    little-endian PCM frames and preserve the frame cadence for their sample
    rate.
    """

    def __init__(
        self,
        api_key: str | None = None,
        language_code: str | None = None,
        sample_rate: int | None = None,
        endpoint: str | None = None,
        vad_threshold: float | None = None,
        min_silence_ms: int | None = None,
        min_speech_ms: int | None = None,
        min_segment_ms: int | None = None,
        connect: Callable[[str, dict[str, str]], Awaitable[Any]] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GNANI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GNANI_API_KEY in .env or pass api_key explicitly")
        self.language_code = language_code or os.getenv("GNANI_STT_LANGUAGE", "en-IN")
        self.sample_rate = sample_rate or int(os.getenv("GNANI_STT_SAMPLE_RATE", "16000"))
        if self.sample_rate not in {8000, 16000, 44100, 48000}:
            raise ValueError("GNANI_STT_SAMPLE_RATE must be 8000, 16000, 44100, or 48000")
        self.endpoint = endpoint or os.getenv("GNANI_STT_ENDPOINT", GNANI_STT_ENDPOINT)
        self.vad_threshold = vad_threshold if vad_threshold is not None else float(os.getenv("GNANI_VAD_THRESHOLD", "0.55"))
        self.min_silence_ms = min_silence_ms if min_silence_ms is not None else int(os.getenv("GNANI_MIN_SILENCE_MS", "100"))
        self.min_speech_ms = min_speech_ms if min_speech_ms is not None else int(os.getenv("GNANI_MIN_SPEECH_MS", "50"))
        self.min_segment_ms = min_segment_ms if min_segment_ms is not None else int(os.getenv("GNANI_MIN_SEGMENT_MS", "100"))
        if not 0.0 <= self.vad_threshold <= 1.0:
            raise ValueError("GNANI_VAD_THRESHOLD must be between 0.0 and 1.0")
        if not 100 <= self.min_silence_ms <= 2000:
            raise ValueError("GNANI_MIN_SILENCE_MS must be between 100 and 2000")
        if not 50 <= self.min_speech_ms <= 2000:
            raise ValueError("GNANI_MIN_SPEECH_MS must be between 50 and 2000")
        if not 100 <= self.min_segment_ms <= 1000:
            raise ValueError("GNANI_MIN_SEGMENT_MS must be between 100 and 1000")
        self._connect = connect
        self._stream: Any | None = None
        self.active_sample_rate: int | None = None

    @property
    def frame_interval_seconds(self) -> float:
        """Natural cadence for one 512-sample PCM frame."""
        return PCM_SAMPLES_PER_FRAME / self.sample_rate

    @property
    def headers(self) -> dict[str, str]:
        """The documented WebSocket upgrade headers for this connection."""
        return {
            "x-api-key-id": self.api_key,
            "lang_code": self.language_code,
            "x-sample-rate": str(self.sample_rate),
            "x-format": os.getenv("GNANI_STT_FORMAT", "verbatim"),
            "x-vad-threshold": str(self.vad_threshold),
            "x-min-silence-ms": str(self.min_silence_ms),
            "x-min-speech-ms": str(self.min_speech_ms),
            "x-min-segment-ms": str(self.min_segment_ms),
        }

    async def __aenter__(self) -> "GnaniRealtimeStt":
        if self._connect is not None:
            self._stream = await self._connect(self.endpoint, self.headers)
        else:
            try:
                import websockets
            except ImportError as error:
                raise RuntimeError("Install Gnani STT support with: uv sync --extra gnani-stt") from error
            header_argument = "additional_headers" if "additional_headers" in inspect.signature(websockets.connect).parameters else "extra_headers"
            self._stream = await websockets.connect(self.endpoint, **{header_argument: self.headers})
        await self._confirm_connected()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._stream is not None:
            await self._stream.close()
        self._stream = None

    async def _confirm_connected(self) -> None:
        """Confirm Gnani accepted the requested sample rate before audio flows."""
        if self._stream is None:
            raise RuntimeError("Gnani stream was not connected")
        try:
            payload = await asyncio.wait_for(self._stream.recv(), timeout=10.0)
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            event = json.loads(payload)
        except (TimeoutError, TypeError, json.JSONDecodeError) as error:
            await self._stream.close()
            self._stream = None
            raise RuntimeError("Gnani did not send a valid connected event") from error
        if event.get("type") != "connected":
            await self._stream.close()
            self._stream = None
            raise RuntimeError("Gnani did not acknowledge the STT connection")
        active_rate = event.get("config", {}).get("sample_rate")
        if active_rate != self.sample_rate:
            await self._stream.close()
            self._stream = None
            raise RuntimeError(
                f"Gnani acknowledged {active_rate} Hz, not requested {self.sample_rate} Hz; "
                "refusing to send mismatched PCM audio"
            )
        self.active_sample_rate = active_rate

    async def send_pcm(self, frame: bytes) -> None:
        """Send precisely one signed-16-bit mono PCM frame to Gnani."""
        if len(frame) != PCM_FRAME_BYTES:
            raise ValueError(f"Gnani requires {PCM_FRAME_BYTES}-byte PCM frames; got {len(frame)} bytes")
        if self._stream is None:
            raise RuntimeError("Enter GnaniRealtimeStt with 'async with' before sending audio")
        await self._stream.send(frame)

    async def send_frames(self, frames: AsyncIterable[bytes], realtime_pace: bool = True) -> None:
        """Send audio frames, pacing file/replay input at the required cadence.

        Live microphone/call sources normally arrive at the natural cadence; set
        ``realtime_pace=False`` for those sources to avoid adding a second wait.
        """
        next_frame_at = time.monotonic()
        async for frame in frames:
            await self.send_pcm(frame)
            if realtime_pace:
                next_frame_at += self.frame_interval_seconds
                await asyncio.sleep(max(0.0, next_frame_at - time.monotonic()))

    async def events(self) -> AsyncIterator[GnaniEvent]:
        """Yield provider-neutral processing, transcript, and error events."""
        if self._stream is None:
            raise RuntimeError("Enter GnaniRealtimeStt with 'async with' before reading events")
        async for payload in self._stream:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            try:
                raw = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                yield GnaniError(message="Gnani returned a non-JSON server event")
                continue
            event_name = raw.get("type")
            if event_name == "transcript":
                yield GnaniTranscript(
                    text=str(raw.get("text", "")),
                    segment_id=raw.get("segment_id"),
                    segment_index=raw.get("segment_index"),
                    audio_duration_ms=raw.get("audio_duration_ms"),
                    latency_ms=raw.get("latency"),
                )
            elif event_name == "processing":
                yield GnaniProcessing(timestamp=raw.get("timestamp"))
            elif event_name == "error":
                yield GnaniError(message=str(raw.get("message", "Unknown Gnani STT error")), timestamp=raw.get("timestamp"))
