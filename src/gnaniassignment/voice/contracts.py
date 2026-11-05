"""Stable contracts for pluggable realtime speech providers."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, Protocol, Self

from gnaniassignment.voice.events import VoiceEvent


class RealtimeSttProvider(Protocol):
    """PCM-in, normalized final-transcript-events-out speech recognizer."""

    sample_rate: int
    frame_bytes: int
    min_silence_ms: int

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...

    async def send_frames(self, frames: AsyncIterable[bytes], realtime_pace: bool = True) -> None: ...

    async def events(self) -> AsyncIterator[VoiceEvent]: ...


class RealtimeTtsProvider(Protocol):
    """Text-in, raw PCM-out synthesizer used by the playback layer."""

    sample_rate: int
    last_first_audio_ms: float | None
    last_synthesis_ms: float | None
    last_audio_chunks: int
    last_audio_bytes: int

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...

    async def synthesize(
        self, text: str, *, add_clear_throat: bool = False, latency_giggle: bool = False
    ) -> AsyncIterator[Any]: ...
