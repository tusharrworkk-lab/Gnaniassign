"""Normalized events every STT adapter emits to the voice pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceProcessing:
    """A provider has closed a speech segment and started transcription."""

    timestamp: str | None = None


@dataclass(frozen=True)
class VoiceTranscript:
    """A transcript hypothesis or final result for one speech segment."""

    text: str
    segment_id: str | None = None
    segment_index: int | None = None
    audio_duration_ms: int | None = None
    latency_ms: int | None = None
    is_final: bool = True


@dataclass(frozen=True)
class VoiceError:
    message: str
    timestamp: str | None = None


VoiceEvent = VoiceProcessing | VoiceTranscript | VoiceError
