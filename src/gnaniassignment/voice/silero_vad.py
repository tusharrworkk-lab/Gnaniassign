"""Small streaming Silero VAD wrapper for telephony audio.

The detector only establishes acoustic speech boundaries. It does not replace
STT and it never decides what the caller meant; the turn manager still owns
text merging and dialogue state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VoiceActivityBoundary:
    kind: Literal["speech_started", "speech_ended"]
    sample_offset: int | float | None = None


class SileroVad:
    """Incremental 8/16 kHz VAD using the lightweight ONNX CPU model."""

    def __init__(
        self,
        sample_rate: int,
        threshold: float = 0.55,
        min_silence_ms: int = 350,
    ) -> None:
        if sample_rate not in {8000, 16000}:
            raise ValueError("Silero VAD supports 8000 or 16000 Hz audio")
        try:
            import numpy as np
            import torch
            from silero_vad import VADIterator, load_silero_vad
        except ImportError as error:
            raise RuntimeError("Install Silero VAD support with: uv sync --extra silero-vad") from error
        self._np = np
        self._torch = torch
        self.sample_rate = sample_rate
        self.frame_samples = 256 if sample_rate == 8000 else 512
        self._buffer = bytearray()
        model = load_silero_vad(onnx=True)
        self._iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
        )

    def feed_pcm_s16le(self, pcm: bytes) -> list[VoiceActivityBoundary]:
        """Return zero or more boundaries for 16-bit mono little-endian PCM."""
        self._buffer.extend(pcm)
        frame_bytes = self.frame_samples * 2
        boundaries: list[VoiceActivityBoundary] = []
        while len(self._buffer) >= frame_bytes:
            frame = bytes(self._buffer[:frame_bytes])
            del self._buffer[:frame_bytes]
            samples = self._np.frombuffer(frame, dtype=self._np.int16).astype(self._np.float32) / 32768.0
            result = self._iterator(self._torch.from_numpy(samples), return_seconds=False)
            if not isinstance(result, dict):
                continue
            if "start" in result:
                boundaries.append(VoiceActivityBoundary("speech_started", result.get("start")))
            if "end" in result:
                boundaries.append(VoiceActivityBoundary("speech_ended", result.get("end")))
        return boundaries
