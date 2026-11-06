"""TEN acoustic VAD plus Pipecat Smart Turn semantic endpointing.

TEN runs on every short 16 kHz frame. Smart Turn v3 is deliberately invoked
only after TEN reports a speech-to-silence boundary, using the most recent
eight seconds of the caller's current turn. This avoids treating a natural
mid-sentence pause as a final user turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TurnBoundary:
    kind: Literal["speech_started", "speech_ended"]
    probability: float


@dataclass(frozen=True)
class SemanticTurnDecision:
    complete: bool
    probability: float | None
    stale: bool = False


class TenPipecatTurnDetector:
    """Streaming acoustic detector with an audio-native semantic endpoint gate."""

    def __init__(self, threshold: float = 0.5, hop_samples: int = 256) -> None:
        try:
            import numpy as np
            from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
            from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
            from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
            from ten_vad import TenVad
        except ImportError as error:
            raise RuntimeError(
                "Install TURN detection support with: uv sync --extra turn-detection"
            ) from error
        if hop_samples not in {160, 256}:
            raise ValueError("TEN VAD hop size must be 160 or 256 samples at 16 kHz")
        self._np = np
        self._complete_state = EndOfTurnState.COMPLETE
        self._vad = TenVad(hop_samples, threshold)
        self._hop_samples = hop_samples
        self._buffer = bytearray()
        self._speech_active = False
        self._revision = 0
        self._semantic = LocalSmartTurnAnalyzerV3(
            sample_rate=16000,
            cpu_count=1,
            params=SmartTurnParams(max_duration_secs=8),
        )

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def revision(self) -> int:
        return self._revision

    def feed_pcm_s16le(self, pcm_16k: bytes) -> list[TurnBoundary]:
        """Feed mono 16-bit 16 kHz PCM and return acoustic transitions."""
        self._buffer.extend(pcm_16k)
        boundaries: list[TurnBoundary] = []
        hop_bytes = self._hop_samples * 2
        while len(self._buffer) >= hop_bytes:
            frame = bytes(self._buffer[:hop_bytes])
            del self._buffer[:hop_bytes]
            samples = self._np.frombuffer(frame, dtype=self._np.int16)
            probability, flag = self._vad.process(samples)
            speaking = bool(flag)
            self._semantic.append_audio(frame, is_speech=speaking)
            if speaking == self._speech_active:
                continue
            self._speech_active = speaking
            self._revision += 1
            boundaries.append(TurnBoundary("speech_started" if speaking else "speech_ended", probability))
        return boundaries

    async def analyze_current_pause(self, revision: int) -> SemanticTurnDecision:
        """Ask Smart Turn whether the acoustic pause completes this utterance.

        A response is ignored when fresh caller speech began while the semantic
        model was running; audio is then retained for a later, richer decision.
        """
        if self._speech_active or revision != self._revision:
            return SemanticTurnDecision(False, None, stale=True)
        state, metrics = await self._semantic.analyze_end_of_turn()
        if self._speech_active or revision != self._revision:
            return SemanticTurnDecision(False, None, stale=True)
        probability = getattr(metrics, "probability", None) if metrics is not None else None
        return SemanticTurnDecision(state == self._complete_state, probability)

    async def close(self) -> None:
        await self._semantic.cleanup()
