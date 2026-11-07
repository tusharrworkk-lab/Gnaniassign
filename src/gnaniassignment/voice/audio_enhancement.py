"""Optional, low-latency inbound audio enhancement for telephone calls.

This module intentionally sits before VAD and STT.  It never processes bot
TTS/output audio, and it exposes metadata only—never raw customer audio—for
debugging and rollout evaluation.
"""

from __future__ import annotations

import audioop
import math
import os
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AudioEnhancementResult:
    audio: bytes
    input_rms_dbfs: float
    output_rms_dbfs: float
    speech_probability: float | None
    processing_ms: float


class AudioEnhancer(Protocol):
    mode: str
    backend: str

    def process_pcm_s16le(self, pcm: bytes) -> AudioEnhancementResult: ...


def _rms_dbfs(pcm: bytes) -> float:
    if not pcm:
        return -120.0
    rms = audioop.rms(pcm, 2)
    return round(20 * math.log10(max(rms, 1) / 32768), 1)


class WebRtcAudioEnhancer:
    """WebRTC APM high-pass + noise suppression adapter.

    WebRTC APM processes 16 kHz mono PCM in its native code.  Echo cancellation
    deliberately remains off: VoiceLink supplies independent inbound and
    outbound telephony tracks, so there is no reliable far-end reference for
    a server-side AEC.
    """

    backend = "webrtc_apm"

    def __init__(self, sample_rate: int, mode: str, level: int) -> None:
        if sample_rate not in {16000, 32000, 48000}:
            raise ValueError("WebRTC audio enhancement requires 16000, 32000, or 48000 Hz PCM")
        if mode not in {"shadow", "enabled"}:
            raise ValueError("Audio enhancement mode must be shadow or enabled")
        if not 0 <= level <= 3:
            raise ValueError("VOICE_LINK_NS_LEVEL must be between 0 and 3")
        try:
            from pywebrtc_audio import AudioProcessor
        except ImportError as error:
            raise RuntimeError(
                "Install audio enhancement support with: uv sync --extra audio-enhancement"
            ) from error
        self.mode = mode
        self._processor = AudioProcessor(
            sample_rate=sample_rate,
            num_channels=1,
            echo_cancellation=False,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=False,
            ns_level=level,
        )

    def process_pcm_s16le(self, pcm: bytes) -> AudioEnhancementResult:
        if len(pcm) % 2:
            raise ValueError("16-bit PCM must contain an even number of bytes")
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Audio enhancement requires NumPy") from error
        started = time.perf_counter()
        input_rms = _rms_dbfs(pcm)
        # A writable copy is intentional: the extension may use a mutable
        # NumPy buffer internally.
        incoming = np.frombuffer(pcm, dtype="<i2").copy()
        processed = self._processor.process(incoming)
        output = np.asarray(processed, dtype="<i2").tobytes()
        if len(output) != len(pcm):
            raise RuntimeError("WebRTC audio enhancement changed PCM frame length")
        probability = getattr(self._processor, "speech_probability", None)
        speech_probability = float(probability) if isinstance(probability, (int, float)) else None
        return AudioEnhancementResult(
            audio=output if self.mode == "enabled" else pcm,
            input_rms_dbfs=input_rms,
            output_rms_dbfs=_rms_dbfs(output),
            speech_probability=speech_probability,
            processing_ms=round((time.perf_counter() - started) * 1000, 3),
        )


def create_audio_enhancer(sample_rate: int) -> AudioEnhancer | None:
    """Create the configured inbound-only enhancement stage.

    `shadow` runs enhancement only for metrics and returns original PCM.
    `enabled` returns cleaned PCM to both turn detection and STT.
    """
    mode = os.getenv("VOICE_LINK_AUDIO_ENHANCEMENT", "off").strip().lower()
    if mode == "off":
        return None
    if mode not in {"shadow", "enabled"}:
        raise ValueError("VOICE_LINK_AUDIO_ENHANCEMENT must be off, shadow, or enabled")
    backend = os.getenv("VOICE_LINK_NS_BACKEND", "webrtc_apm").strip().lower()
    if backend != "webrtc_apm":
        raise ValueError("VOICE_LINK_NS_BACKEND must be webrtc_apm")
    return WebRtcAudioEnhancer(
        sample_rate=sample_rate,
        mode=mode,
        level=int(os.getenv("VOICE_LINK_NS_LEVEL", "1")),
    )
