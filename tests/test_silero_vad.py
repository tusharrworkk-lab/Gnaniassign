"""Tests for the optional acoustic endpoint detector."""

from __future__ import annotations

import unittest

import numpy as np

from gnaniassignment.voice.silero_vad import SileroVad
from gnaniassignment.voice.ten_smart_turn import TenPipecatTurnDetector


class _Events:
    def __init__(self) -> None:
        self.responses = iter(({"start": 128}, {"end": 384}))

    def __call__(self, _samples, return_seconds: bool = False):
        self.assert_return_seconds = return_seconds
        return next(self.responses)


class SileroVadTests(unittest.TestCase):
    def test_streaming_wrapper_buffers_frames_and_maps_boundaries(self) -> None:
        # Avoid loading the ONNX model in this deterministic framing test; the
        # production dependency itself is smoke-tested in CI's installed env.
        vad = object.__new__(SileroVad)
        vad._np = np
        vad._torch = type("Torch", (), {"from_numpy": staticmethod(lambda value: value)})
        vad.frame_samples = 256
        vad._buffer = bytearray()
        vad._iterator = _Events()

        self.assertEqual(vad.feed_pcm_s16le(b"\x00\x00" * 255), [])
        boundaries = vad.feed_pcm_s16le(b"\x00\x00" * 257)

        self.assertEqual([item.kind for item in boundaries], ["speech_started", "speech_ended"])
        self.assertEqual([item.sample_offset for item in boundaries], [128, 384])


class _TenVad:
    def __init__(self) -> None:
        self.results = iter(((0.9, 1), (0.1, 0)))

    def process(self, _samples):
        return next(self.results)


class _Semantic:
    def __init__(self) -> None:
        self.frames = []

    def append_audio(self, frame, is_speech):
        self.frames.append((frame, is_speech))

    async def analyze_end_of_turn(self):
        return "complete", type("Metrics", (), {"probability": 0.88})()


class TenSmartTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_acoustic_boundaries_and_semantic_completion_are_separate(self) -> None:
        detector = object.__new__(TenPipecatTurnDetector)
        detector._np = np
        detector._vad = _TenVad()
        detector._hop_samples = 256
        detector._frame_ms = 16
        detector._min_speech_ms = 16
        detector._min_silence_ms = 16
        detector._buffer = bytearray()
        detector._speech_active = False
        detector._raw_speech_ms = 0.0
        detector._raw_silence_ms = 0.0
        detector._revision = 0
        detector._semantic = _Semantic()
        detector._complete_state = "complete"

        events = detector.feed_pcm_s16le(b"\x00\x00" * 512)
        self.assertEqual([event.kind for event in events], ["speech_started", "speech_ended"])
        self.assertEqual(detector.revision, 2)

        result = await detector.analyze_current_pause(detector.revision)
        self.assertTrue(result.complete)
        self.assertEqual(result.probability, 0.88)

