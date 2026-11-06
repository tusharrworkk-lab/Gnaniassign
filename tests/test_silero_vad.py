"""Tests for the optional acoustic endpoint detector."""

from __future__ import annotations

import unittest

import numpy as np

from gnaniassignment.voice.silero_vad import SileroVad


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

