"""Regression checks for the published voice-agent evaluation definitions."""

import unittest
from types import SimpleNamespace

from gnaniassignment.telephony.evaluation import summarize_turns, summarize_wer
from gnaniassignment.telephony.voicelink import VoiceLinkCall


class VoiceEvaluationTests(unittest.TestCase):
    def test_turn_summary_reports_proxy_latency_and_silence(self) -> None:
        report = summarize_turns([
            {"metrics": {"evaluation_finalized": True, "first_outbound_audio_latency_ms": 300, "first_outbound_audio_at_ms": 1300, "reference_turn_end_ms": 1000}},
            {"metrics": {"evaluation_finalized": True, "first_outbound_audio_latency_ms": 2800}},
            {"metrics": {"evaluation_finalized": True}},
        ], silent_after_ms=2500)
        self.assertEqual(report["first_audible_reply_proxy"], {"count": 2, "p50_ms": 300.0, "p95_ms": 2800.0, "p99_ms": 2800.0})
        self.assertEqual(report["silent_turn_rate"], 0.6667)
        self.assertEqual(report["premature_turn_rate"], 0.0)
        self.assertEqual(report["barge_in_stop_latency"]["count"], 0)

    def test_hinglish_wer_counts_word_errors(self) -> None:
        report = summarize_wer([
            {"reference_text": "kal meri party hai chhah baje", "hypothesis_text": "kal meri party hai chhe baje"},
            {"reference_text": "haan theek hai", "hypothesis_text": "haan theek hai"},
        ])
        self.assertEqual(report, {"samples": 2, "reference_words": 9, "errors": 1, "wer": 0.1111})


class VoiceInterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_barge_in_sends_clear_and_records_stop_latency(self) -> None:
        class Socket:
            def __init__(self) -> None:
                self.messages: list[dict[str, object]] = []

            async def send_json(self, payload: dict[str, object]) -> None:
                self.messages.append(payload)

        socket = Socket()
        call = VoiceLinkCall(
            websocket=socket,
            stream_sid="stream-1",
            call_sid="call-1",
            caller_id="caller-1",
            customer_id="customer-1",
            stt=SimpleNamespace(sample_rate=16000),
            tts=SimpleNamespace(sample_rate=16000),
            session=SimpleNamespace(),
        )
        call.reply_in_progress = True
        await call.begin_turn_evaluation()

        self.assertTrue(await call.interrupt_active_reply())
        self.assertTrue(call.reply_interrupted.is_set())
        self.assertEqual(socket.messages, [{"event": "clear", "stream_sid": "stream-1"}])
        self.assertIsNotNone(call.active_turn_evaluation)
        self.assertGreaterEqual(call.active_turn_evaluation["barge_in_stop_latency_ms"], 0)
