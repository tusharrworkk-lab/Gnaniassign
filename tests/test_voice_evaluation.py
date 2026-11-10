"""Regression checks for the published voice-agent evaluation definitions."""

import unittest
from types import SimpleNamespace

from gnaniassignment.telephony.evaluation import summarize_turns, summarize_wer
from gnaniassignment.telephony.voicelink import VoiceLinkCall


class VoiceEvaluationTests(unittest.TestCase):
    def test_turn_summary_reports_proxy_latency_and_silence(self) -> None:
        report = summarize_turns([
            {"metrics": {"evaluation_schema_version": 2, "evaluation_finalized": True, "first_outbound_audio_latency_ms": 300, "first_outbound_audio_at_ms": 1300, "reference_turn_end_ms": 1000, "llm_model_ttft_ms": 180, "e2e_llm_ttft_ms": 250, "turn_queue_delay_ms": 70}},
            {"metrics": {"evaluation_schema_version": 2, "evaluation_finalized": True, "first_outbound_audio_latency_ms": 2800, "llm_model_ttft_ms": 400, "e2e_llm_ttft_ms": 1400, "turn_queue_delay_ms": 1000}},
            {"metrics": {"evaluation_schema_version": 2, "evaluation_finalized": True}},
            {"metrics": {"evaluation_finalized": True, "llm_ttft_latency_ms": 99999}},
        ], silent_after_ms=2500)
        self.assertEqual(report["first_audible_reply_proxy"], {"count": 2, "p50_ms": 300.0, "p95_ms": 2800.0, "p99_ms": 2800.0})
        self.assertEqual(report["model_ttft"], {"count": 2, "p50_ms": 180.0, "p95_ms": 400.0, "p99_ms": 400.0})
        self.assertEqual(report["e2e_llm_ttft"], {"count": 2, "p50_ms": 250.0, "p95_ms": 1400.0, "p99_ms": 1400.0})
        self.assertEqual(report["turn_queue_delay"], {"count": 2, "p50_ms": 70.0, "p95_ms": 1000.0, "p99_ms": 1000.0})
        self.assertEqual(report["turns"]["excluded_legacy_measurements"], 1)
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

    async def test_turn_measurement_separates_queue_and_model_ttft(self) -> None:
        call = VoiceLinkCall(
            websocket=SimpleNamespace(), stream_sid="stream-2", call_sid="call-2",
            caller_id="caller-2", customer_id="customer-2",
            stt=SimpleNamespace(sample_rate=16000), tts=SimpleNamespace(sample_rate=16000),
            session=SimpleNamespace(),
        )
        call.last_turn_committed_ms = 100.0
        call.elapsed_ms = lambda: 500.0  # type: ignore[method-assign]
        await call.begin_turn_evaluation()
        call.elapsed_ms = lambda: 800.0  # type: ignore[method-assign]
        call.note_llm_first_token()
        call.elapsed_ms = lambda: 900.0  # type: ignore[method-assign]
        call.note_first_outbound_audio()
        metrics = call.active_turn_evaluation
        self.assertEqual(metrics["turn_queue_delay_ms"], 400.0)
        self.assertEqual(metrics["llm_model_ttft_ms"], 300.0)
        self.assertEqual(metrics["e2e_llm_ttft_ms"], 700.0)
        self.assertEqual(metrics["first_outbound_audio_latency_ms"], 800.0)
