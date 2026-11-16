"""Tests for the lightweight operations-dashboard runtime primitives."""

from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from gnaniassignment.api.server import RuntimeEvents, SessionRegistry, create_app
from gnaniassignment.prompts.tree import PromptNode, PromptPlan, _validate_condition
from gnaniassignment.voice.events import VoiceProcessing, VoiceTranscript


class DashboardRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_events_fan_out_to_each_subscriber(self) -> None:
        events = RuntimeEvents()
        first = events.stream()
        second = events.stream()
        await anext(first)
        await anext(second)

        await events.publish("classifier.completed", label="track_order", confidence=0.91)
        first_payload = (await anext(first)).decode()
        second_payload = (await anext(second)).decode()

        self.assertIn('"kind":"classifier.completed"', first_payload)
        self.assertIn('"label":"track_order"', second_payload)
        await first.aclose()
        await second.aclose()


class PromptConditionTests(unittest.TestCase):
    def test_prompt_conditions_only_allow_runtime_context_keys(self) -> None:
        _validate_condition({"language": ["hinglish", "english"], "channel": "voice", "customer_id": "caller-42"})
        with self.assertRaises(ValueError):
            _validate_condition({"caller_text": "refund"})

    def test_customer_specific_rule_is_never_rendered_for_another_caller(self) -> None:
        plan = PromptPlan(
            tenant_id="default",
            agent_id="generic",
            version=1,
            nodes=(
                PromptNode("base", None, {}, "Base rule", 10),
                PromptNode("private", None, {"customer_id": "caller-42"}, "Private rule", 20),
            ),
        )
        self.assertEqual(plan.render({"customer_id": "caller-42"}), "Base rule\n\nPrivate rule")
        self.assertEqual(plan.render({"customer_id": "caller-99"}), "Base rule")


class HarnessApiTests(unittest.TestCase):
    def test_harness_keeps_turn_history_and_rejects_ended_conversations(self) -> None:
        class FakeSession:
            def __init__(self, customer_id: str) -> None:
                self.customer_id = customer_id
                self.working_memory: dict[str, str] = {}
                self.call_turns: list[SimpleNamespace] = []

            def ask(self, text: str) -> dict[str, object]:
                self.working_memory["last_user_text"] = text
                self.call_turns.append(SimpleNamespace(
                    query=text, answer=f"reply: {text}", intent="generic", topic="general"
                ))
                return {"spoken": f"reply: {text}", "intent": "generic", "topic": "general", "metrics": {"agent_ms": 1}}

            def finalize_call(self) -> None:
                return None

        def fake_create_session(*, customer_id: str, **_kwargs: object) -> FakeSession:
            return FakeSession(customer_id)

        environment = {"AGENT_API_KEY": "test-key", "AGENT_ENVIRONMENT": "test"}
        with patch.dict(os.environ, environment, clear=False), patch("gnaniassignment.api.server.create_session", fake_create_session):
            client = TestClient(create_app(SessionRegistry()))
            created = client.post("/v1/harness/conversations", headers={"X-API-Key": "test-key"}, json={"customer_id": "eval-user"})
            self.assertEqual(created.status_code, 201)
            conversation_id = created.json()["conversation_id"]

            turn = client.post(f"/v1/harness/conversations/{conversation_id}/turn", headers={"X-API-Key": "test-key"}, json={"text": "hello"})
            self.assertEqual(turn.status_code, 200)
            self.assertEqual(turn.json()["answer"], "reply: hello")
            self.assertEqual(turn.json()["turns"][0]["user"], "hello")

            ended = client.post(f"/v1/harness/conversations/{conversation_id}/end", headers={"X-API-Key": "test-key"})
            self.assertEqual(ended.status_code, 200)
            self.assertEqual(ended.json()["ended"], True)
            self.assertEqual(client.get(f"/v1/harness/conversations/{conversation_id}", headers={"X-API-Key": "test-key"}).status_code, 404)


class SttLlmDiagnosticApiTests(unittest.TestCase):
    def test_diagnostic_websocket_is_feature_gated(self) -> None:
        environment = {
            "AGENT_API_KEY": "test-key",
            "AGENT_ENVIRONMENT": "production",
            "STT_LLM_DIAGNOSTIC_ENABLED": "false",
        }
        with patch.dict(os.environ, environment, clear=False):
            client = TestClient(create_app(SessionRegistry()))
            with self.assertRaises(WebSocketDisconnect) as rejected:
                with client.websocket_connect(
                    "/v1/diagnostics/stt-llm",
                    headers={"X-API-Key": "test-key"},
                ):
                    pass
            self.assertEqual(rejected.exception.code, 1008)

    def test_diagnostic_websocket_requires_api_key_and_reaches_handler(self) -> None:
        async def fake_handler(websocket: object) -> None:
            await websocket.accept()
            await websocket.send_json({"type": "diagnostic.ready"})
            await websocket.close()

        environment = {
            "AGENT_API_KEY": "test-key",
            "AGENT_ENVIRONMENT": "production",
            "STT_LLM_DIAGNOSTIC_ENABLED": "true",
        }
        target = "gnaniassignment.api.stt_llm_diagnostic.handle_stt_llm_diagnostic_websocket"
        with patch.dict(os.environ, environment, clear=False), patch(target, fake_handler):
            client = TestClient(create_app(SessionRegistry()))
            with self.assertRaises(WebSocketDisconnect) as rejected:
                with client.websocket_connect(
                    "/v1/diagnostics/stt-llm",
                    headers={"X-API-Key": "wrong-key"},
                ):
                    pass
            self.assertEqual(rejected.exception.code, 1008)

            with client.websocket_connect(
                "/v1/diagnostics/stt-llm",
                headers={"X-API-Key": "test-key"},
            ) as websocket:
                self.assertEqual(websocket.receive_json(), {"type": "diagnostic.ready"})

    def test_diagnostic_streams_transcript_answer_and_metrics_without_tts(self) -> None:
        class FakeStt:
            model = "fake-stt"
            language_code = "hi-IN"
            sample_rate = 16000
            frame_bytes = 1024
            min_silence_ms = 400

            def __init__(self) -> None:
                self.frame_received = asyncio.Event()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args: object) -> None:
                return None

            async def send_pcm(self, frame: bytes) -> None:
                self.frame_received.set()

            async def events(self):
                await self.frame_received.wait()
                yield VoiceProcessing()
                yield VoiceTranscript(
                    text="Tushar free hai?",
                    segment_id="segment-1",
                    latency_ms=81,
                )
                await asyncio.Future()

        class FakeSession:
            def record_stt_segment(self, *_args: object) -> None:
                return None

            def feed_final(self, phrase: str, *, detected_language_code: str | None = None):
                return {"pending_text": phrase}

            def feed_end(self, on_chunk=None):
                if on_chunk is not None:
                    on_chunk("Main check karti hoon.")
                return {
                    "spoken": "Main check karti hoon.",
                    "intent": "generic",
                    "topic": "availability",
                    "metrics": {
                        "account_context_ms": 2.0,
                        "router_ms": 3.0,
                        "context_assembly_ms": 4.0,
                        "retrieval_ms": 5.0,
                        "prompt_build_ms": 6.0,
                        "provider_ttft_ms": 40.0,
                        "answer_decode_ttft_ms": 48.0,
                        "llm_total_stream_ms": 75.0,
                    },
                }

        environment = {
            "AGENT_API_KEY": "test-key",
            "AGENT_ENVIRONMENT": "production",
            "STT_LLM_DIAGNOSTIC_ENABLED": "true",
        }
        module = "gnaniassignment.api.stt_llm_diagnostic"
        with (
            patch.dict(os.environ, environment, clear=False),
            patch(f"{module}.create_session", return_value=FakeSession()),
            patch(f"{module}.create_stt_provider", return_value=FakeStt()),
        ):
            client = TestClient(create_app(SessionRegistry()))
            with client.websocket_connect(
                "/v1/diagnostics/stt-llm",
                headers={"X-API-Key": "test-key"},
            ) as websocket:
                websocket.send_json({"type": "start", "customer_id": "test", "persist": False})
                connected = websocket.receive_json()
                self.assertEqual(connected["type"], "connected")
                self.assertFalse(connected["persisted"])
                second = websocket.receive_json()
                self.assertEqual(second["type"], "stt.connected")
                websocket.send_bytes(b"\0" * 1024)
                events = []
                while not any(event.get("type") == "llm.completed" for event in events):
                    received = websocket.receive_json()
                    events.append(received)
                completed = next(event for event in events if event["type"] == "llm.completed")
                self.assertEqual(completed["answer"], "Main check karti hoon.")
                self.assertEqual(completed["metrics"]["stt_provider_ms"], 81.0)
                self.assertEqual(completed["metrics"]["account_context_ms"], 2.0)
                self.assertEqual(completed["metrics"]["router_ms"], 3.0)
                self.assertEqual(completed["metrics"]["context_assembly_ms"], 4.0)
                self.assertEqual(completed["metrics"]["retrieval_ms"], 5.0)
                self.assertEqual(completed["metrics"]["prompt_build_ms"], 6.0)
                self.assertEqual(completed["metrics"]["routing_prompt_ms"], 18.0)
                self.assertEqual(completed["metrics"]["model_ttft_ms"], 40.0)
                self.assertEqual(completed["metrics"]["answer_decode_ttft_ms"], 48.0)
                self.assertEqual(completed["metrics"]["model_stream_ms"], 75.0)
                self.assertTrue(any(event["type"] == "transcript.final" for event in events))
                websocket.send_json({"type": "stop"})
                terminal = websocket.receive_json()
                while terminal["type"] != "session.completed":
                    terminal = websocket.receive_json()
                self.assertEqual(terminal["turns"], 1)

