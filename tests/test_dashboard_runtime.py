"""Tests for the lightweight operations-dashboard runtime primitives."""

from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from gnaniassignment.api.server import RuntimeEvents, SessionRegistry, create_app
from gnaniassignment.prompts.tree import _validate_condition


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
        _validate_condition({"language": ["hinglish", "english"], "channel": "voice"})
        with self.assertRaises(ValueError):
            _validate_condition({"caller_text": "refund"})


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

