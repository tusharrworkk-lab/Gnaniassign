"""Tests for the lightweight operations-dashboard runtime primitives."""

from __future__ import annotations

import asyncio
import unittest

from gnaniassignment.api.server import RuntimeEvents
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

