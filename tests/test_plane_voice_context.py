"""Caller-gated, on-demand Plane context for the voice agent."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from gnaniassignment.app.container import create_plane_work_item_window


class PlaneVoiceContextTests(unittest.TestCase):
    def test_disabled_or_untrusted_callers_never_create_a_lookup_window(self):
        with patch.dict(os.environ, {"PLANE_VOICE_STATUS_ENABLED": "true", "PLANE_VOICE_TRUSTED_CALLERS": "9000000000"}, clear=False):
            self.assertIsNone(create_plane_work_item_window("caller:8005836092"))

    def test_trusted_caller_queries_plane_only_for_explicit_identifier(self):
        calls: list[str] = []

        class Client:
            def find_work_item(self, identifier):
                calls.append(identifier)
                return {"identifier": identifier, "name": "Relay rollout", "state": {"name": "In progress"}}

        with patch.dict(os.environ, {"PLANE_VOICE_STATUS_ENABLED": "true", "PLANE_VOICE_TRUSTED_CALLERS": "+91 80058 36092"}, clear=False):
            window = create_plane_work_item_window("caller:8005836092", Client)
            self.assertIsNotNone(window)
            assert window is not None
            self.assertEqual(window("Kal milte hain", {}), "")
            self.assertEqual(calls, [])
            context = window("SC-42 ka work item update kya hai?", {})
        self.assertEqual(calls, ["SC-42"])
        self.assertIn("Relay rollout", context)

    def test_lookup_failure_never_invents_a_status(self):
        class Client:
            def find_work_item(self, _identifier):
                from gnaniassignment.work_items import WorkItemProviderError
                raise WorkItemProviderError("provider unavailable")

        with patch.dict(os.environ, {"PLANE_VOICE_STATUS_ENABLED": "true", "PLANE_VOICE_TRUSTED_CALLERS": "8005836092"}, clear=False):
            window = create_plane_work_item_window("caller:8005836092", Client)
            assert window is not None
            context = window("Plane SC-42 status", {})
        self.assertIn("temporarily unavailable", context)
        self.assertIn("Do not invent", context)
