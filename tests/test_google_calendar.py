"""Tests for private Calendar OAuth state and current-event narrowing."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from cryptography.fernet import Fernet

from gnaniassignment.google_calendar import CalendarIntegrationError, GoogleCalendarClient, GoogleCalendarSettings, calendar_presence_context


class Response:
    def __init__(self, body: object, ok: bool = True, status_code: int = 200) -> None:
        self.body, self.ok, self.status_code = body, ok, status_code

    def json(self):
        return self.body


class Session:
    def __init__(self, responses: list[Response]) -> None:
        self.responses, self.calls = responses, []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)


class Store:
    def __init__(self, snapshot): self._snapshot = snapshot
    def snapshot(self, _owner_id): return self._snapshot


class GoogleCalendarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = GoogleCalendarSettings(
            client_id="client-id.apps.googleusercontent.com",
            client_secret="client-secret",
            redirect_uri="https://voice.sidecut.me/v1/integrations/google-calendar/callback",
            token_encryption_key=Fernet.generate_key().decode(),
            state_secret="state-secret-for-tests",
            owner_id="tushar",
        )

    def test_authorization_url_requests_offline_owned_event_scope_and_signed_state(self):
        client = GoogleCalendarClient(self.settings)
        query = parse_qs(urlparse(client.authorization_url()).query)
        self.assertEqual(query["access_type"], ["offline"])
        self.assertEqual(query["prompt"], ["consent"])
        self.assertEqual(query["redirect_uri"], [self.settings.redirect_uri])
        self.assertEqual(query["scope"], ["https://www.googleapis.com/auth/calendar.events.owned.readonly"])
        self.assertEqual(client.verify_state(query["state"][0]), "tushar")

    def test_tampered_oauth_state_is_rejected(self):
        client = GoogleCalendarClient(self.settings)
        state = parse_qs(urlparse(client.authorization_url()).query)["state"][0]
        with self.assertRaises(CalendarIntegrationError):
            client.verify_state(state[:-1] + ("a" if state[-1] != "a" else "b"))

    def test_localhost_callback_is_allowed_only_for_development(self):
        with patch.dict("os.environ", {
            "GOOGLE_CALENDAR_CLIENT_ID": "client",
            "GOOGLE_CALENDAR_CLIENT_SECRET": "secret",
            "GOOGLE_CALENDAR_TOKEN_ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "GOOGLE_CALENDAR_STATE_SECRET": "state-secret",
            "GOOGLE_CALENDAR_REDIRECT_URI": "http://localhost:8080/v1/integrations/google-calendar/callback",
        }, clear=True):
            self.assertTrue(GoogleCalendarSettings.from_environment().redirect_uri.startswith("http://localhost"))

    def test_current_event_ignores_transparent_and_returns_only_active_event(self):
        session = Session([
            Response({"access_token": "access-token"}),
            Response({"items": [
                {"summary": "FYI", "transparency": "transparent", "start": {"dateTime": "2026-11-09T10:00:00Z"}, "end": {"dateTime": "2026-11-09T12:00:00Z"}},
                {"summary": "Private planning", "start": {"dateTime": "2026-11-09T10:30:00Z"}, "end": {"dateTime": "2026-11-09T11:30:00Z"}},
            ]}),
        ])
        snapshot = GoogleCalendarClient(self.settings, session).current_event("refresh-token", datetime(2026, 11, 9, 11, 0, tzinfo=UTC))
        self.assertEqual(snapshot["status"], "busy")
        self.assertEqual(snapshot["event_title"], "Private planning")
        self.assertEqual(snapshot["busy_until"], "2026-11-09T11:30:00+00:00")

    def test_presence_context_never_includes_event_title(self):
        context = calendar_presence_context(Store({"status": "busy", "busy_until": "2026-11-09T11:30:00+00:00", "event_title": "Private planning"}), "tushar")
        self.assertIn("unavailable", context)
        self.assertNotIn("Private planning", context)

