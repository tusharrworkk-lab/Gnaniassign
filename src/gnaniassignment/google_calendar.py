"""Private Google Calendar presence integration for the personal voice agent.

OAuth credentials and refresh tokens never reach the browser, voice prompt
tree, or call transcript.  The call path reads a compact cached presence
snapshot; Google API refreshes happen outside the live LLM turn.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode, urlparse

import requests


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
GOOGLE_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.events.owned.readonly"


class CalendarIntegrationError(RuntimeError):
    """Safe error surface for OAuth and Calendar provider failures."""


@dataclass(frozen=True)
class GoogleCalendarSettings:
    client_id: str
    client_secret: str
    redirect_uri: str
    token_encryption_key: str
    state_secret: str
    owner_id: str

    @classmethod
    def from_environment(cls) -> "GoogleCalendarSettings":
        values = {
            "client_id": os.getenv("GOOGLE_CALENDAR_CLIENT_ID", "").strip(),
            "client_secret": os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET", "").strip(),
            "redirect_uri": os.getenv(
                "GOOGLE_CALENDAR_REDIRECT_URI",
                "https://voice.sidecut.me/v1/integrations/google-calendar/callback",
            ).strip(),
            "token_encryption_key": os.getenv("GOOGLE_CALENDAR_TOKEN_ENCRYPTION_KEY", "").strip(),
            "state_secret": os.getenv("GOOGLE_CALENDAR_STATE_SECRET", "").strip(),
            "owner_id": os.getenv("GOOGLE_CALENDAR_OWNER_ID", "owner").strip(),
        }
        missing = [name for name in ("client_id", "client_secret", "token_encryption_key", "state_secret") if not values[name]]
        if missing:
            raise ValueError("Set " + ", ".join(f"GOOGLE_CALENDAR_{name.upper()}" for name in missing) + " to enable Google Calendar")
        redirect_uri = urlparse(values["redirect_uri"])
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        local_development_redirect = redirect_uri.scheme == "http" and redirect_uri.hostname in local_hosts
        if redirect_uri.scheme != "https" and not local_development_redirect:
            raise ValueError("GOOGLE_CALENDAR_REDIRECT_URI must use HTTPS (except localhost during development)")
        return cls(**values)


def _fernet(key: str):
    try:
        from cryptography.fernet import Fernet
    except ImportError as error:
        raise RuntimeError("Install Google Calendar support with: uv sync --extra google-calendar") from error
    try:
        return Fernet(key.encode())
    except (TypeError, ValueError) as error:
        raise ValueError("GOOGLE_CALENDAR_TOKEN_ENCRYPTION_KEY must be a Fernet key") from error


class GoogleCalendarStore:
    """PostgreSQL storage for encrypted OAuth refresh tokens and snapshots."""

    def __init__(self, settings: GoogleCalendarSettings | None = None, database_url: str | None = None) -> None:
        self.settings = settings or GoogleCalendarSettings.from_environment()
        self.database_url = database_url or os.getenv("GOOGLE_CALENDAR_DATABASE_URL") or os.getenv("CONVERSATION_LOG_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
        if not self.database_url:
            raise ValueError("Set GOOGLE_CALENDAR_DATABASE_URL or an existing PostgreSQL database URL")
        self._ensure_schema()

    def save_refresh_token(self, owner_id: str, refresh_token: str, scopes: str | None = None) -> None:
        encrypted = _fernet(self.settings.token_encryption_key).encrypt(refresh_token.encode()).decode()
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO google_calendar_connections (owner_id, encrypted_refresh_token, scopes, connected_at, updated_at)
                VALUES (%s, %s, %s, now(), now())
                ON CONFLICT (owner_id) DO UPDATE SET
                    encrypted_refresh_token = EXCLUDED.encrypted_refresh_token,
                    scopes = EXCLUDED.scopes, updated_at = now(), disconnected_at = NULL
                """,
                (owner_id, encrypted, scopes),
            )

    def refresh_token(self, owner_id: str) -> str | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT encrypted_refresh_token FROM google_calendar_connections
                   WHERE owner_id = %s AND disconnected_at IS NULL""",
                (owner_id,),
            )
            row = cursor.fetchone()
        return _fernet(self.settings.token_encryption_key).decrypt(row[0].encode()).decode() if row else None

    def disconnect(self, owner_id: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE google_calendar_connections SET disconnected_at = now(), updated_at = now() WHERE owner_id = %s", (owner_id,))
            cursor.execute("DELETE FROM google_calendar_current_events WHERE owner_id = %s", (owner_id,))

    def is_connected(self, owner_id: str) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM google_calendar_connections WHERE owner_id = %s AND disconnected_at IS NULL", (owner_id,))
            return cursor.fetchone() is not None

    def save_snapshot(self, owner_id: str, snapshot: dict[str, Any]) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO google_calendar_current_events (owner_id, snapshot, refreshed_at)
                VALUES (%s, %s::jsonb, now())
                ON CONFLICT (owner_id) DO UPDATE SET snapshot = EXCLUDED.snapshot, refreshed_at = now()
                """,
                (owner_id, json.dumps(snapshot)),
            )

    def snapshot(self, owner_id: str) -> dict[str, Any] | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT snapshot, refreshed_at FROM google_calendar_current_events WHERE owner_id = %s", (owner_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        snapshot = dict(row[0]) if isinstance(row[0], dict) else json.loads(row[0])
        snapshot["refreshed_at"] = row[1].isoformat()
        return snapshot

    def _ensure_schema(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS google_calendar_connections (
                    owner_id TEXT PRIMARY KEY,
                    encrypted_refresh_token TEXT NOT NULL,
                    scopes TEXT,
                    connected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    disconnected_at TIMESTAMPTZ
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS google_calendar_current_events (
                    owner_id TEXT PRIMARY KEY,
                    snapshot JSONB NOT NULL,
                    refreshed_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    def _connect(self):
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Install PostgreSQL support with: uv sync --extra postgres-memory") from error
        return psycopg.connect(self.database_url)


class GoogleCalendarClient:
    def __init__(self, settings: GoogleCalendarSettings | None = None, session: requests.Session | None = None) -> None:
        self.settings = settings or GoogleCalendarSettings.from_environment()
        self.session = session or requests.Session()

    def authorization_url(self, owner_id: str | None = None) -> str:
        state = self._sign_state(owner_id or self.settings.owner_id)
        query = urlencode(
            {
                "client_id": self.settings.client_id,
                "redirect_uri": self.settings.redirect_uri,
                "response_type": "code",
                "scope": GOOGLE_CALENDAR_SCOPE,
                "access_type": "offline",
                "prompt": "consent",
                "include_granted_scopes": "true",
                "state": state,
            }
        )
        return f"{GOOGLE_AUTH_URL}?{query}"

    def verify_state(self, state: str) -> str:
        try:
            payload64, signature = state.split(".", 1)
            expected = hmac.new(self.settings.state_secret.encode(), payload64.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            payload = json.loads(_urlsafe_decode(payload64))
            if not isinstance(payload, dict) or not isinstance(payload.get("owner_id"), str) or float(payload.get("expires_at", 0)) < datetime.now(UTC).timestamp():
                raise ValueError
            return payload["owner_id"]
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise CalendarIntegrationError("Calendar authorization state is invalid or expired. Start connection again.") from error

    def exchange_code(self, code: str) -> dict[str, Any]:
        result = self._post_token(
            {
                "code": code,
                "client_id": self.settings.client_id,
                "client_secret": self.settings.client_secret,
                "redirect_uri": self.settings.redirect_uri,
                "grant_type": "authorization_code",
            }
        )
        if not isinstance(result.get("refresh_token"), str):
            raise CalendarIntegrationError("Google did not return a refresh token. Reconnect and approve the requested access.")
        return result

    def current_event(self, refresh_token: str, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        token = self._post_token(
            {
                "client_id": self.settings.client_id,
                "client_secret": self.settings.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        )
        access_token = token.get("access_token")
        if not isinstance(access_token, str):
            raise CalendarIntegrationError("Google Calendar authorization needs to be reconnected.")
        try:
            response = self.session.get(
                GOOGLE_EVENTS_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "timeMin": (now - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                    "timeMax": (now + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "maxResults": "25",
                    "fields": "items(id,status,transparency,summary,start,end)",
                },
                timeout=(3.05, 12),
            )
        except requests.RequestException as error:
            raise CalendarIntegrationError("Google Calendar is temporarily unavailable.") from error
        if not response.ok:
            raise CalendarIntegrationError(f"Google Calendar rejected the request ({response.status_code}).")
        try:
            events = response.json().get("items", [])
        except (ValueError, AttributeError) as error:
            raise CalendarIntegrationError("Google Calendar returned an invalid response.") from error
        active = next((event for event in events if _is_active(event, now)), None)
        if active is None:
            return {"status": "available", "busy_until": None, "event_title": None}
        end = _event_datetime(active.get("end", {}), now)
        return {
            "status": "busy",
            "busy_until": end.isoformat() if end else None,
            # Private context only. The prompt explicitly prohibits telling a
            # caller the title/reason unless a future owner policy allows it.
            "event_title": str(active.get("summary") or "")[:240] or None,
        }

    def _post_token(self, payload: dict[str, str]) -> dict[str, Any]:
        try:
            response = self.session.post(GOOGLE_TOKEN_URL, data=payload, timeout=(3.05, 12))
        except requests.RequestException as error:
            raise CalendarIntegrationError("Google authorization is temporarily unavailable.") from error
        if not response.ok:
            raise CalendarIntegrationError("Google authorization was rejected. Reconnect the calendar.")
        try:
            result = response.json()
        except ValueError as error:
            raise CalendarIntegrationError("Google authorization returned an invalid response.") from error
        return result if isinstance(result, dict) else {}

    def _sign_state(self, owner_id: str) -> str:
        payload = {"owner_id": owner_id, "nonce": secrets.token_urlsafe(18), "expires_at": (datetime.now(UTC) + timedelta(minutes=10)).timestamp()}
        encoded = _urlsafe_encode(json.dumps(payload, separators=(",", ":")).encode())
        signature = hmac.new(self.settings.state_secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"


def refresh_current_event(store: GoogleCalendarStore, client: GoogleCalendarClient, owner_id: str) -> dict[str, Any]:
    token = store.refresh_token(owner_id)
    if token is None:
        raise CalendarIntegrationError("Google Calendar is not connected.")
    snapshot = client.current_event(token)
    store.save_snapshot(owner_id, snapshot)
    return snapshot


def calendar_presence_context(store: GoogleCalendarStore, owner_id: str) -> str:
    snapshot = store.snapshot(owner_id)
    if not snapshot:
        return "Calendar presence: unavailable. Do not claim the owner's availability."
    if snapshot.get("status") != "busy":
        return "Calendar presence: the owner currently appears available. This is private operational context; do not mention calendar details unless necessary."
    until = snapshot.get("busy_until") or "an unknown time"
    return f"Calendar presence: the owner is currently unavailable until {until}. This is private operational context. Do not disclose the event title, attendee, reason, or other schedule details. If directly relevant, you may simply say the owner is unavailable and offer to take a message."


def _event_datetime(value: object, now: datetime) -> datetime | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("dateTime")
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None
    date_raw = value.get("date")
    if isinstance(date_raw, str):
        try:
            return datetime.fromisoformat(date_raw).replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def _is_active(event: object, now: datetime) -> bool:
    if not isinstance(event, dict) or event.get("status") == "cancelled" or event.get("transparency") == "transparent":
        return False
    start, end = _event_datetime(event.get("start"), now), _event_datetime(event.get("end"), now)
    return start is not None and end is not None and start <= now < end


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _urlsafe_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
