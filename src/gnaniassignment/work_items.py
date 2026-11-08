"""Provider adapters and approval ledger for work-management integrations.

The agent may read provider data, but provider mutations are represented as a
durable local draft first.  This module intentionally does not contain an LLM
call: only an authenticated owner approval can invoke ``apply_comment``.
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import requests


class WorkItemProviderError(RuntimeError):
    """A safe error raised by an external work-management provider."""


@dataclass(frozen=True)
class PlaneSettings:
    """The minimum non-secret Plane connection configuration."""

    base_url: str
    workspace_slug: str
    api_key: str

    @classmethod
    def from_environment(cls) -> "PlaneSettings":
        api_key = os.getenv("PLANE_API_KEY", "").strip()
        workspace_slug = os.getenv("PLANE_WORKSPACE_SLUG", "").strip()
        base_url = os.getenv("PLANE_API_BASE_URL", "https://api.plane.so").strip().rstrip("/")
        if not api_key or not workspace_slug:
            raise ValueError("Set PLANE_API_KEY and PLANE_WORKSPACE_SLUG to enable Plane")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("PLANE_API_BASE_URL must be an absolute HTTP(S) URL")
        return cls(base_url=base_url, workspace_slug=workspace_slug, api_key=api_key)


class PlaneClient:
    """Small v2 Plane client restricted to the required work-item operations."""

    def __init__(self, settings: PlaneSettings | None = None, session: requests.Session | None = None) -> None:
        self.settings = settings or PlaneSettings.from_environment()
        self.session = session or requests.Session()

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.settings.api_key, "Accept": "application/json"}

    def _request(self, method: str, path: str, *, params: dict[str, str] | None = None, payload: dict[str, Any] | None = None) -> Any:
        try:
            response = self.session.request(
                method,
                f"{self.settings.base_url}{path}",
                headers=self._headers,
                params=params,
                json=payload,
                timeout=(3.05, 12),
            )
        except requests.RequestException as error:
            raise WorkItemProviderError("Plane is unavailable; the action was not applied.") from error
        if not response.ok:
            # Provider responses can include workspace/project metadata. Never
            # pass them through to an untrusted caller or the voice model.
            raise WorkItemProviderError(f"Plane rejected the request ({response.status_code}).")
        try:
            return response.json()
        except ValueError as error:
            raise WorkItemProviderError("Plane returned an invalid response.") from error

    def health(self) -> dict[str, Any]:
        result = self._request("GET", "/api/v2/users/me/")
        return result if isinstance(result, dict) else {}

    def list_work_items(self, limit: int = 25) -> list[dict[str, Any]]:
        result = self._request(
            "GET",
            f"/api/v2/workspaces/{self.settings.workspace_slug}/work-items/",
            params={
                "per_page": str(max(1, min(limit, 100))),
                "fields": "id,identifier,name,priority,state_id,target_date,updated_at,assignee_ids,project_id",
                "expand": "state,assignees",
            },
        )
        rows = result.get("data", []) if isinstance(result, dict) else []
        return [row for row in rows if isinstance(row, dict)]

    def apply_comment(self, *, project_id: str, work_item_id: str, comment: str, external_id: str) -> dict[str, Any]:
        """Post one approved comment, tagged for local audit and deduplication."""
        clean = comment.strip()
        if not clean:
            raise ValueError("Comment cannot be empty")
        path = (
            f"/api/v2/workspaces/{self.settings.workspace_slug}/projects/{project_id}"
            f"/work-items/{work_item_id}/comments/"
        )
        existing = self._request(
            "GET", path,
            params={"external_id": external_id, "external_source": "voice-agent"},
        )
        rows = existing.get("data", []) if isinstance(existing, dict) else []
        if rows and isinstance(rows[0], dict):
            return rows[0]
        return self._request(
            "POST",
            path,
            payload={
                "comment_html": f"<p>{html.escape(clean, quote=False)}</p>",
                "external_id": external_id,
                "external_source": "voice-agent",
            },
        )


class PostgresWorkActionStore:
    """Approval ledger; each Plane mutation begins as a local draft."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("WORK_ACTION_DATABASE_URL") or os.getenv("CONVERSATION_LOG_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
        if not self.database_url:
            raise ValueError("Set WORK_ACTION_DATABASE_URL or a PostgreSQL database URL")
        self._ensure_schema()

    def create_comment_draft(self, *, owner_user_id: str, requested_by: str, project_id: str, work_item_id: str, comment: str) -> dict[str, Any]:
        clean = comment.strip()
        if not clean:
            raise ValueError("Comment cannot be empty")
        action_id = str(uuid4())
        payload = {"project_id": project_id, "work_item_id": work_item_id, "comment": clean}
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO pending_work_actions
                    (id, owner_user_id, requested_by, provider, action_type, work_item_id, proposed_payload, status)
                VALUES (%s, %s, %s, 'plane', 'comment', %s, %s::jsonb, 'pending')
                RETURNING id, status, created_at
                """,
                (action_id, owner_user_id, requested_by, work_item_id, json.dumps(payload)),
            )
            row = cursor.fetchone()
        return {"id": str(row[0]), "status": row[1], "created_at": row[2].isoformat(), "provider": "plane", "action_type": "comment", "work_item_id": work_item_id, "proposed_payload": payload}

    def list_actions(self, owner_user_id: str, status: str = "pending", limit: int = 100) -> list[dict[str, Any]]:
        if status not in {"pending", "approved", "rejected", "applying", "applied", "failed"}:
            raise ValueError("Unsupported work-action status")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, requested_by, provider, action_type, work_item_id, proposed_payload,
                       status, failure_reason, created_at, approved_at, applied_at
                FROM pending_work_actions
                WHERE owner_user_id = %s AND status = %s
                ORDER BY created_at DESC LIMIT %s
                """,
                (owner_user_id, status, max(1, min(limit, 500))),
            )
            rows = cursor.fetchall()
        keys = ("id", "requested_by", "provider", "action_type", "work_item_id", "proposed_payload", "status", "failure_reason", "created_at", "approved_at", "applied_at")
        return [
            {
                **dict(zip(keys, row, strict=True)),
                "id": str(row[0]),
                "created_at": row[8].isoformat(),
                "approved_at": row[9].isoformat() if row[9] else None,
                "applied_at": row[10].isoformat() if row[10] else None,
            }
            for row in rows
        ]

    def reject(self, owner_user_id: str, action_id: str) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE pending_work_actions SET status = 'rejected', rejected_at = now()
                   WHERE id = %s AND owner_user_id = %s AND status = 'pending'""",
                (action_id, owner_user_id),
            )
            return cursor.rowcount == 1

    def claim_for_approval(self, owner_user_id: str, action_id: str) -> dict[str, Any] | None:
        """Atomically claim a pending action so it cannot be applied twice."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_work_actions
                SET status = 'applying', approved_at = now()
                WHERE id = %s AND owner_user_id = %s AND status = 'pending'
                RETURNING proposed_payload, provider, action_type
                """,
                (action_id, owner_user_id),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {"proposed_payload": row[0], "provider": row[1], "action_type": row[2]}

    def mark_applied(self, action_id: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE pending_work_actions SET status = 'applied', applied_at = now(), failure_reason = NULL WHERE id = %s AND status = 'applying'", (action_id,))

    def mark_failed(self, action_id: str, reason: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE pending_work_actions SET status = 'failed', failure_reason = %s WHERE id = %s AND status = 'applying'", (reason[:500], action_id))

    def _ensure_schema(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_work_actions (
                    id UUID PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    work_item_id TEXT NOT NULL,
                    proposed_payload JSONB NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'applying', 'applied', 'failed')),
                    failure_reason TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    approved_at TIMESTAMPTZ,
                    rejected_at TIMESTAMPTZ,
                    applied_at TIMESTAMPTZ
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS pending_work_actions_owner_status_idx ON pending_work_actions (owner_user_id, status, created_at DESC)")

    def _connect(self):
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Install PostgreSQL support with: uv sync --extra postgres-memory") from error
        return psycopg.connect(self.database_url)
