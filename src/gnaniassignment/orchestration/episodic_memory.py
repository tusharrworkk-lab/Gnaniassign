"""Persistent, bounded customer episodic-memory adapters."""

from __future__ import annotations

import os
from typing import Protocol

from gnaniassignment.llm.memory import ChatTurn


class EpisodicMemoryStore(Protocol):
    """Store completed turns and restore them in chronological order."""

    def load_recent(self, customer_id: str, limit: int) -> list[ChatTurn]: ...

    def append(self, customer_id: str, session_id: str, turn: ChatTurn) -> None: ...


class PostgresEpisodicMemoryStore:
    """PostgreSQL implementation; data retention is managed by the database."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
        if not self.database_url:
            raise ValueError("Set EPISODIC_MEMORY_DATABASE_URL to enable PostgreSQL episodic memory")

    def _connect(self):
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Install PostgreSQL memory support with: uv sync --extra postgres-memory") from error
        return psycopg.connect(self.database_url)

    def load_recent(self, customer_id: str, limit: int) -> list[ChatTurn]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT query_text, answer_text, summary, intent, topic
                FROM episodic_memory_turns
                WHERE customer_id = %s
                ORDER BY occurred_at DESC, id DESC
                LIMIT %s
                """,
                (customer_id, limit),
            )
            rows = cursor.fetchall()
        return [ChatTurn(*row) for row in reversed(rows)]

    def append(self, customer_id: str, session_id: str, turn: ChatTurn) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO episodic_memory_turns
                    (customer_id, session_id, query_text, answer_text, summary, intent, topic)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (customer_id, session_id, turn.query, turn.answer, turn.summary, turn.intent, turn.topic),
            )

