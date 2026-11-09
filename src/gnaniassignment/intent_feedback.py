"""Durable shadow-intent observations and human feedback for periodic training."""

from __future__ import annotations

import json
import os
from typing import Any


class PostgresIntentFeedbackStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("INTENT_FEEDBACK_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
        if not self.database_url:
            raise ValueError("Set INTENT_FEEDBACK_DATABASE_URL or EPISODIC_MEMORY_DATABASE_URL")

    def record_prediction(self, customer_id: str | None, query: str, prediction: dict[str, Any], session_id: str | None = None) -> None:
        decision = prediction.get("answers", {}).get("intent", {})
        choice = decision.get("choice")
        confidence = decision.get("confidence")
        if not isinstance(choice, str) or not isinstance(confidence, (int, float)):
            return
        probabilities = decision.get("probabilities", {})
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO intent_shadow_observations
                    (customer_id, session_id, query_text, predicted_intent, confidence, probabilities)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
                (customer_id, session_id, query, choice, float(confidence), json.dumps(probabilities)),
            )

    def pending(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the private operator queue; never treat predictions as labels."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, query_text, predicted_intent, confidence, created_at
                FROM intent_shadow_observations
                WHERE reviewed_intent IS NULL
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [
                {
                    "id": row[0], "text": row[1], "intent": row[2],
                    "confidence": row[3], "created_at": row[4].isoformat(),
                }
                for row in cursor.fetchall()
            ]

    def review(self, observation_id: int, final_intent: str) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE intent_shadow_observations
                SET reviewed_intent = %s, reviewed_at = now()
                WHERE id = %s AND reviewed_intent IS NULL
                """,
                (final_intent, observation_id),
            )
            return cursor.rowcount == 1

    def _connect(self):
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Install PostgreSQL memory support with: uv sync --extra postgres-memory") from error
        return psycopg.connect(self.database_url)
