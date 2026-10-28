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

    def _connect(self):
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Install PostgreSQL memory support with: uv sync --extra postgres-memory") from error
        return psycopg.connect(self.database_url)
