"""Durable metadata-only latency measurements for telephony calls."""
from __future__ import annotations
import json, os
from typing import Any

VOICE_TURN_TABLE_SQL = "CREATE TABLE IF NOT EXISTS voice_turn_evaluations (stream_sid TEXT NOT NULL, turn_index INTEGER NOT NULL, call_sid TEXT NOT NULL, customer_id TEXT, metrics JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY(stream_sid, turn_index))"
VOICE_TURN_INDEX_SQL = "CREATE INDEX IF NOT EXISTS voice_turn_evaluations_updated_idx ON voice_turn_evaluations (updated_at DESC)"

class CallLatencyStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("CONVERSATION_LOG_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
        if not self.database_url: raise ValueError("Call latency storage requires PostgreSQL")
    def record(self, stream_sid: str, call_sid: str, customer_id: str, metrics: dict[str, Any]) -> None:
        import psycopg
        with psycopg.connect(self.database_url) as c, c.cursor() as q:
            q.execute("CREATE TABLE IF NOT EXISTS call_latency_metrics (stream_sid TEXT PRIMARY KEY, call_sid TEXT NOT NULL, customer_id TEXT, metrics JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now())")
            q.execute("INSERT INTO call_latency_metrics (stream_sid,call_sid,customer_id,metrics) VALUES (%s,%s,%s,%s::jsonb) ON CONFLICT (stream_sid) DO UPDATE SET metrics=call_latency_metrics.metrics || EXCLUDED.metrics, updated_at=now()", (stream_sid,call_sid,customer_id,json.dumps(metrics)))
    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        import psycopg
        with psycopg.connect(self.database_url) as c, c.cursor() as q:
            q.execute("SELECT stream_sid,call_sid,customer_id,metrics,created_at,updated_at FROM call_latency_metrics ORDER BY updated_at DESC LIMIT %s", (limit,))
            return [{"stream_sid":r[0],"call_sid":r[1],"customer_id":r[2],"metrics":r[3],"created_at":r[4].isoformat(),"updated_at":r[5].isoformat()} for r in q.fetchall()]

    def get(self, stream_sid: str) -> dict[str, Any] | None:
        import psycopg
        with psycopg.connect(self.database_url) as c, c.cursor() as q:
            q.execute("SELECT stream_sid,call_sid,customer_id,metrics,created_at,updated_at FROM call_latency_metrics WHERE stream_sid = %s", (stream_sid,))
            row = q.fetchone()
            return {"stream_sid": row[0], "call_sid": row[1], "customer_id": row[2], "metrics": row[3], "created_at": row[4].isoformat(), "updated_at": row[5].isoformat()} if row else None

    def record_turn(self, stream_sid: str, call_sid: str, customer_id: str, turn_index: int, metrics: dict[str, Any]) -> None:
        """Persist one privacy-preserving evaluation record per user turn."""
        import psycopg
        with psycopg.connect(self.database_url) as c, c.cursor() as q:
            q.execute(VOICE_TURN_TABLE_SQL)
            q.execute(VOICE_TURN_INDEX_SQL)
            q.execute("INSERT INTO voice_turn_evaluations (stream_sid,turn_index,call_sid,customer_id,metrics) VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT (stream_sid,turn_index) DO UPDATE SET metrics=voice_turn_evaluations.metrics || EXCLUDED.metrics, updated_at=now()", (stream_sid, turn_index, call_sid, customer_id, json.dumps(metrics)))

    def recent_turns(self, limit: int = 100) -> list[dict[str, Any]]:
        import psycopg
        with psycopg.connect(self.database_url) as c, c.cursor() as q:
            # Existing deployments have a persistent PostgreSQL volume, so
            # docker-entrypoint init scripts are not re-run on upgrade.
            q.execute(VOICE_TURN_TABLE_SQL)
            q.execute(VOICE_TURN_INDEX_SQL)
            q.execute("SELECT stream_sid,turn_index,call_sid,customer_id,metrics,created_at,updated_at FROM voice_turn_evaluations ORDER BY updated_at DESC LIMIT %s", (limit,))
            return [{"stream_sid": r[0], "turn_index": r[1], "call_sid": r[2], "customer_id": r[3], "metrics": r[4], "created_at": r[5].isoformat(), "updated_at": r[6].isoformat()} for r in q.fetchall()]

    def turns_for_stream(self, stream_sid: str) -> list[dict[str, Any]]:
        import psycopg
        with psycopg.connect(self.database_url) as c, c.cursor() as q:
            q.execute(VOICE_TURN_TABLE_SQL)
            q.execute(VOICE_TURN_INDEX_SQL)
            q.execute("SELECT stream_sid,turn_index,call_sid,customer_id,metrics,created_at,updated_at FROM voice_turn_evaluations WHERE stream_sid = %s ORDER BY turn_index", (stream_sid,))
            return [{"stream_sid": r[0], "turn_index": r[1], "call_sid": r[2], "customer_id": r[3], "metrics": r[4], "created_at": r[5].isoformat(), "updated_at": r[6].isoformat()} for r in q.fetchall()]
