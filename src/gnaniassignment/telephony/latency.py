"""Durable metadata-only latency measurements for telephony calls."""
from __future__ import annotations
import json, os
from typing import Any

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
