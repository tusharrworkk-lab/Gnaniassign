-- PostgreSQL schema for persisted, per-customer episodic conversation memory.
-- Apply with: psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/episodic_memory.sql

CREATE TABLE IF NOT EXISTS episodic_memory_turns (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    summary TEXT NOT NULL,
    intent TEXT NOT NULL,
    topic TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS episodic_memory_customer_recent_idx
    ON episodic_memory_turns (customer_id, occurred_at DESC, id DESC);

-- A new call loads only these compact summaries. Raw rows above remain for
-- audit/analytics and are not replayed verbatim into the live voice prompt.
CREATE TABLE IF NOT EXISTS episodic_memory_call_summaries (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    intent TEXT NOT NULL,
    topic TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS episodic_memory_call_summary_customer_recent_idx
    ON episodic_memory_call_summaries (customer_id, occurred_at DESC, id DESC);

-- Retention is a policy choice. Example: delete rows older than 30 days with a
-- scheduled job rather than storing customer transcripts indefinitely.
