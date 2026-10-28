-- Conversation transcripts are sensitive. This local review ledger is opt-in.
CREATE TABLE IF NOT EXISTS conversation_turns (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id TEXT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    user_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    routed_intent TEXT NOT NULL,
    classifier_intent TEXT,
    classifier_confidence DOUBLE PRECISION,
    final_intent TEXT,
    reviewed_at TIMESTAMPTZ,
    working_memory JSONB NOT NULL DEFAULT '{}'::jsonb,
    flow_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
    flow_status TEXT NOT NULL DEFAULT 'ok' CHECK (flow_status IN ('ok', 'needs_review')),
    next_user_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, turn_number)
);

ALTER TABLE conversation_turns
    ADD COLUMN IF NOT EXISTS final_intent TEXT,
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS conversation_turns_review_idx
    ON conversation_turns (flow_status, created_at DESC);
CREATE INDEX IF NOT EXISTS conversation_turns_customer_idx
    ON conversation_turns (customer_id, created_at DESC);

CREATE TABLE IF NOT EXISTS conversation_calls (
    session_id TEXT PRIMARY KEY,
    customer_id TEXT,
    stt_segments JSONB NOT NULL DEFAULT '[]'::jsonb,
    transcript JSONB NOT NULL DEFAULT '[]'::jsonb,
    judge_status TEXT NOT NULL DEFAULT 'not_requested',
    judge_result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS conversation_pattern_rules (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_session_id TEXT REFERENCES conversation_calls(session_id),
    rule_kind TEXT NOT NULL CHECK (rule_kind IN ('do', 'dont')),
    rule_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'active', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    activated_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS conversation_pattern_rules_status_idx
    ON conversation_pattern_rules (status, created_at DESC);
