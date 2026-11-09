CREATE TABLE IF NOT EXISTS call_latency_metrics (
    stream_sid TEXT PRIMARY KEY,
    call_sid TEXT NOT NULL,
    customer_id TEXT,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS call_latency_metrics_updated_idx ON call_latency_metrics (updated_at DESC);

-- One record for every committed user turn. No raw caller audio or transcript
-- is stored here; this table exists solely for voice quality SLOs.
CREATE TABLE IF NOT EXISTS voice_turn_evaluations (
    stream_sid TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    call_sid TEXT NOT NULL,
    customer_id TEXT,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream_sid, turn_index)
);
CREATE INDEX IF NOT EXISTS voice_turn_evaluations_updated_idx ON voice_turn_evaluations (updated_at DESC);
