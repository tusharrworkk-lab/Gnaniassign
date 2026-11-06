CREATE TABLE IF NOT EXISTS call_latency_metrics (
    stream_sid TEXT PRIMARY KEY,
    call_sid TEXT NOT NULL,
    customer_id TEXT,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS call_latency_metrics_updated_idx ON call_latency_metrics (updated_at DESC);
