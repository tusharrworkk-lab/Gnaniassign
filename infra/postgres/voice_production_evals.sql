CREATE TABLE IF NOT EXISTS voice_eval_runs (
    id UUID PRIMARY KEY,
    layer TEXT NOT NULL CHECK (layer IN ('real_telephony', 'production_sample')),
    scenario_id TEXT,
    scenario_version TEXT,
    session_id TEXT,
    call_sid TEXT,
    customer_id TEXT,
    status TEXT NOT NULL,
    selection_reason TEXT NOT NULL,
    expected JSONB NOT NULL DEFAULT '{}'::jsonb,
    provider_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    judge_result JSONB,
    recording_url TEXT,
    human_verdict TEXT CHECK (human_verdict IS NULL OR human_verdict IN ('pass', 'fail', 'inconclusive')),
    human_scores JSONB,
    reviewer_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    reviewed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS voice_eval_runs_created_idx ON voice_eval_runs (created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS voice_eval_runs_session_idx ON voice_eval_runs (session_id) WHERE session_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS voice_eval_turn_feedback (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES voice_eval_runs(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL CHECK (turn_number > 0),
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'disapproved', 'never_say_this', 'reduce_frequency')),
    comment TEXT,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    delivery_status TEXT,
    proposal_status TEXT NOT NULL DEFAULT 'not_requested'
        CHECK (proposal_status IN ('not_requested', 'candidate', 'failed', 'approved', 'rejected')),
    candidate_rule_id BIGINT,
    proposal_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, turn_number)
);
CREATE INDEX IF NOT EXISTS voice_eval_turn_feedback_queue_idx
    ON voice_eval_turn_feedback (proposal_status, created_at DESC);

ALTER TABLE conversation_pattern_rules
    ADD COLUMN IF NOT EXISTS priority INTEGER NOT NULL DEFAULT 100,
    ADD COLUMN IF NOT EXISTS source_feedback_id UUID,
    ADD COLUMN IF NOT EXISTS review_note TEXT;
