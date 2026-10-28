-- Shadow predictions are not training labels until a reviewer fills reviewed_intent.
CREATE TABLE IF NOT EXISTS intent_shadow_observations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id TEXT,
    query_text TEXT NOT NULL,
    predicted_intent TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    probabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
    reviewed_intent TEXT,
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (reviewed_intent IS NULL OR reviewed_intent IN
        ('track_order', 'return_item', 'cancel_order', 'refund_status', 'wrong_item',
         'size_issue', 'payment_issue', 'delivery_address', 'other'))
);

ALTER TABLE intent_shadow_observations
    ADD COLUMN IF NOT EXISTS session_id TEXT;

CREATE INDEX IF NOT EXISTS intent_shadow_unreviewed_idx
    ON intent_shadow_observations (reviewed_intent, created_at DESC);
