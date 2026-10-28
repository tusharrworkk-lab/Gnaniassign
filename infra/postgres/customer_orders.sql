-- Mock-order schema for local voice-pipeline testing only.
CREATE TABLE IF NOT EXISTS customer_orders (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    item_name TEXT NOT NULL,
    order_status TEXT NOT NULL,
    payment_status TEXT NOT NULL,
    delivery_note TEXT,
    refund_status TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (customer_id, order_id)
);

CREATE INDEX IF NOT EXISTS customer_orders_customer_idx
    ON customer_orders (customer_id, updated_at DESC, id DESC);
