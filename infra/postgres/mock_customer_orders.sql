-- Fictitious records for local demonstrations. Never use these in production.
DELETE FROM customer_orders WHERE customer_id = 'mock-customer-001';

INSERT INTO customer_orders
    (customer_id, order_id, item_name, order_status, payment_status, delivery_note, refund_status)
VALUES
    ('mock-customer-001', 'ORD-1042', 'Wireless headphones', 'shipped', 'paid', 'Parcel is in transit; no confirmed delivery date is available.', NULL),
    ('mock-customer-001', 'ORD-7731', 'Cotton kurta, size M', 'delivered', 'paid', NULL, 'Refund request received; verification is pending.'),
    ('mock-customer-001', 'ORD-8820', 'Running shoes, size 8', 'delivered', 'paid', NULL, NULL);
