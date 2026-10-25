-- TEST ONLY. Replace every chunk with approved business policy before production.
DELETE FROM approved_knowledge_chunks WHERE title LIKE 'MOCK-%';

INSERT INTO approved_knowledge_chunks (intent, title, content, priority) VALUES
('track_order', 'MOCK-order-tracking', 'TEST POLICY ONLY: Ask for an order ID, then explain that shipment status can be checked after verification. Never invent a delivery date or live order status.', 100),
('return_item', 'MOCK-return-request', 'TEST POLICY ONLY: Ask for an order ID and item condition before discussing a return. Do not promise eligibility, pickup, or a refund without an approved policy.', 100),
('cancel_order', 'MOCK-cancellation', 'TEST POLICY ONLY: Ask for an order ID and explain that cancellation depends on fulfilment status. Do not promise a cancellation.', 100),
('refund_status', 'MOCK-refund', 'TEST POLICY ONLY: Ask for an order ID or payment reference and explain that refund progress requires verification. Do not promise a refund date.', 100),
('wrong_item', 'MOCK-wrong-item', 'TEST POLICY ONLY: Ask for an order ID and a brief description of the received item. Do not promise replacement or pickup.', 100),
('size_issue', 'MOCK-size', 'TEST POLICY ONLY: Ask for an order ID, product, and requested size. Do not promise exchange eligibility.', 100),
('payment_issue', 'MOCK-payment', 'TEST POLICY ONLY: Ask for an order ID or payment reference. Do not claim payment success, failure, or refund status without verification.', 100),
('delivery_address', 'MOCK-address', 'TEST POLICY ONLY: Ask for an order ID. Explain that address changes depend on dispatch status; do not request a full address in voice chat.', 100);
