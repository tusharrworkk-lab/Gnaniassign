-- Safe mock data only: these IDs and utterances are fictitious.
-- Apply after episodic_memory.sql.

DELETE FROM episodic_memory_turns WHERE customer_id IN ('mock-customer-001', 'mock-customer-002');

INSERT INTO episodic_memory_turns
    (customer_id, session_id, query_text, answer_text, summary, intent, topic, occurred_at)
VALUES
    ('mock-customer-001', 'mock-session-a', 'Mera parcel abhi tak nahi aaya.', 'Main shipment status check karne ke liye order ID maangungi.', 'Customer is waiting for a delayed parcel and needs shipment status.', 'track_order', 'timeline', now() - interval '20 minutes'),
    ('mock-customer-001', 'mock-session-a', 'Order ID <ORDER_ID> hai.', 'Order ID was provided; next step is to check shipment status.', 'Customer provided the typed order-ID placeholder for shipment lookup.', 'track_order', 'timeline', now() - interval '18 minutes'),
    ('mock-customer-002', 'mock-session-b', 'Payment cut gaya lekin order create nahi hua.', 'Main payment issue ke liye order details verify karungi.', 'Customer reports debited payment with no order creation.', 'payment_issue', 'operations', now() - interval '10 minutes'),
    ('mock-customer-002', 'mock-session-b', 'Reference <PAYMENT_REF> hai.', 'Payment reference was provided for a payment investigation.', 'Customer provided a typed payment-reference placeholder.', 'payment_issue', 'operations', now() - interval '8 minutes');
