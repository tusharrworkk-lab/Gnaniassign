-- Fictional Nukkad Fresh neighborhood-grocery scenario for local testing.
-- It deliberately uses customer IDs, not names or addresses.

DELETE FROM customer_orders WHERE customer_id LIKE 'grocery-customer-%';

INSERT INTO customer_orders
    (customer_id, order_id, item_name, order_status, payment_status, delivery_note, refund_status)
VALUES
    ('grocery-customer-001', 'GRC-10001', 'Aashirvaad atta 5 kg, milk 2 L, bananas 1 kg', 'out_for_delivery', 'paid', 'Rider is on the Green Park route; expected within 25 minutes.', NULL),
    ('grocery-customer-001', 'GRC-10002', 'Tomatoes 1 kg, onions 1 kg, coriander', 'delivered', 'paid', 'Delivered at 09:18; no live rider update remains.', NULL),
    ('grocery-customer-001', 'GRC-10003', 'Paneer 500 g, curd 400 g, bread', 'delivered', 'paid', NULL, 'Refund request received for damaged paneer; verification pending.'),
    ('grocery-customer-001', 'GRC-10004', 'Rice 5 kg, dal 1 kg, cooking oil 1 L', 'packed', 'paid', 'Not yet assigned to a rider; delivery address can still be reviewed.', NULL),
    ('grocery-customer-002', 'GRC-20001', 'Eggs 12 pack, milk 1 L, brown bread', 'delivered', 'paid', NULL, NULL),
    ('grocery-customer-002', 'GRC-20002', 'Potatoes 2 kg, onions 2 kg, ginger 250 g', 'out_for_delivery', 'paid', 'Rider is on the Lake View route; expected within 35 minutes.', NULL),
    ('grocery-customer-002', 'GRC-20003', 'Basmati rice 5 kg, moong dal 1 kg', 'payment_pending', 'debited_pending', 'Order was not packed because payment confirmation is pending.', NULL),
    ('grocery-customer-002', 'GRC-20004', 'Apples 1 kg, oranges 1 kg, grapes 500 g', 'delivered', 'paid', NULL, 'Refund request approved; bank credit is pending.'),
    ('grocery-customer-003', 'GRC-30001', 'Spinach, methi, coriander, tomatoes', 'packed', 'paid', 'Not yet assigned to a rider.', NULL),
    ('grocery-customer-003', 'GRC-30002', 'Yogurt 400 g, butter 100 g, cheese slices', 'delivered', 'paid', NULL, NULL),
    ('grocery-customer-003', 'GRC-30003', 'Toor dal 1 kg, sugar 1 kg, tea 500 g', 'cancelled', 'refund_initiated', NULL, 'Cancellation refund initiated; bank credit is pending.'),
    ('grocery-customer-003', 'GRC-30004', 'Dishwash liquid, detergent, garbage bags', 'out_for_delivery', 'paid', 'Rider is on the Market Road route; expected within 20 minutes.', NULL),
    ('grocery-customer-004', 'GRC-40001', 'Mangoes 1 kg, watermelon, bananas 1 kg', 'delivered', 'paid', NULL, 'Refund request received for bruised mangoes; verification pending.'),
    ('grocery-customer-004', 'GRC-40002', 'Milk 2 L, eggs 12 pack, bread', 'packed', 'paid', 'Not yet assigned to a rider; delivery address can still be reviewed.', NULL),
    ('grocery-customer-004', 'GRC-40003', 'Jeera 100 g, turmeric 100 g, chilli powder 100 g', 'delivered', 'paid', NULL, NULL),
    ('grocery-customer-004', 'GRC-40004', 'Flour 5 kg, cooking oil 1 L, salt 1 kg', 'payment_pending', 'debited_pending', 'Order was not packed because payment confirmation is pending.', NULL),
    ('grocery-customer-005', 'GRC-50001', 'Cauliflower, cabbage, carrots, peas', 'out_for_delivery', 'paid', 'Rider is on the Temple Street route; expected within 30 minutes.', NULL),
    ('grocery-customer-005', 'GRC-50002', 'Almonds 500 g, cashews 500 g, raisins 250 g', 'delivered', 'paid', NULL, NULL),
    ('grocery-customer-005', 'GRC-50003', 'Idli batter 1 kg, coconut chutney, curd', 'delivered', 'paid', NULL, 'Refund request approved; bank credit is pending.'),
    ('grocery-customer-005', 'GRC-50004', 'Biscuits, juice, breakfast cereal', 'packed', 'paid', 'Not yet assigned to a rider.', NULL);
