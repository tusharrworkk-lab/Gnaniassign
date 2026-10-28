# Episodic memory in PostgreSQL

Install the optional driver, create the schema, then seed only the fictitious
test rows:

```bash
uv sync --extra postgres-memory
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/episodic_memory.sql
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/mock_episodic_memory.sql
```

To collect shadow MiniLM predictions for later human review and retraining:

```bash
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/intent_feedback.sql
```

Enable `SHADOW_INTENT_ENABLED=true` while running the generic agent. Review
labels locally with `uv run python -m gnaniassignment.pipeline.review_shadow_intents`,
then export the reviewed JSONL with
`uv run python -m gnaniassignment.pipeline.export_reviewed_intents --output data/external/retail-reviewed-v1.jsonl`.
Reviewed customer text is sensitive and is ignored by Git.

For the voice-agent conversation ledger and final-label training workflow:

```bash
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/conversation_analytics.sql
uv run python -m gnaniassignment.pipeline.review_conversations --flagged-only
uv run python -m gnaniassignment.pipeline.export_reviewed_conversations \
  --output data/external/retail-conversations-reviewed-v1.jsonl
```

Set `CONVERSATION_LOGGING_ENABLED=true` to capture turns. Set
`ACIL_TRACE_ENABLED=true` plus `ACIL_TRACE_URL`, `ACIL_TRACE_PROJECT_ID`, and a
scoped `ACIL_TRACE_API_KEY` to mirror the same completed turns and detected
flow failures to ACIL. Transcript logging is sensitive: configure retention,
access controls, consent, and deletion before real-customer use.

Set `CONVERSATION_JUDGE_ENABLED=true` to submit the complete STT/LLM call
record only after microphone capture ends. Intent labels are evaluated in a
separate Bedrock request that receives **only final STT text**. A second request
receives STT/LLM turn pairs solely to evaluate conversation flow, including
observable user negation or clarification. The second judge creates `candidate`
do/don't rules; it does not alter live prompts by itself. Inspect and explicitly
activate a candidate with:

```bash
uv run python -m gnaniassignment.pipeline.manage_conversation_rules
uv run python -m gnaniassignment.pipeline.manage_conversation_rules --activate <rule-id>
```

For local mock order lookups, also apply:

```bash
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/customer_orders.sql
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/mock_customer_orders.sql
```

For the fictional Nukkad Fresh neighborhood-grocery scenario, apply the order
schema and grocery seed instead:

```bash
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/customer_orders.sql
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/grocery_neighborhood_seed.sql
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/grocery_knowledge_base.sql
```

Set `EPISODIC_MEMORY_CUSTOMER_ID=grocery-customer-001` to test four orders:
`GRC-10001` (out for delivery), `GRC-10002` (delivered), `GRC-10003` (refund
verification pending), and `GRC-10004` (packed/address-review eligible).

In a voice-call integration, resolve the authenticated caller before starting
the session. The microphone runner simulates this with:

```bash
uv run python -m gnaniassignment.pipeline.gnani_microphone \
  --device plughw:1,0 --with-llm --with-tts \
  --customer-id grocery-customer-001
```

The order window is loaded before the call and refreshed per turn. A spoken
`GRC-*` or `ORD-*` ID narrows the context to that order; otherwise only a few
recent orders are supplied.

For a local mock run, set `EPISODIC_MEMORY_DATABASE_URL` and
`EPISODIC_MEMORY_CUSTOMER_ID=mock-customer-001`. A new `TurnSession` restores
that customer's latest 100 turns. Completed turns are then appended to the
same customer's episodic history.

Do not use a fixed environment customer ID in production. Pass the authenticated
caller or account ID to `create_session(customer_id=...)`. Set a retention and
deletion policy before storing real customer transcripts.
