# Episodic memory in PostgreSQL

Install the optional driver, create the schema, then seed only the fictitious
test rows:

```bash
uv sync --extra postgres-memory
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/episodic_memory.sql
psql "$EPISODIC_MEMORY_DATABASE_URL" -f infra/postgres/mock_episodic_memory.sql
```

For a local mock run, set `EPISODIC_MEMORY_DATABASE_URL` and
`EPISODIC_MEMORY_CUSTOMER_ID=mock-customer-001`. A new `TurnSession` restores
that customer's latest 100 turns. Completed turns are then appended to the
same customer's episodic history.

Do not use a fixed environment customer ID in production. Pass the authenticated
caller or account ID to `create_session(customer_id=...)`. Set a retention and
deletion policy before storing real customer transcripts.
