# Deferred improvements

## Caller order-window refresh

The current design preloads a small, caller-scoped order window before a voice
call and refreshes it on every turn. This keeps the order context current, but
adds a PostgreSQL lookup to the turn path.

For the neighborhood-grocery use case, prefer loading the recent orders once
at call start and retaining them in session memory. Re-fetch only when:

- the caller explicitly names a different order ID;
- the order backend can provide genuinely newer live status; or
- a long call reaches a defined freshness limit.

Before changing this, measure database lookup latency and verify that status
updates during calls are common enough to justify the extra turn latency.
