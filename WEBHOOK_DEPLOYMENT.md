# Webhook deployment

Install API dependencies:

```bash
uv sync --extra api
```

Set secrets only in the deployment environment:

```dotenv
AGENT_ENVIRONMENT=production
AGENT_API_KEY=replace-with-a-long-random-value
WEBHOOK_SIGNING_SECRET=replace-with-a-separate-long-random-value
```

Run the service on a private network interface or behind an HTTPS reverse proxy:

```bash
uv run uvicorn gnaniassignment.api.server:app --host 0.0.0.0 --port 8080
```

Endpoints:

- `GET /healthz` — load-balancer health check.
- `POST /v1/turn` — authenticated text turn using `X-API-Key`.
- `POST /v1/webhooks/call` — signed normalized call lifecycle/transcript event using `X-Webhook-Signature: sha256=<hex-hmac>`.

The generic call webhook payload is:

```json
{
  "event": "transcript.final",
  "call_id": "provider-call-123",
  "caller_id": "+919999999999",
  "transcript": "Mujhe Tushar se baat karni hai"
}
```

Use `call.started`, then one or more `transcript.final` events, followed by
`call.ended`. A telephony provider-specific adapter must convert its webhook
and live-media protocol into this normalized shape. Do not expose this process
directly to the public internet without TLS, a firewall, and webhook signing.

For one API process, call sessions are held in memory. For multiple cloud
replicas, route a call consistently to one replica or replace `SessionRegistry`
with a Redis-backed session/ownership adapter.
