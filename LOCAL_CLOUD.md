# Local cloud rehearsal

This Compose stack exercises the same service boundaries as deployment:

```text
host port 8080 -> API container -> PostgreSQL container
                            \-> Redis container
                            \-> Bedrock and Gnani (hosted services)
```

PostgreSQL and Redis have no host ports. The API is the only exposed service.
Docker volumes preserve their data across restarts. This is a single-machine
practice environment, not a high-availability production database setup.

## First run

1. Install Docker Engine and the Docker Compose plugin.
2. Create the ignored local environment file:

   ```bash
   cp .env.local-cloud.example .env.local-cloud
   ```

3. Set a strong `POSTGRES_PASSWORD`, `AGENT_API_KEY`, and
   `WEBHOOK_SIGNING_SECRET`. Add the Gnani and Bedrock credentials you will
   use. For Bedrock IAM credentials, make them available to the container by
   your normal deployment method; for a simple local rehearsal, set
   `BEDROCK_AUTH_MODE=api_key` and add `BEDROCK_API_KEY` instead.
4. Build and start the stack:

   ```bash
   docker compose -f docker-compose.local-cloud.yml up --build -d
   docker compose -f docker-compose.local-cloud.yml ps
   ```

5. Confirm the API is ready:

   ```bash
   curl http://127.0.0.1:8080/healthz
   ```

6. Exercise one authenticated turn:

   ```bash
   curl -X POST http://127.0.0.1:8080/v1/turn \
     -H 'Content-Type: application/json' \
     -H 'X-API-Key: replace-with-your-AGENT_API_KEY' \
     --data '{"customer_id":"practice-user-001","session_id":"practice-call-001","text":"Mujhe kal subah mummy ko call karna yaad dila dena."}'
   ```

## Observe and operate

```bash
docker compose -f docker-compose.local-cloud.yml logs -f api
docker compose -f docker-compose.local-cloud.yml stats
docker compose -f docker-compose.local-cloud.yml exec postgres psql -U agent -d agent
```

Useful database checks:

```sql
SELECT customer_id, query_text, answer_text, occurred_at
FROM episodic_memory_turns
ORDER BY id DESC
LIMIT 10;

SELECT session_id, turn_number, user_text, response_text, routed_intent
FROM conversation_turns
ORDER BY id DESC
LIMIT 10;
```

## Safe reset for practice data only

This deletes the local Compose database and Redis volumes. Never use it for
real customer data.

```bash
docker compose -f docker-compose.local-cloud.yml down -v
```

## Mapping to AWS

| Local rehearsal | AWS deployment |
| --- | --- |
| `api` container | ECS Fargate task behind ALB |
| Compose private network | VPC private subnets/security groups |
| PostgreSQL container/volume | RDS PostgreSQL |
| Redis container/volume | ElastiCache Redis |
| `.env.local-cloud` | Secrets Manager + ECS task secrets |
| `curl /healthz` | ALB target-group health check |
