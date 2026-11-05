# Nightly MiniLM review queue

The live agent remains generic. It records completed user/assistant turns in
PostgreSQL. This job runs later, loads the retail MiniLM once, labels only rows
where `classifier_intent` is empty, and optionally sends candidate-label review
cards to a private Telegram channel. It never retrains MiniLM and never changes
`final_intent`; human review remains required.

## Prerequisites

The machine that runs the job needs the trained model directory and CPU PyTorch:

```bash
uv sync --extra retail-classifier --extra postgres-memory
```

Set these in the job environment (or its ignored `.env`):

```env
RETAIL_MINILM_MODEL=/opt/agent/models/systemone-retail-v1
RETAIL_MINILM_DEVICE=cpu
CONVERSATION_LOG_DATABASE_URL=postgresql://...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_REVIEW_CHAT_ID=...
TELEGRAM_REVIEW_MAX_ITEMS=20
```

Create a Telegram bot with BotFather, add it as an administrator to a **private**
review channel, and use that channel's chat ID. The digest includes STT text, so
do not use a public channel and obtain the appropriate caller consent before
sending real transcripts there.

## Run once

First check the pending data without changing it:

```bash
uv run --extra retail-classifier --extra postgres-memory \
  python -m gnaniassignment.pipeline.nightly_intent_review --dry-run
```

Then classify up to 500 unlabelled turns and send at most 20 review cards:

```bash
uv run --extra retail-classifier --extra postgres-memory \
  python -m gnaniassignment.pipeline.nightly_intent_review \
  --limit 500 --telegram --telegram-max-items 20
```

Review predictions using the stored turn IDs:

```bash
uv run --extra postgres-memory python -m gnaniassignment.pipeline.review_conversations --limit 50
```

Only reviewed labels are suitable for export and later fine-tuning:

```bash
uv run --extra postgres-memory \
  python -m gnaniassignment.pipeline.export_reviewed_conversations \
  --output data/external/retail-conversations-reviewed-v1.jsonl
```

## Schedule at 02:15 every night

On the VM, add a cron entry with `crontab -e`. Use absolute paths that match
your deployment directory and `uv` binary:

```cron
15 2 * * * cd /opt/personal-aiagent && /home/ubuntu/.local/bin/uv run --extra retail-classifier --extra postgres-memory python -m gnaniassignment.pipeline.nightly_intent_review --limit 500 --telegram >> /var/log/personal-agent/nightly-intent-review.log 2>&1
```

Before enabling cron, run the command manually once. The job is idempotent for
already-classified rows: they are not selected again unless you explicitly
clear `classifier_intent` in PostgreSQL.
