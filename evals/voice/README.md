# Voice evaluation data

The production collector stores metadata only: timestamps, reply/frame state,
and interruption state. It does not put caller audio or transcripts in this
directory or the evaluation database.

`turns.jsonl` is an export of the protected `/v1/dashboard/evals/voice`
records. Each line contains either a turn record or `{ "metrics": ... }`.

For an offline end-of-turn benchmark, add `reference_turn_end_ms` to a turn
record. It must come from a human annotation, not from the VAD being measured.

`stt-labelled.jsonl` is local-only and contains consented/synthetic examples:

```json
{"id":"hinglish-001","reference_text":"kal meri party hai chhah baje","hypothesis_text":"kal meri party hai chhah baje"}
```

Copy the versioned examples to begin a controlled benchmark. Do not commit
real caller audio, phone numbers, or unredacted production transcripts.

```bash
cp evals/voice/turns.jsonl.example /tmp/voice-turns.jsonl
cp evals/voice/stt-labelled.jsonl.example /tmp/hinglish-stt.jsonl
```

Run a report with:

```bash
uv run python -m gnaniassignment.pipeline.voice_eval_report \
  --turns evals/voice/turns.jsonl \
  --stt-labels evals/voice/stt-labelled.jsonl \
  --output artifacts/voice-eval-report.json
```

`first_audible_reply_proxy` means the first A-law media frame handed from this
server to VoiceLink. The current telephony interface does not confirm the
moment the caller's handset plays it, so published reports must keep the
`proxy` qualifier.

For every benchmark record, use the human-labelled
`reference_turn_end_ms` to calculate premature turns. A live VAD boundary is
never an independent ground truth source. The offline report treats an agent
frame timestamp before this annotation as a premature turn.

## Live operational view

After deployment, the protected dashboard endpoint provides the live, rolling
turn-level measure and its exact source records:

```bash
curl -sS -H "X-API-Key: $AGENT_API_KEY" \
  "https://voice.sidecut.me/v1/dashboard/evals/voice?limit=500&silent_after_ms=2500"
```

It returns:

- `model_ttft`: p50, p95 and p99 from turn processing start to first LLM
  token. This is the model/service figure and deliberately excludes a queued
  earlier turn.
- `e2e_llm_ttft`: p50, p95 and p99 from committed transcript to first LLM
  token. Compare this with `model_ttft` to identify endpointing or queueing
  delay.
- `turn_queue_delay`: p50, p95 and p99 time a committed caller turn waited
  before processing started. This is a pipeline-health metric, not model
  latency.
- `first_audible_reply_proxy`: p50, p95 and p99 milliseconds from committed
  caller turn to the first outbound media frame handed to VoiceLink.
- `silent_turn_rate`: completed turns with no outbound frame within 2.5 s
  (change `silent_after_ms` when viewing, or `VOICE_EVAL_SILENT_TURN_TIMEOUT_MS`
  for the stored boolean).
- `barge_in_stop_latency`: intentionally has zero samples until the call
  adapter sends a VoiceLink `clear`/stop on interruption. Detected attempts are
  still counted; no latency is invented.
- `premature_turn_rate`: remains `null` until records are human-labelled with
  `reference_turn_end_ms`.

Live records include `evaluation_schema_version`. The dashboard only combines
the current schema version; historical records from an earlier schema are
reported as excluded rather than being silently mixed into the current p95.

For a public eval, use a consented or synthetic fixed test suite with: clean
and noisy Hinglish speech, pauses inside a sentence, code-switching, short
confirmations, interruptions, and multi-turn requests. Keep the audio and
labels access-controlled; publish aggregate scores and the test protocol, not
call recordings.
