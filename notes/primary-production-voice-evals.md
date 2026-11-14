# Primary voice evaluations: real telephony and production sampling

The primary evaluation path uses the actual VoiceLink telephone leg. It does
not claim that a local WebSocket simulator measures carrier or handset quality.

## Enable

Set these only in the production secret environment:

```dotenv
VOICE_PRODUCTION_EVALS_ENABLED=true
VOICE_PRODUCTION_EVAL_SAMPLE_RATE=0.10
VOICELINK_ACCESS_TOKEN=<VoiceLink bearer token>
VOICELINK_EVAL_DID_NUMBER=<purchased outbound DID>
```

The DID must have active outbound WebSocket routing. For Indian destinations,
the API sends a ten-digit `customer_number` and the separate country code `91`.

## Layer 4: controlled real-telephone call

List the versioned scripts:

```bash
curl -sS -H "X-API-Key: $AGENT_API_KEY" \
  https://voice.sidecut.me/v1/dashboard/evals/telephony/scenarios
```

Queue a call to the evaluator's phone:

```bash
curl -sS -X POST -H "X-API-Key: $AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"scenario_id":"personal-message-correction-v1","customer_number":"9876543210"}' \
  https://voice.sidecut.me/v1/dashboard/evals/telephony/runs
```

Answer the call and follow the returned caller script exactly. The outbound
lead carries only the run/scenario correlation identifiers, never the expected
answer. The normal VoiceLink WebSocket pipeline records turn metrics and the
post-call Bedrock judge scores the speech actually handed to the caller.

After VoiceLink has processed its call log, synchronize recording metadata:

```bash
curl -sS -X POST -H "X-API-Key: $AGENT_API_KEY" \
  https://voice.sidecut.me/v1/dashboard/evals/telephony/runs/RUN_ID/sync-provider
```

The system stores the VoiceLink recording URL rather than copying recordings
into PostgreSQL. DID call recording must be enabled in VoiceLink for that URL
to exist.

## Layer 5: sampled production calls

Every completed call receives the existing end-of-call judge. A stable hash of
the session ID samples the configured percentage. Calls are also selected when
there is a judge failure, failed quality verdict, requested review, critical
failure, or interrupted/failed delivery. Reprocessing the same call cannot
change its sampling decision or create a duplicate sample.

View all runs or only the review queue:

```bash
curl -sS -H "X-API-Key: $AGENT_API_KEY" \
  'https://voice.sidecut.me/v1/dashboard/evals/telephony/runs?limit=100&pending_only=true'
```

Submit the human verdict after reading the transcript, metrics and judge
evidence and—when present—listening to the recording:

```bash
curl -sS -X POST -H "X-API-Key: $AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"verdict":"pass","scores":{"task_completion":4,"naturalness":3,"audio_quality":3},"notes":"Correct result; one slightly long pause."}' \
  https://voice.sidecut.me/v1/dashboard/evals/telephony/runs/RUN_ID/review
```

Scores use explicit 0–4 anchors: unmet, minimal, partial, mostly satisfied and
fully satisfied. Human verdicts remain the release-quality ground truth; judge
scores prioritize the queue and provide diagnostic evidence.

## Turn-level answer feedback and rule review

The private call-detail screen offers four decisions for every answer that was
actually handed off as speech: `approved`, `disapproved`, `never_say_this`, and
`reduce_frequency`. The reviewer may also add a comment.

A plain approval is stored as positive eval evidence and creates no rule. A
negative decision—or an approval with a useful comment—asks Bedrock for one
short, generalized rule candidate. The generator does not receive the model's
unheard answer; interrupted turns include only the text handed to the caller.

```bash
curl -sS -X POST -H "X-API-Key: $AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"decision":"reduce_frequency","comment":"Confirm once, then move the conversation forward."}' \
  https://voice.sidecut.me/v1/dashboard/evals/telephony/runs/RUN_ID/turns/2/feedback
```

Generated rules are always candidates. They cannot enter a live call until an
admin reviews the wording and supplies an explicit priority:

```bash
curl -sS -H "X-API-Key: $AGENT_API_KEY" \
  'https://voice.sidecut.me/v1/dashboard/evals/rules?status=candidate'

curl -sS -X POST -H "X-API-Key: $AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"decision":"approve","priority":40,"rule_text":"Confirm a captured message once; do not repeat it unless the caller corrects it."}' \
  https://voice.sidecut.me/v1/dashboard/evals/rules/RULE_ID/review
```

Lower priority numbers are rendered earlier. Active rules are read once while
a new call session is created, so rule generation adds no latency to an ongoing
call. Rejection keeps the candidate and its evidence for audit without adding
it to the prompt.

## Privacy and safety

- All endpoints require `AGENT_API_KEY`.
- The VoiceLink token and evaluator phone number are never returned or logged.
- Production sampling is stable and bounded.
- Provider metadata is allow-listed before persistence.
- Expected outcomes are hidden from the live agent.
- LLM-generated rules never self-activate; admin approval and priority are
  mandatory.
- Recording URLs inherit VoiceLink access controls and must not be exposed in a
  public dashboard.
