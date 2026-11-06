# VoiceLink call-flow incident — 6 November 2026

## Scope

This note records the investigation and remediation of two linked VoiceLink
telephone-call problems:

1. The caller would speak and sometimes receive no reply.
2. The agent would become stuck in a repetitive loop, asking for the same
   detail after each short fragment of one caller sentence.

No credentials, WebSocket tokens, caller numbers, or other secrets are stored
in this note.

## Supporting work completed in this session

### Text conversation harness

A protected, text-only LLM harness was added to isolate multi-turn agent
behavior from STT, TTS, and telephony:

```text
POST /v1/harness/conversations
GET  /v1/harness/conversations/{conversation_id}
POST /v1/harness/conversations/{conversation_id}/turn
POST /v1/harness/conversations/{conversation_id}/end
```

It uses the normal `TurnSession`, so it preserves the actual bounded history,
working memory, prompts, routing, intent/topic result, and per-turn metrics.
It is protected by `X-API-Key`. Harness sessions intentionally live only in
the active API process; an API restart clears unfinished harness sessions.

Clean production commit:

```text
e322696 Add LLM conversation harness API
```

### Mobile dashboard harness UI

The React/Vite mobile dashboard now exposes the harness under:

```text
Setup -> Test chat
```

It can create a clean conversation, send sequential messages, display the
resulting user/agent transcript plus intent/topic, inspect current working
memory, refresh state, and end the test. The browser adapter uses the API URL
and key only from local Vite environment variables. A production WebView must
not embed `VITE_AGENT_API_KEY`; it should eventually use an authenticated
same-origin backend session.

Clean production commit:

```text
0ad76bc Add dashboard conversation harness
```

## Relevant call path

```text
VoiceLink inbound A-law 8 kHz media
  -> VoiceLink adapter (`telephony/voicelink.py`)
  -> PCM conversion and 16 kHz Gnani realtime STT
  -> final STT transcript(s)
  -> TurnSession / LangGraph / Bedrock GPT-OSS
  -> Gnani TTS PCM
  -> A-law 8 kHz media sent back to VoiceLink
```

The VoiceLink adapter owns call media and telephony turn-taking. The agent
graph owns memory, prompt selection, routing, and LLM response generation.

## Incident 1: agent listened but did not reply

### Evidence

The production VoiceLink call traceback showed:

```text
ValueError: GPT-OSS returned no final text (stopReason='max_tokens')
```

The failure occurred in the Bedrock GPT-OSS stream while the VoiceLink STT
receiver awaited the LLM generation task. The exception escaped the receiver
task, so the WebSocket remained open but that call stopped processing later
STT turns. To a caller this appeared as: the agent heard the speech, then went
silent.

### Cause

`BEDROCK_GPT_OSS_MAX_OUTPUT_TOKENS` had been reduced to `80` to constrain
voice length. GPT-OSS counts reasoning and visible answer generation against
the same budget. At 80 tokens it occasionally exhausted the budget during
reasoning and emitted no caller-facing text.

### Remediation

1. Restored production `BEDROCK_GPT_OSS_MAX_OUTPUT_TOKENS=256` in the VM
   runtime environment.
2. Recreated the API container using its existing image so the setting took
   effect immediately.
3. Confirmed `https://voice.sidecut.me/healthz` returned HTTP 200.
4. Added a per-turn exception boundary in
   `src/gnaniassignment/telephony/voicelink.py`.
   - An LLM/TTS failure is logged and recorded as an `agent.failed` runtime
     event.
   - The rest of the call's STT receiver remains alive.
   - The caller receives a short retry message rather than unexplained
     silence.

Committed and pushed as clean production commit:

```text
78976c2 Keep VoiceLink calls alive after LLM failures
```

## Incident 2: repetitive agent loop

### Evidence

The persisted `conversation_turns` ledger showed one VoiceLink session with
many consecutive turns that were fragments of a single caller message. For
example, individual turns contained partial statements equivalent to:

```text
"aaj shaam ko party hai toh"
"aaj shaam ko party hai toh shaam"
"chhah baje hai kuen ke paas"
"chhah baje kuen ke paas hello"
```

Each partial turn made the LLM ask again for the missing time, place, or
message content. This was not a MiniLM classification issue and not primarily
an LLM-memory issue: the LLM was repeatedly receiving incomplete user turns.

### Cause

Gnani STT can emit a final transcript after a natural pause inside a longer
utterance. The original VoiceLink adapter immediately dispatched every final
STT segment to the agent graph.

Additionally, because each reply was handled serially, final STT events that
arrived while the agent was generating or speaking could remain queued. Once
the reply finished, stale fragments were then processed as fresh requests.

### Remediation

Updated `src/gnaniassignment/telephony/voicelink.py`:

1. Added a `VOICE_LINK_TURN_COALESCE_MS` turn-boundary window.
   - Default: `1100` ms.
   - Allowed range: 250–3000 ms.
   - Closely spaced final STT segments are joined into one user utterance
     before the LLM is invoked.
2. Added a per-call turn worker queue.
   - It drains closely spaced STT final segments, constructs one combined
     `VoiceTranscript`, and invokes the agent once.
3. Added `reply_in_progress` to the per-call VoiceLink state.
   - Inbound STT transcripts received while an LLM/TTS reply is in progress
     are discarded rather than replayed afterward.
   - A `transcript.discarded` runtime event is emitted with the reason
     `agent_reply_in_progress`.

This is deliberately non-barge-in behavior. A caller who speaks over the
agent's reply is not yet allowed to interrupt it; their overlapping speech is
discarded. This is preferable to replaying old fragments and creating a
conversation loop. Proper barge-in requires a separate design: cancel active
LLM/TTS work, send VoiceLink `clear`, and then prioritise the new caller turn.

Committed and pushed as clean production commit:

```text
3b0246f Coalesce VoiceLink speech segments into turns
```

## Validation completed

- `uv run python -m compileall -q src`
- `uv run python -m unittest tests.test_enquiry_pipeline -v`
- Result: **49 tests passed**.
- Production health endpoint returned HTTP 200 after the output-token setting
  change.
- The regular production GitHub Actions deployment was triggered by the two
  pushed commits above.

## Expected behavior after deployment

1. Caller speaks a complete sentence.
2. The adapter waits roughly 1.1 seconds after the last nearby final STT
   segment.
3. It joins those segments and invokes the LLM once.
4. The agent gives one concise reply.
5. Speech occurring while the agent is already replying is not replayed as a
   later, stale turn.

## What to test next

Use one complete message with natural pauses, for example:

```text
Tushar ko batana ki aaj shaam ko party hai, chhah baje kuen ke paas aana hai.
```

Expected result: one agent response acknowledging/confirming the complete
message, rather than several questions about partial fragments.

If the agent still repeats after this deployment, inspect in this order:

1. `conversation_turns` for the exact STT text and reply sequence.
2. `GET /v1/dashboard/latency` for STT-final and LLM timing per call.
3. Redacted API logs for `agent.failed`, `stt.failed`, and
   `transcript.discarded` events.
4. Whether the caller began speaking while the agent was replying; that is
   currently intentionally suppressed until a true barge-in implementation is
   added.

