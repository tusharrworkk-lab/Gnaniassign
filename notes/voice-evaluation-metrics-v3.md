# Voice evaluation metrics v3

The live VoiceLink pipeline writes one metadata-only row per committed caller
turn to `voice_turn_evaluations`. The authenticated report is available at:

```bash
curl -sS -H "X-API-Key: $AGENT_API_KEY" \
  "https://voice.sidecut.me/v1/dashboard/evals/voice?limit=100"
```

Schema v3 deliberately excludes schema v1/v2 rows from percentiles. Earlier
rows mixed queue/prompt work into “model TTFT” and allowed a latency cue to
count as the answer. The API reports how many legacy rows were excluded.

## Measured online

- Endpointing: speech end to commit, provider-final wait, semantic-VAD wait,
  and committed-turn queue delay.
- Prompt path: account context, router, context assembly, retrieval and prompt
  build time; prompt characters, estimated tokens and a non-secret prompt hash.
- LLM: Bedrock request setup, provider TTFT, first decoded answer character,
  total stream time and provider token usage when Bedrock returns it.
- TTS: decoded text to first speakable fragment, reconnect time, Gnani TTFA,
  zero-audio retries and reconnects.
- Telephony: first response frame and first real answer frame separately, first answer
  frame hand-off, frame-gap p50/p95/max across every answer fragment, and media
  duration derived from 20 ms frames.
- Reliability: model/JSON/TTS failures, silence, interruption and partial
  delivery.

`first_answer_audio_ms` is the earliest response audio and therefore includes a
clear-throat cue when one is emitted. `first_real_answer_audio_ms` ends when the
server hands the first spoken-answer frame to VoiceLink. Neither proves handset
playback because VoiceLink does not expose that acknowledgement. Silent-turn
evaluation uses `first_real_answer_audio_ms`, so a cue cannot hide a lost answer.

## Screening signals

Repeated greetings, repeated clarifications, caller corrections, possible
unsupported action claims and privacy-boundary handling are deterministic
screening flags. They identify turns for review; they are not a substitute for
labelled human or judge evaluation.

## Requires labelled replay data

Hinglish WER, false endpoints, premature turns, missed turns and noise triggers
cannot be inferred honestly from live VAD/STT events. Add reference transcripts
or explicit annotations to an offline evaluation set. `summarize_wer` computes
corpus WER from `reference_text` and `hypothesis_text`; the voice report only
aggregates endpoint/noise fields on rows where those booleans were labelled.

## Privacy

Metric rows contain timing, sizes, counts, booleans and prompt-version hashes.
They do not persist prompts, raw audio, STT text or LLM answers.
