# Voice agent handoff — 2026-11-07

## Read this first

The current Codex sandbox mounts `.git` read-only. Project files remain
writable, but this session cannot commit or push. A fresh Codex workspace must
verify Git metadata access before deployment:

```bash
cd /home/tushar/CODES/Gnaniimplerment
touch .git/write-test && rm .git/write-test
```

Do **not** delete or stage the unrelated untracked directories:

```text
.github/agents/
.github/hooks/
.github/skills/
.impeccable/
```

## Deployment state

- GitHub `origin/master` last known commit: `eaa12c6 Keep continuing-call prompt regression wording`.
- Production VM last verified image: `asia-south1-docker.pkg.dev/gen-lang-client-0813070445/voice-agent/voice-agent:e8f475c...`.
- Therefore production has **not** received the conversational-memory/TTS
  reconnect changes or the newer uncommitted endpoint/trace changes.
- GCP project: `gen-lang-client-0813070445`
- VM: `voice-agent-01`, zone `asia-south1-a`.
- Public API / VoiceLink host: `https://voice.sidecut.me`.
- Never expose `VOICELINK_WS_TOKEN`, `AGENT_API_KEY`, AWS credentials, or
  database passwords in commits, logs, or replies.

## Confirmed production behavior and evidence

1. A real VoiceLink call logged `ConnectionClosedError` from Gnani TTS:
   `keepalive ping timeout`; its fallback TTS reply failed as well. This is a
   confirmed source of silent responses after an otherwise successful LLM turn.
2. The old VoiceLink turn worker starts only after Gnani STT emits a transcript:
   `first = await turns.get()`. TEN/Pipecat Smart Turn can detect a completed
   pause but previously could not commit text by itself. This is a likely cause
   of “I say something, it is silent, then it responds after a nudge.”
3. The VoiceLink payload conversion is structurally correct: inbound 8 kHz
   A-law → 16 kHz PCM for STT; outbound TTS PCM → 8 kHz A-law in 20 ms frames.
   There is no evidence yet proving a codec mismatch causes distorted audio.

## Committed but not verified deployed changes

Commit `18de1fb` (promoted to production branch as `83aab4d`) contains:

- TTS reconnect before a reply when an idle provider WebSocket closed.
- Per-reply hand-off metrics: TTS first audio, first/last outbound frame,
  frame count, final mark.
- Compact per-caller call summaries instead of loading raw historic turns into
  a new call.
- Mandatory active-call context for short follow-ups such as “haan, bata do”.
- Generic prompt prevents repeated questions and false “message delivered”
  claims without a real tool.

Commit `8061055` / production `eaa12c6` only corrected a prompt regression
test phrase: it requires the literal text `continuing call`.

## Current uncommitted changes (must be tested, committed, then pushed)

Tracked modified files:

```text
.env.example
.env.local-cloud.example
README.md
frontend/src/App.jsx
src/gnaniassignment/stt/gnani.py
src/gnaniassignment/telephony/voicelink.py
src/gnaniassignment/voice/events.py
tests/test_enquiry_pipeline.py
```

They implement:

1. `VoiceTranscript.is_final` (default `True` preserves old behavior).
2. Parsing explicit Gnani `is_final` / `final` flags when present.
3. `VOICE_LINK_STT_MIN_SILENCE_MS=350` for the VoiceLink adapter, regardless
   of an older `GNANI_MIN_SILENCE_MS=700` production setting.
4. Retain the latest STT hypothesis. When Smart Turn says `complete`, commit
   a non-final hypothesis exactly once; provider-final still works unchanged.
5. Live dashboard/SSE trace events, safe for debugging:
   - `media.inbound_started`
   - `transcript.partial`, `transcript.final`, `transcript.committed`
   - `smart_turn.complete` / `smart_turn.incomplete`
   - `llm.started`, `llm.first_token`, `llm.completed`
   - `tts.started`, `tts.first_audio`, `tts.outbound_first_frame`,
     `tts.completed`
6. Dashboard titles/details for those trace events.

## Validation already done

Focused tests passed:

```bash
.venv/bin/python -m unittest \
  tests.test_enquiry_pipeline.PipelineTests.test_smart_turn_can_commit_a_nonfinal_stt_hypothesis_once \
  tests.test_enquiry_pipeline.PipelineTests.test_new_call_uses_compact_prior_summaries_and_keeps_active_turns \
  tests.test_enquiry_pipeline.PipelineTests.test_tts_reopens_an_idle_closed_provider_socket_before_a_reply \
  tests.test_enquiry_pipeline.PipelineTests.test_voicelink_reply_metrics_separate_tts_from_websocket_handoff \
  tests.test_enquiry_pipeline.PipelineTests.test_generic_prompt_prevents_repeated_greetings_on_continuing_calls \
  tests.test_dashboard_runtime.DashboardRuntimeTests.test_runtime_events_fan_out_to_each_subscriber -v
```

`unittest discover` locally stalled in the existing dashboard harness after its
first tests. A prior GitHub run completed 58 tests and failed only the exact
prompt wording assertion, which `eaa12c6` corrected.

## Required release process

The user requested the normal rule: **locally validate STT → LLM → TTS before
pushing to production**. They later explicitly asked to deploy immediately;
confirm intent before bypassing the local live test.

Correct local test for the new endpoint logic is the local VoiceLink adapter,
not `gnani_microphone` (that older harness bypasses TEN/Pipecat Smart Turn):

Terminal 1:

```bash
CONVERSATION_LOGGING_ENABLED=false EPISODIC_MEMORY_PROVIDER=none \
ORDER_DATA_ENABLED=false ORDER_CONTEXT_PROVIDER=none PROMPT_TREE_ENABLED=false \
SHADOW_INTENT_ENABLED=false VOICE_LINK_TURN_DETECTION_MODE=endpoint \
VOICE_LINK_TURN_DETECTOR=ten_pipecat VOICE_LINK_STT_MIN_SILENCE_MS=350 \
VOICELINK_WS_TOKEN=local-test-token UV_CACHE_DIR=/tmp/gnani-uv-cache \
uv run uvicorn gnaniassignment.api.server:app --host 127.0.0.1 --port 8081
```

Terminal 2 (live trace):

```bash
curl -N http://127.0.0.1:8081/v1/dashboard/events
```

Terminal 3 (audio simulator):

```bash
VOICELINK_WS_TOKEN=local-test-token UV_CACHE_DIR=/tmp/gnani-uv-cache \
uv run python -m gnaniassignment.pipeline.voicelink_ws_simulator \
  --url ws://127.0.0.1:8081/v1/media/voicelink \
  --input-device default --playback-device default \
  --customer-id local-turn-test
```

The current Codex sandbox cannot open local sockets, so the user must run the
three terminals themselves. Test phrase: `Kal meri party hai.` Then remain
silent for two seconds. The trace must show a transcript, Smart Turn decision,
turn commit, LLM, and TTS stages.

## Production deployment after validation

1. Commit only the tracked files listed above.
2. Use the existing clean production worktree:
   `/tmp/voice-agent-production-clean`.
3. Push its HEAD to `origin/master`.
4. Verify GitHub Actions. `gh` is installed but not authenticated; it needs
   `gh auth login` or `GH_TOKEN` before `gh run list` can inspect private runs.
5. If CI does not roll the VM, build an immutable Artifact Registry image and
   run `/opt/voice-agent/app/deploy-gcp-vm.sh <image-uri>` through IAP.
6. Verify the VM image is the new SHA and both:
   - `http://127.0.0.1:8080/healthz` on the VM
   - `https://voice.sidecut.me/healthz`

## Live trace after deployment

The event endpoint requires the agent API key and should never expose it in a
browser bundle:

```bash
curl -N -H "X-API-Key: $AGENT_API_KEY" \
  https://voice.sidecut.me/v1/dashboard/events
```

The React dashboard source is in `frontend/`; it is not currently hosted as a
public dashboard. Do not bake the API key into Vite. Use server-side auth or a
protected proxy before publishing it.
