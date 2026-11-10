# Relay / Gnaniassign — exhaustive project handoff

> This is the detailed, secret-free transfer document for a new coding agent.
> It captures the project history, architecture, decisions, operating model,
> deployment posture, experiments, failures and known work. It intentionally
> omits passwords, bearer tokens, API keys, OAuth secrets, private database
> URLs, raw caller transcripts, phone numbers and cloud credentials.
>
> Read `CLAUDE.md` first for guardrails and the short operating brief. Read
> `docs/relay-launch-story.md` for launch language and product narrative.

## 1. Product intent

Relay is a configurable real-time AI phone-agent platform. A person or
business should be able to bring their own:

- phone number / telephony provider;
- system prompt and small scenario instructions;
- knowledge base and account-level context;
- MCP-style tools / APIs;
- data-access policy and trusted callers;
- voice/STT/LLM/TTS providers.

The agent answers incoming calls, identifies itself as an AI assistant,
understands the request in Hinglish, uses authorised context and tools, and
creates a useful hand-off for the owner. The long-term direction is D2C: a
user configures an assistant in a mobile-first dashboard and can later use it
from a Flutter app WebView.

The project currently demonstrates two primary configurations:

| Configuration | Main goal | Context/tools |
|---|---|---|
| Grocery-store concierge | Help neighbourhood customers with orders and service requests | MiniLM retail routing, customer/order context, grocery knowledge/policies |
| Corporate employee assistant | Represent an employee who cannot answer, capture messages and answer limited work/availability questions | Google Calendar presence, Plane status lookup, approval queue for writes |

## 2. Non-negotiable principles

1. **Disclose AI identity.** The assistant must never impersonate the owner or
   a human.
2. **Least privilege.** Only caller-appropriate, owner-approved context/tools
   are available.
3. **No fabricated actions.** A request is not “sent,” “updated,” “forwarded”
   or “completed” until a tool returns success.
4. **Speech quality over cosmetic metrics.** A cue that hides delay is not a
   latency improvement.
5. **Measure the actual boundary.** Distinguish model speed, queueing, TTS,
   server-to-carrier hand-off and unobservable handset playback.
6. **No raw secrets in git/chat/docs.** Historical credentials accidentally
   appeared in conversation; treat them as compromised and rotate them through
   their providers. Do not reproduce them.
7. **Local audible testing before voice-production changes.** The owner wants
   to run the local STT → LLM → TTS test personally before a push/deploy,
   unless they explicitly order deployment.

## 3. Chronological build history

### Phase A — MiniLM and retail intent classification

The first system was a retail intent assistant. A multilingual MiniLM was
fine-tuned in Google Colab because the local environment had OOM pressure.

#### Training sources

The IndicJevBench Hugging Face dataset configuration did not expose a single
`train` split; it exposed named source splits. The training job was corrected
to combine these four:

- `intent_massive` — 57,922 rows
- `fintech_banking77` — 7,816 rows
- `hinglish_lid` — 3,220 rows
- `synthetic_enterprise` — 444 rows

Total: **69,402 examples**.

The fine-tune used `microsoft/Multilingual-MiniLM-L12-H384`. It treated the
state embedding as a query and option/instruction embeddings as candidate
answers, optimizing cross-entropy over scaled cosine similarities. This is not
generative fine-tuning; it is contrastive-style multiple-choice routing.

The original Colab run completed three epochs. Logged mean loss decreased:

| Epoch | Mean loss |
|---|---:|
| 1 | 0.9146 |
| 2 | 0.4234 |
| 3 | 0.2817 |

The artifact was saved to a Google Drive path and later downloaded as a ZIP
archive. It should remain out of git. The eventual local archive is an
untracked `systemone-retail...zip` file; do not add it to commits. The model
was later prepared for Hugging Face publishing, but treat its current hosted
state as something to verify rather than assume.

#### Retail synthetic dataset

A separate synthetic 60,000-row retail dataset was designed to cover:

```text
track_order
return_item
cancel_order
refund_status
wrong_item
size_issue
payment_issue
delivery_address
other
```

The purpose was to train a more business-specific classifier. A retail run
reported very low/zero loss and 100% accuracy on synthetic validation; this is
not reliable evidence of production success because train and generated
validation language can be too similar.

#### Current MiniLM policy

- MiniLM is a switchable/shadow router, not the only decision-maker.
- The generic personal-agent route can operate without it.
- Production conversations should be collected, reviewed and explicitly
  labelled before being used for further training.
- The model is a good fit for the grocery persona’s fixed taxonomy, not for
  the corporate assistant’s open-ended conversational requests.

### Phase B — Generic voice personal agent

The product direction shifted away from a narrow retail-only agent to a
generic phone assistant. The key prompt evolved to an assistant that:

- asks what callers need from the owner;
- takes concise messages and confirms intent;
- can arrange owner-approved outbound communication later;
- never presents itself as the owner;
- communicates naturally in Roman-script Hinglish;
- avoids repeated greetings in continuing calls;
- handles short confirmations as context-dependent follow-ups.

The first generic prompt had looping/repetition issues: it repeatedly said
variations of “I will check and tell you,” repeatedly greeted, asked for an
order ID despite available context, and sometimes over-explained. Prompt rules
were strengthened, but the lasting lesson is that dialogue state, turn-taking
and tool truthfulness matter more than prompt wording alone.

### Phase C — Memory and data model

Memory was introduced to prevent every call from being stateless.

#### Memory layers

| Layer | Purpose | Expected scope |
|---|---|---|
| Working memory | Current active request, confirmations and immediate facts | One call/session |
| Episodic memory | Compact end-of-call summary | Per customer/caller across calls |
| Structured facts | Durable named factual fields | Per customer/owner, policy controlled |
| Knowledge base | Stable business information | Per tenant/persona |
| Customer context window | Small preloaded order/customer context | Per caller/current call |

PostgreSQL was selected for durable episodes, conversations, review records,
knowledge/order records and operations. Redis was introduced for fast prompt
plan caching so relational prompt retrieval is not a per-turn latency cost.

An important product decision: do not automatically dump all historic
conversation text into every new call. A previous context retrieval mistake
caused the assistant to bring old irrelevant messages into a new conversation.
The intended policy is retrieval only when the caller explicitly asks to
remember/revisit a prior discussion, and then only the relevant structured
facts/summary should be injected.

### Phase D — Conversation analytics and evaluation data

The system began storing complete call transcripts as caller-STT ↔ agent-reply
pairs, with persistent session/call identifiers. There are two separate
post-call analysis streams:

1. **Intent judgement:** a Bedrock judge receives STT text only. It labels
   available classifier intent categories without being biased by agent reply
   text.
2. **Conversation-flow judgement:** a judge receives STT plus LLM output to
   discover loops, ignored corrections, contradictory replies, needless
   clarifications, repeated greetings, safety failures and opportunities for
   do/don’t prompt rules.

The intent classifier can run in shadow mode while a generic agent serves the
caller. The aim is an active-learning workflow: save prediction + transcript,
review the label in a queue, then use reviewed examples for later training.

### Phase E — Modular provider architecture

The codebase was refactored around adapters/composition so STT, LLM, TTS,
memory, prompts and tools are replaceable. The dependency composition point is
`src/gnaniassignment/app/container.py`.

Supported/experimented provider choices include hosted LLMs, Bedrock,
Cerebras/Groq/Simplismart/Vercel, local Laya variants, local model
experiments, Gnani STT/TTS, and multiple VAD choices. Not all are production
enabled at once. Production should use the configured provider, not arbitrary
fallbacks, and all provider changes should be measured.

### Phase F — Prompt tree and dashboard

A prompt tree was designed so prompts are small instructions that compose into
one active plan. The plan resolver can use Redis caching. The dashboard was
designed as a dark, mobile-first React interface that can be wrapped in a
Flutter WebView:

- active call state;
- live events;
- caller/agent transcript;
- MiniLM review queue;
- call logs and latency;
- configurable prompt/tool surfaces;
- approval queue.

Flutter was installed locally and a wrapper application was created. The
current mobile approach is deliberately WebView-first to avoid rebuilding all
dashboard feature logic in native Flutter/React Native.

### Phase G — Telephony, real calls and VoiceLink

VoiceLink was chosen for Indian telephony. Its documented WebSocket media is:

- A-law encoded;
- 8-bit mono;
- 8 kHz;
- events including `connected`, `start`, `media`, `mark`, `clear`, `transfer`,
  and `stop`.

The implementation:

1. validates the VoiceLink `start` media format;
2. decodes inbound base64 A-law into PCM;
3. resamples to the STT rate;
4. sends exact STT frames to a persistent Gnani streaming connection;
5. resamples TTS output back to 8 kHz;
6. A-law encodes it;
7. sends fixed 160-byte/20 ms paced VoiceLink frames;
8. sends a `mark` at a response boundary;
9. sends `clear` for barge-in.

The production WSS endpoint requires a token query parameter. It was tested
with a local VoiceLink WebSocket simulator so a local microphone could exercise
the hosted cloud pipeline without an actual PSTN call.

### Phase H — Turn-taking and VAD experiments

#### Original issues

Real calls exposed the critical problems:

- agent stayed silent after a caller spoke;
- caller had to nudge it (“bata rahe ho?”) to get a reply;
- replies were interrupted, looped, or continued after a new topic;
- noise/background audio triggered false starts;
- telephony voice was choppy or distorted;
- the agent spoke too long;
- exact spoken work IDs were badly transcribed.

#### Endpointing experiments

1. **Provider VAD only:** Gnani final-silence threshold was reduced from 700
   ms toward 350 ms so provider final text arrives earlier.
2. **Silero VAD:** introduced as optional/off/shadow/endpoint mode, but the
   large Torch dependency was later removed from production to avoid CI/disk
   pressure.
3. **TEN VAD + Pipecat Smart Turn:** current preferred design.
   - TEN processes 16 kHz audio in small hops for acoustic boundaries.
   - A pause triggers Smart Turn semantic analysis over up to ~8 seconds.
   - `complete` commits a caller turn.
   - `incomplete` keeps listening, preventing a reply to a mid-sentence pause.
   - production uses `ten_pipecat`; Silero remains only for optional local
     experimentation where dependencies are installed.

One diagnostic initially crashed with a zero sample rate in Smart Turn. The
setup was corrected. Local logs then showed repeated short speech start/end
events and semantic completion/incompletion decisions; that behaviour must be
interpreted with real microphone/noise conditions, not assumed perfect.

#### Barge-in

When acoustic speech starts during agent output:

- `reply_interrupted` is set;
- queued outbound audio is cleared;
- VoiceLink receives `clear`;
- more outbound frames are stopped.

The measured metric is only server detection → carrier `clear` hand-off. It
does not claim handset audio ceased at that exact time, because the carrier
does not acknowledge handset playback.

### Phase I — TTS cues and audio quality

Gnani Timbre TTS supports constrained audio tags. Relay intentionally controls
the tags in code; the LLM’s markup is stripped.

#### Giggle

- A safe `latency_giggle` path exists in the **local microphone harness**.
- It waits 450 ms for the first LLM text token.
- If no token arrives, it plays standalone `<giggle>` audio.
- Answer generation continues in parallel; answer playback is gated until the
  cue is done to avoid overlapping audio.
- VoiceLink production giggle is currently disabled. Previous live behaviour
  was “giggle, then long pause,” which felt worse than silence.
- Positive, low-stakes responses may be decorated with a controlled giggle
  tag if the TTS tag rules/model permit it; never use it in complaints,
  refunds, sensitive matters or conflict.

#### Clear throat

- VoiceLink production currently has the clear-throat timer enabled.
- The timer starts after a committed caller turn and LLM request.
- If no first model text appears at **600 ms**, it creates standalone
  `<clears throat>` TTS audio.
- When first text arrives, the cue is allowed to finish before speech output
  begins, preventing overlap.
- The cue is rule-based—not decided by an LLM.
- It must not be presented as a TTFT improvement. High cue rate means a real
  latency problem remains.

#### Audio failure investigation

Direct production TTS/VoiceLink probing found:

- low-level PCM → A-law → paced WebSocket hand-off works;
- Gnani can emit an initial small PCM chunk promptly, then leave a material
  gap before more audio arrives;
- 20 ms pacing cannot conceal a gap when there is too little buffered audio;
- this explains choppy/cut-off output even when first token timing is okay.

Do not add a fixed large jitter buffer to every reply without measuring: it
would improve continuity but hurt first-audio latency. Preferred next design:
an **adaptive jitter buffer**, only waiting for a second chunk if the first
chunk is too short to safely sustain playback.

### Phase J — Infrastructure and deployment

The system originally considered AWS/ECS/Jenkins, but the actual production
path became GCP with GitHub Actions. Jenkins is not needed for the current
workflow.

#### Current production topology

```text
GitHub origin/master
  → GitHub Actions
  → regression tests
  → Docker build
  → Artifact Registry immutable image
  → IAP SSH into GCP VM
  → Docker Compose replace API
  → local health / rollback guard
  → public HTTPS health check

GCP VM
  ├─ API container (FastAPI + voice pipeline)
  ├─ PostgreSQL container + persistent volume
  └─ Redis container + persistent volume
```

Non-secret identifiers:

- GCP project: `gen-lang-client-0813070445`
- region: `asia-south1`
- VM: `voice-agent-01`
- zone: `asia-south1-a`
- Artifact Registry repository: `voice-agent`
- public host: `voice.sidecut.me`
- health: `https://voice.sidecut.me/healthz`
- WSS: `wss://voice.sidecut.me/v1/media/voicelink`

#### Important deployment incidents resolved

| Incident | Cause | Resolution |
|---|---|---|
| GitHub runner “No space left on device” | huge Python/Torch/Silero dependency installation | CI changed to lightweight test dependencies; production removed legacy Torch/Silero stack |
| `iam.serviceAccounts.actAs` denial | deploy identity lacked needed role | IAM permission corrected for GitHub Actions deployment path |
| `AGENT_IMAGE` missing | compose interpolation lacked image env | deployment workflow supplied/retained immutable image variable |
| API port 8080 allocated | old local process/container bound host port | deployment/container setup corrected; rollback behaviour verified |
| VM disk nearly full | old Docker images accumulated | unused old images/build cache pruned while preserving active resources/volumes |
| initial WSS accepted but call did not answer/speak | adapter/integration issues evolved across several fixes | authentication, media adapter, TTS terminal events and diagnostics added |
| calls connected but TTS had 0 frames | TTS stream terminal/zero-audio handling | retry/diagnostic path added; still monitor live audio quality |

#### Deployment policy

- Push `origin/master` to deploy production.
- Do not deploy from the `test` remote.
- Use GitHub Actions run status plus `/healthz` after every deploy.
- VM secrets live only on the VM’s ignored runtime environment; GitHub Actions
  does not receive application/provider secrets.
- Never casually run broad Docker pruning, destructive volume commands,
  `git reset --hard`, or force push.

### Phase K — Tool integrations

#### Google Calendar

Google Calendar OAuth was created with a web redirect at the production host.
The owner has completed a connection flow. Operational design:

1. owner authorises Google Calendar;
2. encrypted/token-backed integration stores authorisation outside git;
3. a background routine refreshes a small availability/presence snapshot;
4. call context reads the existing snapshot rather than waiting for a live
   Google call;
5. the LLM receives only safe presence wording;
6. caller hears a privacy-preserving answer such as “they are in a meeting”;
7. no event title, participants, links, location or full schedule is exposed.

This feature should be verified after provider token refreshes or deploys;
do not infer that OAuth secrets/environment values are present just because a
dashboard says “connected.”

#### Plane

Plane was connected with a workspace and an API integration during setup.
Current intended flow:

1. check feature enablement;
2. normalise inbound caller number and compare with trusted caller allowlist;
3. require an explicit request mentioning task/work item/status and a concrete
   identifier;
4. request only that one item from Plane;
5. render verified status context into the current turn;
6. speak a concise answer;
7. for comments/mutations, create an internal pending action;
8. owner approves via dashboard/API;
9. only then send mutation to Plane.

Observed limitation: real STT distorted IDs (`SC-42` into words such as “test
eight”/“test aid”), so tool retrieval often did not trigger and the generic
agent repeatedly clarified. Improving identifier capture must come before
claiming reliable voice work-item retrieval.

### Phase L — Dashboard, mobile and design work

- A React dashboard is under `frontend/`.
- It is mobile-first / dark themed, intended for 9:16 viewport and Flutter
  WebView use.
- The Flutter wrapper is in `mobile/relay_mobile/`.
- The dashboard should show live events, conversation pairs, latency, prompt
  controls, MiniLM review queue and tool/action state.
- The dashboard must consume real protected backend data, not static mock data.
- A Figma/Messenger-inspired dark reference was used as a styling influence.
- The dashboard needs a valid dashboard API key in its configuration; never
  hard-code a key in frontend source or mobile artifact.

### Phase M — Launch-material work

Created `docs/relay-launch-story.md`, a product/engineering launch story with:

- two personas;
- safety policy;
- pipeline; memory; prompt tree; analytics; deployment;
- Calendar/Plane details;
- giggle/clear-throat details;
- current latency baseline;
- a suggested launch-video narrative;
- safe launch claims and claims to avoid.

The launch story is documentation, not a statement that all features are
finished or production-perfect. Preserve metric qualifiers.

## 4. Current source map

| Area | Files/directories |
|---|---|
| FastAPI | `src/gnaniassignment/api/server.py` |
| VoiceLink WSS/media | `src/gnaniassignment/telephony/voicelink.py` |
| Evaluation | `src/gnaniassignment/telephony/evaluation.py`, `latency.py` |
| Audio enhancement | `src/gnaniassignment/voice/audio_enhancement.py` |
| TEN/Smart Turn | `src/gnaniassignment/voice/ten_smart_turn.py` |
| Silero optional | `src/gnaniassignment/voice/silero_vad.py` |
| Gnani STT | `src/gnaniassignment/stt/gnani.py` |
| Gnani TTS tags/streaming | `src/gnaniassignment/tts/gnani.py` |
| Local mic runner | `src/gnaniassignment/pipeline/gnani_microphone.py` |
| Hosted WSS simulator | `src/gnaniassignment/pipeline/voicelink_ws_simulator.py` |
| TEN diagnostic | `src/gnaniassignment/pipeline/ten_smart_turn_microphone.py` |
| Dependency factory | `src/gnaniassignment/app/container.py` |
| Agent graph/session | `src/gnaniassignment/orchestration/` |
| Memory | `src/gnaniassignment/orchestration/episodic_memory.py` |
| Prompt tree/cache | `src/gnaniassignment/prompts/` |
| Calendar | `src/gnaniassignment/google_calendar.py` |
| Plane/work actions | `src/gnaniassignment/work_items.py` |
| Conversation analytics | `src/gnaniassignment/conversation_analytics.py` |
| MiniLM/retail router | `src/gnaniassignment/llm/retail_minilm.py` and related agents |
| Dashboard | `frontend/` |
| Mobile wrapper | `mobile/relay_mobile/` |
| GCP deployment | `.github/workflows/` and `infra/gcp/README.md` |
| Evaluation examples | `evals/voice/` |

## 5. Current live voice evaluation state

### Why the original numbers were wrong

The legacy dashboard called committed-turn-to-first-token time “LLM TTFT.” It
included serial turn queue waiting. It also could calculate first-audio latency
from a stale prior VAD speech-end. This produced implausibly high reported
p95s, including a legacy value above eight seconds.

Commit `9cd66f1` introduced measurement schema version 2:

| Field | Definition |
|---|---|
| `turn_queue_delay_ms` | transcript committed → turn processing starts |
| `llm_model_ttft_ms` | turn processing starts → first LLM token |
| `e2e_llm_ttft_ms` | transcript committed → first LLM token |
| `first_outbound_audio_latency_ms` | transcript committed → first server-to-VoiceLink media hand-off |

Schema-v1 data is intentionally excluded from current percentile summaries;
never mix the two. The first corrected production sample had six turns:

| Metric | p50 | p95 |
|---|---:|---:|
| Model TTFT | 579 ms | 824 ms |
| E2E TTFT | 1,065 ms | 1,748 ms |
| Turn queue delay | 470 ms | 993 ms |
| First outbound audio | 924 ms | not publication-ready from this sample |

The initial “silent turn rate” reported 3/6, but two of those were caller
interruptions before first audio. It currently conflates interruption with a
true failed/silent reply. Split this before public reporting.

## 6. Immediate technical backlog

### Highest priority

1. **Separate interruption from silence in evaluation.** Add outcome-aware
   metrics: `interrupted_before_first_audio`, `true_silent_turn`, and perhaps
   `barge_in_after_first_audio`. Preserve raw timing.
2. **Fix turn queueing after barge-in.** A queued caller turn can wait while a
   prior synchronous/threaded `ask_stream` generation drains. The code must
   safely abort/supersede stale generation without racing the session or
   corrupting conversation history.
3. **Fix TTS continuity.** Prototype adaptive buffering only when the first
   chunk cannot cover expected provider gaps. Measure first-audio and frame
   gaps before/after.
4. **Build real evaluation data.** At least 30 clean, labelled turns before
   presenting p95/p99; separately test noisy Hinglish, pauses, confirmation,
   barge-in, work IDs and long conversations.

### Product quality

5. Improve exact identifier capture/confirmation for Plane.
6. Ensure generic assistant asks a single useful question and does not loop.
7. Ensure spoken replies contain no JSON, Markdown, code fences, stray tags
   or non-Roman text unless caller language changes.
8. Validate dashboard real-data rendering, call-detail API and tool/action
   approval flow.
9. Confirm Calendar token refresh and snapshot lifecycle in production.
10. Finish configurable user-level instructions and tool/MCP configuration UI.

### Research / growth

11. Build reviewed real-call data loop for MiniLM; do not overfit to synthetic
    retail data.
12. Publish model artefact only after verifying its model card, licence, files
    and no customer data.
13. Produce launch video from the launch story after UI/real call footage is
    ready; do not show real personal data.

## 7. Local commands

### Basic test suite

```bash
cd /home/tushar/CODES/Gnaniimplerment
uv sync
uv run python -m unittest tests.test_voice_evaluation tests.test_enquiry_pipeline
```

### Local microphone full voice chain

```bash
uv run python -m gnaniassignment.pipeline.gnani_microphone \
  --device default --with-llm --with-tts
```

Use `arecord -L` and `aplay -L` to find the actual device. The prior
`plughw:1,0` error meant the system had no card index 1, not a Python bug.

### Hosted VoiceLink pipeline with local microphone

```bash
export VOICELINK_WS_TOKEN='obtain privately; never paste into files'
uv run python -m gnaniassignment.pipeline.voicelink_ws_simulator \
  --input-device default \
  --playback-device default \
  --customer-id local-voicelink-test
```

This opens the hosted production WSS path from the local computer; it does not
place a public phone call.

### TEN/Smart Turn diagnostic

```bash
uv run python -m gnaniassignment.pipeline.ten_smart_turn_microphone \
  --device default
```

### Evaluation report from examples

```bash
uv run python -m gnaniassignment.pipeline.voice_eval_report \
  --turns evals/voice/turns.jsonl.example \
  --stt-labels evals/voice/stt-labelled.jsonl.example \
  --output /tmp/voice-eval-report.json
```

## 8. Production verification commands

Use these only when authorised and with credentials sourced privately from the
VM/local secret store. Do not print secrets.

```bash
# Public status only.
curl -fsS https://voice.sidecut.me/healthz

# Deployment history.
gh run list --limit 10

# VM shell via IAP.
gcloud compute ssh voice-agent-01 --tunnel-through-iap \
  --project=gen-lang-client-0813070445 --zone=asia-south1-a

# Current containers after connecting to VM.
sudo docker ps
```

The private dashboard endpoints require `X-API-Key`. Retrieve the key only
from the protected runtime environment as needed, do not echo it, and never
paste it into a document, terminal history, source file or agent response.

## 9. Git workflow and deployment state at handoff

- `origin/master` deployed `9cd66f1` successfully and public `/healthz`
  returned `{"status":"ok"}`.
- Current local branch may be ahead with documentation commits:
  - `aa6354e` expanded launch personas/cue story;
  - `c4a94e8` added the initial `CLAUDE.md` handoff (this document extends it).
- Do not assume documentation commits are production code changes.
- Before pushing any code: inspect `git status`, preserve user changes and
  untracked files, run relevant tests, and follow local audible test policy.

## 10. Things explicitly not to do

- Do not move raw model ZIPs into git.
- Do not add `.env`, OAuth JSON, database connection strings, API keys,
  VoiceLink WSS token or cloud secrets to git.
- Do not use a generic data export tool to copy raw chat history containing
  credentials into another agent/service.
- Do not reset/checkout hard or delete Docker/Postgres/Redis volumes.
- Do not use historical legacy p95 as a model TTFT benchmark.
- Do not enable a tool to mutate Calendar/Plane based solely on a caller’s
  speech; require the designed approval boundary.
- Do not expose calendar details or work information to an untrusted caller.
- Do not claim actual handset playback timing based on server frame-send time.

## 11. Suggested first actions for a new agent

1. Read `CLAUDE.md`, then this file, then `docs/relay-launch-story.md`.
2. Run `git status --short` and preserve user-owned untracked files.
3. Run the focused unit tests.
4. If working on real voice behaviour, have the owner test local audio before
   production deployment.
5. Inspect current v2 call metrics after a fresh call; never use old data for
   a new claim.
6. Address evaluation interruption-vs-silence first, then safe generation
   cancellation/queueing, then adaptive TTS buffering.

