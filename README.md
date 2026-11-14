# Relay

### Your AI phone presence — for customers, colleagues, and the calls you cannot pick up.

Relay is a real-time personal AI phone agent. It answers authorised calls,
speaks in Hinglish, understands the caller’s need, uses only owner-approved
context/tools, and leaves an actionable call hand-off.

It is a configurable platform, not a hard-coded bot. The same voice core can
run as a grocery-store concierge or a corporate employee assistant.

> Relay always identifies itself as an AI assistant. It does not impersonate
> the owner, leak private data, or claim a tool action succeeded without proof.

## Product configurations

| Grocery-store concierge | Corporate employee assistant |
|---|---|
| Handles delivery status, returns, refunds, cancellations, wrong items, payment/address questions and owner hand-offs. | Captures messages, shares privacy-safe availability, retrieves one authorised Plane work-item status and queues writes for approval. |
| Uses a fast MiniLM retail intent route plus customer-scoped order context. | Uses Google Calendar presence and Plane under trusted-caller and explicit-request controls. |
| Never invents order status or promises a refund without a connected system confirming it. | Never reveals private calendar data, broad project data, or writes to an external system from caller speech alone. |

### Persona contract

The persona is not only a system-prompt label. It selects a complete runtime
contract: prompt nodes, knowledge scopes, available tools, caller permissions,
intent-routing policy and post-call evaluation criteria.

| Concern | Grocery-store concierge | Corporate employee assistant |
|---|---|---|
| Primary caller | Neighbourhood customer | Colleague, client or collaborator |
| Live objective | Resolve or capture an order/service request | Explain safe availability, capture a message or retrieve an authorised work-item update |
| Authoritative data | Caller-scoped customer and order records | Calendar presence snapshot, trusted-caller policy and a specific Plane work item |
| Intent layer | Nine-label multilingual MiniLM in switchable/shadow mode | Generic conversational route with explicit Calendar/Plane tool gates |
| Mutating action | Capture a requested cancellation/return/refund for owner handling | Create a pending comment/action for owner approval |
| Hard boundary | No invented status, cross-customer data or unconfirmed refund promise | No private calendar detail, workspace browsing or caller-authorised external write |
| Hand-off | Concise store-owner task with verified identifiers | Concise owner message with caller need, timing and pending approval state |

Both configurations use the same invariants:

1. disclose that the speaker is an AI assistant;
2. use only data belonging to the resolved caller/account;
3. distinguish a proposed action from a completed action;
4. ask only for the single missing detail;
5. preserve corrections made later in the call;
6. store a compact hand-off without replaying an entire historic transcript.

## Why Relay is a real voice agent

- Streaming STT → LLM → TTS, rather than generating a complete voice note.
- Semantic turn-taking: it distinguishes a natural pause from a completed
  thought.
- Barge-in: when the caller starts speaking, Relay clears queued bot audio and
  listens.
- Memory is layered and scoped; historic conversations do not flood each new
  prompt.
- Tools are constrained by policy, caller trust and approval boundaries.
- Every turn is measured so queue delay, model latency, TTS continuity and
  carrier hand-off are not confused with one another.

---

# Technical architecture

## End-to-end call path

```text
                         CALLER
                           │ PSTN
                           ▼
              VoiceLink telephony / authenticated WSS
                           │ 8 kHz A-law media
                           ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ Relay VoiceLink adapter                                        │
 │ validate → A-law decode → PCM → resample                      │
 └───────────────┬───────────────────────────────────┬───────────┘
                 │ inbound                           │ outbound
                 ▼                                   ▲
 ┌──────────────────────────────┐       ┌──────────────────────────────┐
 │ Optional audio enhancement   │       │ Gnani streaming TTS          │
 │ TEN VAD acoustic boundaries  │       │ safe controlled audio tags   │
 │ Smart Turn semantic endpoint │       │ PCM → 8 kHz A-law            │
 └──────────────┬───────────────┘       │ paced 160-byte / 20 ms frames│
                ▼                       └──────────────▲───────────────┘
        Gnani streaming STT                             │
                ▼                                       │
 ┌───────────────────────────────────────────────────────────────┐
 │ Relay TurnSession                                              │
 │ prompt plan · safety · working memory · approved context · LLM │
 └───────┬──────────────────────┬──────────────────────┬─────────┘
         ▼                      ▼                      ▼
  PostgreSQL              Redis cache          approved tools
 calls/memory/evals       prompt plans          Calendar / Plane / orders
```

## Runtime component map

| Component | Responsibility | Latency boundary |
|---|---|---|
| `api/server.py` | Health, protected dashboard APIs, prompt/tool administration and VoiceLink WebSocket upgrade | HTTP/WSS ingress only |
| `telephony/voicelink.py` | VoiceLink protocol, media negotiation, inbound/outbound pacing, barge-in and per-turn workers | Hot real-time path |
| `stt/gnani.py` | Prisma WebSocket configuration, PCM frame streaming and partial/final transcript events | Hot real-time path |
| `voice/ten_smart_turn.py` | TEN acoustic speech boundaries plus Pipecat semantic completion | Hot real-time path |
| `orchestration/session.py` | Call identity, current-turn state, memory boundaries, delivery tracking and post-call finalisation | Per committed turn |
| `orchestration/graph.py` | Intent decision, prompt input construction, retrieval and agent invocation | Per committed turn |
| `prompts/tree.py` | Versioned conditional prompt nodes and Redis-cached prompt plans | Primarily call setup |
| `llm/agents/*` | Persona prompt, streaming model invocation and structured response parsing | Hot generation path |
| `tts/gnani.py` | Timbre streaming synthesis, audio tags and PCM output | Hot real-time path |
| `conversation_analytics.py` | Durable transcript ledger, post-call judge, conversation patterns and active reviewed guidance | Off first-audio path |
| `telephony/production_evals.py` | Real-telephony eval runs, production sampling, human review and candidate-rule governance | Post-call/admin path |
| PostgreSQL | Calls, turns, episodic memory, orders, knowledge, prompts, evals and approvals | Durable source of truth |
| Redis | Resolved prompt-plan cache | Avoids repeated prompt-tree reads |

## One complete call, chronologically

```text
1. VoiceLink opens authenticated WSS and sends `connected` / `start`.
2. Relay resolves stream ID, call ID, caller identity and negotiated media format.
3. Call-scoped context is prepared: persona, prompt plan, active reviewed rules,
   caller/order snapshot, Calendar presence and prior compact summaries.
4. Base64 media frames are decoded to PCM and resampled only where required.
5. The same cleaned PCM feeds streaming STT and acoustic/semantic turn detection.
6. Partial STT is observable; only a committed turn enters the conversation graph.
7. The graph builds the smallest safe prompt from current-call state and approved context.
8. The LLM streams the spoken `answer` field first; memory operations follow later.
9. Complete speakable fragments enter streaming TTS without waiting for the full answer.
10. PCM is converted to the negotiated telephony codec and paced in 20 ms frames.
11. Delivery is recorded as audio-sent, interrupted or failed—not merely generated.
12. The next prompt sees only assistant speech actually handed to the caller.
13. On `stop`, Relay persists the transcript, compact hand-off, metrics and judge evidence.
14. Selected calls enter the production human-review queue; no post-call work delays audio.
```

The architecture deliberately separates four clocks that are often collapsed
into one misleading “latency” number: end-of-speech detection, prompt/request
setup, model first token and first real outbound answer audio.

## The layers

### 1. Telephony transport

VoiceLink sends JSON protocol events (`connected`, `start`, `media`, `mark`,
`clear`, `transfer`, `stop`) over an authenticated bidirectional WebSocket.
Relay reads the `start.media_format` declaration instead of assuming every call
has the same codec. The current adapter accepts:

```text
audio/alaw @ 8 kHz
audio/ulaw @ 8 kHz
audio/l16  @ 8 kHz
audio/l16  @ 16 kHz
```

For the common A-law path, base64 media is decoded to 16-bit linear PCM. If the
telephony and STT sample rates differ, `audioop.ratecv` maintains streaming
resampler state across frames; individual frames are not resampled as isolated
files. Native L16/16 kHz removes that inbound conversion when VoiceLink is
configured to supply it.

Outbound synthesis follows the inverse path. Timbre PCM is resampled to the
negotiated telephony rate, encoded as A-law/u-law or retained as L16, buffered,
and paced as fixed 20 ms frames. Frame pacing prevents a provider from receiving
an entire answer as a network burst and then playing it with gaps or distortion.
The final short frame is padded with the codec's correct silence byte—not an
arbitrary zero byte.

The media endpoint requires a VoiceLink WebSocket token. Dashboard and
management APIs separately require an API key.

The adapter keeps protocol events, audio payloads and management APIs separated:

- media WSS authenticates with `VOICELINK_WS_TOKEN`;
- dashboard and eval APIs require `AGENT_API_KEY`;
- VoiceLink REST operations use a server-side bearer token;
- secrets and raw audio are never emitted in routine application logs.

### 2. Streaming STT

Gnani Prisma v3.0 receives signed 16-bit little-endian mono PCM over a persistent
WebSocket. Relay requests 16 kHz STT and verifies the rate acknowledged by the
provider; a silent provider fallback to a different rate is treated as a
configuration error rather than accepted invisibly.

The configured provider-final silence window is 400 ms. It is intentionally a
transcription segmentation parameter, not the final conversational endpoint:

```text
PCM frames
  → Prisma partial hypotheses (dashboard/debug visibility)
  → Prisma final transcript
  → turn coordinator waits for semantic completion evidence
  → one normalized committed caller turn
```

STT media is streamed continuously at audio cadence. Bursting buffered frames
would break provider VAD timing and make both transcript quality and latency
measurements unreliable. Each final segment records text, provider segment ID
and provider latency for post-call evaluation.

### 3. Turn detection

```text
caller audio
  → optional enhancement
  → TEN VAD detects acoustic speech/pause
  → Pipecat Smart Turn decides whether the pause completes a thought
  → Gnani STT hypothesis/final is committed as one caller turn
```

TEN VAD works in small audio hops. Smart Turn prevents the agent from replying
to an incomplete phrase just because the caller paused for breath.

The two detectors answer different questions:

- **TEN VAD (acoustic):** is the current short PCM hop speech or non-speech?
- **Pipecat Smart Turn (semantic):** does the recent utterance sound complete
  enough for the assistant to respond?

The VoiceLink path feeds 16 kHz PCM into TEN in small hops. A debounced
speech-to-silence transition starts semantic analysis over a longer rolling
window. `COMPLETE` permits a commit; `INCOMPLETE` retains audio and transcript
state so the caller can continue the same sentence. A generation lock prevents
multiple boundaries from starting duplicate LLM turns.

Provider-final text and turn evidence are coordinated rather than allowed to
race. If semantic completion exists and a trustworthy provider final is already
available, Relay can commit immediately. A short grace period is retained only
when transcript text is still in flight. This removes avoidable endpoint delay
without responding to a half-finished sentence.

Silero VAD remains an optional comparison/fallback component. TEN + Smart Turn
is the primary production design because acoustic silence alone cannot tell
whether “kal party…” is a completed request.

### 4. Streaming conversation

1. A committed transcript enters a persona-aware `TurnSession`.
2. Relay resolves a prompt plan, working memory and approved context.
3. The LLM streams text.
4. Relay sends a safe sentence-level fragment to TTS as soon as possible.
5. Gnani streams PCM; Relay converts it to paced telephony frames.

This is designed to make the agent begin speaking before a full long answer is
complete.

The response contract is answer-first:

```json
{
  "answer": "Short spoken response...",
  "memory_summary": "Compact semantic state for the open call...",
  "memory_operations": []
}
```

The streaming parser extracts `answer` while its JSON string is arriving.
Sentence-safe fragments can therefore enter TTS while the model is still
producing memory fields. Memory generation never needs to block first speech.
The final parsed response still validates memory operations before applying
them.

### 5. Barge-in

When a caller begins talking while Relay is speaking:

```text
speech detected → reply marked interrupted → local outbound audio cleared
                → VoiceLink `clear` sent → further bot frames stop
```

Relay measures server detection to VoiceLink clear hand-off. It does not claim
that this equals handset playback stop time; the carrier does not report that
boundary.

Interruption is also a memory event. The generated answer and the heard answer
are stored separately:

```text
generated text        = complete model output, retained for audit
handed-off text        = fragments submitted as caller-facing audio
delivery status        = audio_sent | interrupted | failed
next-turn context      = only the handed-off portion
```

This prevents the next turn from assuming the caller heard a confirmation that
was cut off halfway through or never synthesized.

### 6. Prompt assembly and safety

Prompts are a tree of small composable instructions, cached in Redis after
resolution. The active policy enforces:

- first-reply AI disclosure and only one greeting per call;
- short, spoken, Roman-script Hinglish responses by default;
- follow-up interpretation for `haan`, `theek hai`, `bhej do`, `kar do` and
  `nahi`;
- no invented tool outcomes;
- privacy boundaries for OTPs, passwords, financial data, contacts, location,
  account details and private history;
- retrieved context treated as data, never as instructions.

#### How a prompt moves through the system

Relay does not store one uneditable prompt string. A prompt plan is a versioned
tree of small instructions with explicit priority and optional conditions. At
call creation, `PromptPlanResolver` performs this flow:

```text
PostgreSQL prompt tree
  → select active version for tenant/persona
  → evaluate node conditions against call context
  → sort matching nodes by priority
  → render one resolved plan
  → cache resolved plan in Redis
  → append beneath immutable safety rules
```

The final model input for a complex turn is assembled in this order:

```text
1. Mandatory base contract
   AI disclosure · privacy · consent · no fabricated action · speech style

2. Configured runtime instructions
   persona/tenant nodes · user-level preferences · language · business policy

3. Admin-approved conversation rules
   rules learned from reviewed calls, bounded and priority ordered

4. Verified account context
   caller/order snapshot · safe Calendar presence · permitted Plane result

5. Current-call state
   working memory · structured facts · rolling memo · heard dialogue only

6. Explicit prior-call recall context
   compact summaries only when recall is intentionally requested

7. Current caller utterance
```

Runtime instructions cannot override mandatory privacy and safety rules.
Retrieved knowledge, tool results and caller text are delimited as data, so a
prompt-like string inside an order record or transcript is never promoted to a
system instruction.

The prompt plan is normally resolved once for a call. Small account snapshots
are also call-scoped. This avoids repeated PostgreSQL/Redis/tool reads on every
turn and keeps prompt construction visible as its own latency metric.

#### Learned rules are governed, not self-modifying

Post-call feedback may ask an LLM to propose a generalized `do` or `don't`
rule. That proposal is always a candidate. An admin must inspect the source
turn, edit the wording, choose the rule type, assign an explicit priority and
approve it. Only active rules are loaded into future calls, and the active set
is bounded to prevent an ever-growing prompt from degrading latency or causing
conflicting behaviour.

### 7. Memory and context movement

| Layer | Job |
|---|---|
| Working memory | active request, confirmations and current-call facts |
| Episodic memory | compact post-call summary per caller |
| Structured facts | durable named facts with policy boundaries |
| Knowledge base | stable business/tenant information |
| Customer context window | small caller-scoped order/customer context |

Recall should be explicit. Relay should not mention an old message during an
unrelated new call unless the caller asks to revisit prior context.

#### Memory read path

At call setup, Relay resolves the authenticated/normalised caller to a customer
identity. It loads at most a small number of compact prior-call summaries; it
does not replay raw historic transcripts. The call then maintains separate
state containers:

| State | Lifetime | Example |
|---|---|---|
| `working_memory` | Current call | active intent, topic, confirmation state |
| structured facts | Current call, auditable | caller name, callback number, event time, work-item ID |
| rolling call memo | Current call | concise semantic state of what is being discussed now |
| heard turn history | Current call | caller text plus only assistant text handed to audio |
| prior-call summaries | Explicit recall/new-call seed | compact completed-call hand-offs |
| account context | Call-scoped, selectively refreshable | recent orders or Calendar presence |

The graph renders these sources independently. Structured facts protect exact
identifiers and corrections; the rolling memo preserves semantic continuity;
heard history keeps short acknowledgements such as “haan” attached to the
immediately preceding subject.

#### Memory write path

```text
caller turn committed
  → deterministic facts extracted/updated
  → LLM streams answer + proposed memory operations
  → operations validated against the current utterance
  → TTS delivery outcome recorded
  → rolling memo updated using what was actually handed off
  → call ends
  → compact completed-call summary persisted to episodic memory
```

Corrections supersede earlier values instead of coexisting ambiguously. An
interrupted reply retains its already handed-off prefix and marks the remaining
portion unfinished. A failed TTS turn stores the caller's unresolved request,
not the model's unheard answer.

Historic recall is intentionally gated. The normal new-call path knows that
prior summaries exist but does not inject their content into an unrelated
conversation. When the caller explicitly asks to remember a prior discussion,
the retrieval can happen as a separate conversational step, preserving the
current turn's responsiveness.

### 8. Approved tools and account context

Tools are adapters behind a policy boundary, not unrestricted model plugins.
Every tool has a read/write classification, caller-scope requirement and
failure representation. Tool output becomes verified context; the model cannot
convert an unexecuted plan into a success claim.

#### Google Calendar availability

Calendar uses owner-scoped OAuth and a pre-refreshed presence snapshot:

```text
OAuth → secure token storage → background snapshot refresh
     → call begins → safe availability context → spoken answer
```

The call does not wait for a live Google request. Relay can say “they are in a
meeting” or “availability cannot be verified,” but must not expose event title,
attendees, location, links or a complete schedule.

The OAuth token is owner-scoped and stored server-side. A background refresh
reduces the calendar to a minimal presence record such as `available`, `busy
until 16:30`, `stale` or `unavailable`. That snapshot is loaded before the
conversation needs it, so an availability question does not perform OAuth and
a Google network round trip while the caller waits.

#### Plane work-item status and update capture

```text
trusted caller?
  → explicit status/work-item request?
    → concrete identifier, e.g. SC-42?
      → retrieve exactly one item
        → render verified context for this one turn
          → speak concise status
```

Writes such as comments are stored as pending Relay actions. The owner must
approve them before anything reaches Plane.

The read path is deliberately narrow:

```text
normalised caller identity
  → trusted-caller check
  → explicit work-item/status intent
  → concrete identifier extraction and confirmation
  → one Plane API lookup
  → allow-listed fields rendered as verified context
```

For an update, Relay captures the requested comment and its target in its own
database. The dashboard shows a pending action. Approval performs the Plane
write with an idempotency/external identifier so a retry cannot create duplicate
comments. Rejection leaves an audit record and never contacts Plane.

#### Grocery orders and knowledge retrieval

The grocery persona receives a small caller-scoped order window at call setup.
If the caller names a different order ID, only that order is refreshed. Stable
business policies live in the knowledge base and are retrieved by relevance;
they are not dumped into every prompt. Order state is authoritative data, while
the LLM is responsible only for explaining it naturally and capturing the next
requested action.

### 9. MiniLM shadow classifier and future multi-agent route

The grocery persona uses multilingual MiniLM as a fast, switchable/shadow
intent router for:

```text
track_order · return_item · cancel_order · refund_status · wrong_item
size_issue · payment_issue · delivery_address · other
```

It guides the workflow; it does not create business truth. Order/customer
context and an approved operation must support any actual claim or action.

MiniLM currently runs safely in shadow beside the generic customer-facing
agent. It sees caller STT text, predicts one of the nine labels and stores the
label, confidence and model version without controlling the live reply. Human
reviewers approve or correct those predictions. The reviewed rows—not raw
synthetic accuracy—become the next training/evaluation set.

```text
real caller STT
  → shadow MiniLM prediction
  → confidence/drift monitoring
  → human-approved final label
  → reviewed dataset export
  → offline retraining + held-out evaluation
  → shadow release candidate
  → quality gate
  → gradual live routing
```

After it meets the defined accuracy, calibration and regression gates, MiniLM
can become a fast first-stage router to specialised subagents such as order,
refund, returns, payment and general-support agents. Promotion is reversible:
the generic agent remains a fallback, and a low-confidence route must not trap
the caller in the wrong flow.

### 10. Streaming TTS and outbound audio

Gnani Timbre v3.0 is used as a real-time synthesis socket rather than a
request/response voice-note API. Relay keeps model text and non-verbal audio
behaviour separate:

```text
first safe answer fragment
  → persistent Timbre WebSocket request
  → PCM chunks
  → ignore metadata/terminal events as audio
  → resample to negotiated VoiceLink rate
  → encode telephony codec
  → append to bounded outbound buffer
  → send one 20 ms frame per media tick
```

The TTS adapter validates requested model, voice, language, speed and sample
rate. Production speed is configured as a provider parameter—not by telling the
LLM to “speak slowly.” Each request recognises Gnani's terminal completion
event; a normal end is not treated as a disconnect and does not trigger a
reconnect/retry loop. Zero-audio completion, genuine socket failure and caller
interruption are tracked as different outcomes.

Relay begins synthesis from a speakable clause rather than waiting for the full
JSON response. It does not send unstable one-token fragments: fragmenting too
early can produce silent provider chunks, unnatural prosody and many expensive
TTS requests. The latency target is therefore **first real answer audio**, not
an empty first byte returned by a provider.

Non-verbal cues are deterministic audio policy:

- a light giggle is allowed only in low-stakes positive contexts and is
  disabled as a live latency filler when it would create a giggle-then-silence
  experience;
- a clear-throat timer can fire after 600 ms without first model text;
- cue playback and answer playback are serialized so their PCM streams never
  overlap;
- cue frames and actual answer frames are measured separately.

### 11. Analytics, evaluation and active learning

Relay stores caller STT ↔ agent reply pairs and supports:

- intent judgement using STT only;
- conversation-flow judgement using caller and agent turns;
- MiniLM shadow prediction logging;
- human review and corrected labels;
- future classifier training from reviewed data rather than synthetic-only
  validation.

#### Online observability

Every turn carries a shared stream/call/session identity across STT, LLM, TTS,
conversation storage and latency storage. This makes the event sequence
reconstructable without matching timestamps by hand.

| Stage | Representative measurements |
|---|---|
| Endpointing | `speech_end_to_commit_ms`, provider-final wait, semantic-turn wait |
| Prompt setup | account context, router, retrieval and prompt-build duration; prompt size/version |
| LLM | request setup, model TTFT, answer-decoder TTFT, total stream time and token counts |
| TTS | text-to-fragment, connect time, first audio, zero-audio retries and reconnect count |
| Telephony | first cue frame, first answer frame, first **real** answer frame, frame-gap p50/p95/max |
| Reliability | JSON parse, LLM/TTS failure, silent/interrupted/partial-delivery turns |
| Conversation | repeated greeting, repeated clarification, correction handling, fabricated action |
| Audio/STT | Hinglish WER, false/missed endpoint, noise trigger and barge-in stop latency |

`first_answer_audio_ms` may include an intentional cue. The product-performance
number used for genuine responsiveness is `first_real_answer_audio_ms`, which
starts at the first frame produced from the LLM's answer.

#### Layer 4: controlled real-telephony evals

Versioned scenarios place real outbound calls through VoiceLink and ask a human
evaluator to follow a fixed script. Scenarios cover complete messages,
corrections, privacy challenges, acknowledgements, Plane status, Calendar
availability and verified retail orders. The test therefore includes carrier,
codec, endpointing, STT, generation, synthesis and playback transport—not only
a text harness.

Expected results are stored with the eval run but are not injected into the
live agent. After the call, Relay correlates the provider call ID, metrics,
transcript, judge evidence and recording URL. DID recording must be enabled at
VoiceLink; the database stores the controlled URL rather than copying binary
recordings into PostgreSQL.

#### Layer 5: sampled production evals

Completed real calls enter review in two ways:

1. deterministic stable sampling using a hash of the session ID;
2. targeted escalation for judge failure, failed quality verdict, critical
   issue, requested review or interrupted/failed answer delivery.

Stable sampling makes reprocessing idempotent. The post-call Bedrock judge
scores task completion, context fidelity, clarification, grounding, coherence,
naturalness, concision and persona consistency on a 0–4 scale. It receives
assistant text actually handed to the caller; the full generated-but-unheard
answer remains audit data and is excluded from quality judgment.

The intent judge is a separate request that receives STT text only. This avoids
letting the assistant's response leak the intended label back into intent
classification.

#### Turn-level human feedback and rule governance

An admin can grade each heard answer as:

```text
approved · disapproved · never_say_this · reduce_frequency
```

An optional comment explains the desired behaviour. Negative feedback—or an
approval with a substantive comment—can ask Bedrock to propose one generalized
candidate rule. The generator receives the caller utterance, handed-off answer,
delivery state and reviewer comment. It is instructed to remove private facts
and produce a reusable behaviour rather than memorise the exact conversation.

```text
human feedback
  → inactive LLM rule proposal
  → admin reviews source evidence
  → admin edits do/don't type and wording
  → admin assigns explicit priority
  → approve or reject
  → approved rule becomes available to future call sessions
```

No generated rule self-activates. Active guidance is read when a new session is
created, so rule generation adds no latency to the live call that produced the
feedback.

#### Release gates

Voice changes should be evaluated on distributions rather than one impressive
call. A release set should contain clean and noisy Hinglish, short pauses,
corrections, barge-in, long answers, multiple speakers and tool failures. The
primary gates are first-real-audio p50/p95, silent-turn rate, premature-turn
rate, barge-in stop p50/p95, Hinglish WER, task completion and critical safety
failures. MiniLM additionally requires per-class recall, macro F1, calibration
and a reviewed confusion matrix before live-routing promotion.

## Persistence and data ownership

PostgreSQL is the durable source of truth; Redis is an acceleration layer and
may be rebuilt. The main data domains remain separate so retention and access
policies can be applied independently.

| Domain | Representative records | Purpose |
|---|---|---|
| Call ledger | `conversation_calls`, `conversation_turns` | STT/assistant pairs, delivery state, summaries and judge evidence |
| Episodic memory | customer/session summaries | Compact caller-scoped completed-call context |
| Operational latency | call and turn metrics | Endpoint, prompt, LLM, TTS and telephony timings |
| Intent feedback | shadow observation + reviewed final label | MiniLM learning and export |
| Production evals | eval runs, turn feedback, candidate rules | Layer-4/5 review and governance |
| Prompt system | plans, versions, conditional nodes | Configurable tenant/persona instructions |
| Tools/actions | Calendar tokens/snapshots, Plane pending actions | Approved integration state |
| Business context | customers, orders, knowledge chunks | Persona-specific authoritative data |

A conversation is joined end-to-end by the provider stream/session ID. A
customer ID scopes account memory; it is not used as the call's unique key.
Sensitive transcripts, recordings and tokens require explicit retention,
deletion, encryption and operator-access policies before multi-tenant use.

## Operator dashboard and mobile shell

The React dashboard is designed first for a 9:16 mobile viewport and can run
inside the Flutter WebView shell. It consumes protected APIs rather than a
separate demo database. Its operational surfaces include:

- active call state and live SSE pipeline events;
- completed caller/assistant transcript pairs and compact summaries;
- granular voice timings and failure evidence;
- MiniLM prediction review and corrected labels;
- Layer-4/5 evaluation queues;
- per-answer human quality feedback and proposed-rule approval;
- prompt-plan instructions and version state;
- Google Calendar connection status;
- pending Plane actions requiring owner approval.

The browser never contains provider secrets. Production dashboard requests are
same-origin or pass through an authenticated boundary; `VITE_AGENT_API_KEY` is
only acceptable for local development.

## Failure semantics

Voice systems fail partially, so Relay avoids a single generic exception state:

| Failure | Behaviour |
|---|---|
| STT disconnect | Preserve the call, reconnect where safe and never commit empty text |
| VAD false boundary | Semantic incomplete result keeps listening |
| LLM failure | Keep the call alive, emit a short recoverable response and retain the caller request |
| Invalid streamed JSON | Record parse failure; do not apply memory operations blindly |
| TTS completes with zero audio | Mark silent/failed delivery and retry only under bounded policy |
| Caller barge-in | Cancel/clear output, retain heard prefix and process the new caller turn |
| Calendar/Plane timeout | Say availability/status cannot be verified; never fabricate stale truth |
| PostgreSQL analytics failure | Do not block already generated live audio; emit an operational error |
| Candidate-rule generation failure | Preserve human feedback; never alter the active prompt |

This distinction is important for both memory and evaluation: “the model wrote
an answer” is not equivalent to “the caller heard an answer.”

---

# Voice performance engineering

## Correct metrics

The current measurement schema prevents queueing from being misreported as
model TTFT and separates cue audio from the first real answer frame. Old,
incompatible schema rows are excluded from current percentile summaries.

| Metric | Definition |
|---|---|
| Model TTFT | turn processing begins → first LLM text token |
| Queue delay | committed transcript → processing begins |
| End-to-end TTFT | committed transcript → first LLM text token |
| First outbound audio | committed transcript → first outbound media frame, including an intentional cue |
| First real answer audio | committed transcript → first frame synthesized from the LLM answer |
| Barge-in stop | caller speech detected → VoiceLink clear hand-off |

Early six-turn production diagnostic baseline—not a finished benchmark:

| Metric | Current value |
|---|---:|
| Model TTFT p50 | 579 ms |
| Model TTFT p95 | 824 ms |
| End-to-end TTFT p50 | 1,065 ms |
| End-to-end TTFT p95 | 1,748 ms |
| Queue delay p50 | 470 ms |
| Queue delay p95 | 993 ms |
| First outbound audio p50 | 924 ms |

This is an early historical diagnostic sample, not the permanent launch claim.
Use the dashboard to insert a current, version-matched
`first_real_answer_audio_ms` distribution before publishing an external
benchmark. The useful engineering conclusion remains: production delay is not
solely model speed. Queueing, endpointing and TTS chunk continuity are visible,
separately measurable work.

## Audio cues

### Clear throat

Production VoiceLink uses a rule-based 600 ms timer. If no first LLM text
arrives after a committed caller turn, Relay can synthesize standalone
`<clears throat>` audio. The actual answer follows cleanly after the cue. This
reduces perceived dead air; it does **not** reduce actual TTFT.

### Giggle

The local microphone harness supports a standalone `<giggle>` cue after 450 ms
of pre-token silence, while generation continues in parallel. Live VoiceLink
giggle delay fill is currently disabled because giggle followed by a long pause
felt worse than natural silence. Never use it for complaints, refunds,
conflict, urgency or sensitive conversations.

---

# Deployment

```text
push to origin/master
  → GitHub Actions regression tests
  → pinned SSH connection to production VPS
  → checkout exact origin/master revision
  → Docker Compose image build
  → API/PostgreSQL/Redis service replacement
  → private local /healthz retry loop
  → public HTTPS /healthz verification
```

Production runs on a VPS with an API container, PostgreSQL, Redis and the HTTPS
reverse proxy. GitHub Actions authenticates with a pinned host key and a scoped
deployment SSH key. Runtime secrets remain in the VPS environment file and are
not stored in the repository. The AWS ECS workflow remains a manual fallback;
it is not triggered by a normal production push.

The Compose network exposes only the API loopback port to the host. PostgreSQL
and Redis remain on the private Docker network. The public reverse proxy owns
TLS and forwards HTTPS/WSS traffic to the local API.

The public production host is `voice.sidecut.me`. Do not place keys, tokens,
database URLs, raw audio, private transcripts or model ZIPs in source control.

## Provider and feature swap points

Construction is centralized in `app/container.py`. Interfaces isolate the
conversation graph from concrete providers, allowing a deployment to swap one
layer without rewriting the others.

| Layer | Current production-oriented implementation | Swap boundary |
|---|---|---|
| Telephony | VoiceLink WebSocket + REST | Provider-specific media adapter |
| STT | Gnani Prisma v3.0 WebSocket | Streaming STT contract |
| Turn detection | TEN VAD + Pipecat Smart Turn | Incremental PCM boundary detector |
| LLM | Configured hosted chat model/Bedrock adapter | Streaming `ChatModel` contract |
| TTS | Gnani Timbre v3.0 WebSocket | Streaming TTS contract |
| Intent | Generic route + MiniLM shadow/switchable route | Router contract |
| Prompt plan | PostgreSQL tree + Redis cache | `PromptPlanResolver` |
| Memory | PostgreSQL episodic/call stores | Memory-store contracts |
| Tools | Calendar, Plane, orders and knowledge adapters | Account-context/tool providers |

Environment variables select providers and safe modes; source code must not
contain account credentials. `.env.example` documents names and defaults while
`.env`, `.env.local-cloud`, model archives, audio recordings and exported
review data remain ignored.

## Important API surfaces

| Surface | Purpose | Authentication |
|---|---|---|
| `GET /healthz` | Container/reverse-proxy health | Public minimal response |
| `WSS /v1/media/voicelink` | Bidirectional call media | VoiceLink token |
| `GET /v1/dashboard/events` | Live runtime SSE stream | Dashboard API key |
| `GET /v1/dashboard/calls/{session_id}` | Transcript, summary and eval detail | Dashboard API key |
| `/v1/dashboard/prompt-plan/*` | Read/update composable instructions | Dashboard API key |
| `/v1/dashboard/evals/telephony/*` | Layer-4/5 runs and human review | Dashboard API key |
| `/v1/dashboard/evals/rules/*` | Candidate-rule approval/rejection | Dashboard API key |
| `/v1/integrations/google-calendar/*` | OAuth connect/status/disconnect | Dashboard API key except provider callback |
| Plane action endpoints | Draft, inspect and approve pending work-item comments | Dashboard API key |

Production authorization should evolve from the current owner API key into
tenant-aware sessions and role-based access before exposing administration to
multiple customers.

---

# Current priorities

1. Distinguish `interrupted_before_first_audio` from a genuinely silent turn.
2. Safely supersede stale LLM generation after barge-in to reduce queue delay.
3. Add adaptive—not fixed—TTS buffering for short initial chunks/provider gaps.
4. Improve spoken work-item ID confirmation and normalisation.
5. Gather 30+ labelled clean/noisy Hinglish calls before publishing p95/p99.
6. Ensure every dashboard view uses protected real backend data.
7. Continue the reviewed-data loop for MiniLM.

---

# Local development

Python 3.12+ and `uv` are required.

```bash
uv sync
uv run python -m unittest tests.test_voice_evaluation tests.test_enquiry_pipeline
```

### Local microphone pipeline

```bash
uv run python -m gnaniassignment.pipeline.gnani_microphone \
  --device default --with-llm --with-tts
```

Use `arecord -L` and `aplay -L` to find valid audio devices. Use headphones to
avoid speaker output feeding back into the microphone.

### Local microphone against hosted VoiceLink WSS

```bash
export VOICELINK_WS_TOKEN='obtain securely; never save in source files'
uv run python -m gnaniassignment.pipeline.voicelink_ws_simulator \
  --input-device default \
  --playback-device default \
  --customer-id local-voicelink-test
```

### TEN / Smart Turn diagnostic

```bash
uv run python -m gnaniassignment.pipeline.ten_smart_turn_microphone \
  --device default
```

Before a production voice change, run relevant tests and let the owner test
the audible local STT → LLM → TTS path.

---

# Learn more

- [Full Claude Code handoff](docs/claude-handoff-full.md)
- [Launch story and product-video narrative](docs/relay-launch-story.md)
- [Voice evaluation documentation](evals/voice/README.md)
- [GCP deployment guide](infra/gcp/README.md)

> **Relay turns a missed call into a safe, real-time, context-aware hand-off.**
