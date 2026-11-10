# Relay — product launch story and technical proof

## One-line description

Relay is a personal AI phone agent: when a caller reaches the owner, Relay can
understand the request in Hinglish, respond in a natural voice, capture a
message, retrieve authorised work or calendar context, and leave the owner
with a searchable, auditable call record.

It is designed as a configurable platform, rather than a single hard-coded
bot: prompts, knowledge, provider adapters, business tools and account-level
instructions can be changed without rebuilding the voice pipeline. The same
voice infrastructure can become a neighbourhood grocery-store concierge or a
corporate employee’s work assistant.

## The problem

Calls are still the fastest way for people to ask for something urgent, but
the owner cannot always answer. A missed call loses context; a voicemail is
friction; a generic IVR is impersonal; and a typical chatbot cannot act as a
real-time phone presence.

Relay answers as an explicitly disclosed AI assistant for the owner. Its job
is not to pretend to be the owner. It understands what the caller needs,
collects the minimum missing information, uses only authorised data/tools, and
captures an actionable hand-off.

## What was built

### The two product personas

Relay is a platform with a shared voice core and different account personas.
The phone number, prompt plan, available knowledge and tools determine what
the agent is allowed to do.

| Persona | Who calls | What Relay does | What it must not do |
|---|---|---|---|
| **Grocery-store personal agent** | Neighbourhood grocery customers | Answers delivery and order questions, finds an order, captures a return/cancellation/refund request, confirms address or payment issues, and hands off a concise task to the store owner | Invent order facts, promise a refund/return, expose another customer’s order, or claim an action was completed without the connected system confirming it |
| **Corporate working-employee agent** | Colleagues, clients and collaborators | Captures a message, explains the owner’s approved availability, retrieves a specified Plane work-item status for trusted callers, records a proposed comment/action for owner approval, and gives the owner a concise post-call hand-off | Expose private calendar details, reveal work items to unauthorised callers, write external comments without approval, or impersonate the employee |

The core conversation rules are the same for both personas: identify as an AI
assistant, understand the caller’s need, use only authorised context, ask for
only the missing detail, and leave behind a clear action or summary.

#### Grocery-store personal agent

The grocery persona is the natural successor to the MiniLM intent-classifier
work. It can be configured with the retail intent set:

`track_order`, `return_item`, `cancel_order`, `refund_status`, `wrong_item`,
`size_issue`, `payment_issue`, `delivery_address`, and `other`.

The MiniLM classifier is a fast routing layer, not a substitute for the LLM.
It narrows the likely task; the voice agent still uses the caller’s actual
words, customer-scoped order context and approved tools before making a claim.
For example:

```text
Caller: “Mera order abhi tak nahi aaya.”
MiniLM shadow route: track_order
Relay: retrieves only that caller’s small order window
Relay: “Main check karti hoon. Aap order number bol sakte hain?”
```

This persona is appropriate for a local business where callers want quick,
repeatable answers and the owner needs structured follow-up instead of a pile
of missed calls.

#### Corporate working-employee agent

The corporate persona acts as an opt-in executive/employee presence on the
phone. It can answer: “Is Tushar available?”, “Can you pass this message?”,
or, for an authorised colleague, “What is the status of SC-42?”

It is deliberately not a broad internal-data search engine. It retrieves only
the small, explicitly relevant context needed for the current request. This
keeps calls fast, reduces accidental disclosure and gives the owner an
approval step before anything externally mutating happens.

### 1. Real telephony entry point

- VoiceLink provides Indian telephony routing and a bidirectional WebSocket
  media stream.
- Inbound media is documented 8 kHz, A-law, mono audio.
- Relay validates the media format, decodes A-law to PCM, resamples for speech
  services, then sends reply audio back as paced 20 ms A-law frames.
- The production WebSocket is authenticated with a secret token. It is not an
  open public audio endpoint.
- VoiceLink lifecycle and media events are recorded as safe operational events,
  without storing raw audio in the evaluation store.

### 2. Streaming voice pipeline

```text
Caller
  → VoiceLink PSTN / WebSocket
  → A-law decode + resample
  → optional audio enhancement
  → TEN VAD + Pipecat Smart Turn endpointing
  → Gnani streaming STT
  → Relay session / prompt / approved context
  → streaming LLM
  → Gnani streaming TTS
  → paced VoiceLink media frames
  → caller
```

Key interaction behaviours built into the pipeline:

- Hinglish speech handling.
- Streaming generation: speech begins from an LLM fragment, rather than after
  waiting for the full response.
- Barge-in: new caller speech sends a VoiceLink `clear`, clears queued bot
  audio, and stops additional output.
- TEN acoustic VAD detects speech boundaries using small audio hops.
- Pipecat Smart Turn checks whether a pause is semantically complete before
  committing a turn, preventing replies to mid-sentence pauses.
- A configurable proactive clear-throat cue can fill a slow response gap. It
  is a latency mask only, never a replacement for lower TTFT.
- Replies are short, speech-oriented and constrained by a personal-assistant
  safety policy.

#### Micro-cues: where giggle and clear throat belong

Voice agents need to fill silence without sounding like they are reading a
script or masking a broken system. Relay treats these as **audio-pipeline
behaviours**, not words inserted by the LLM prompt.

**Light giggle** is an optional conversational acknowledgement cue. Relay has
two supported uses for it: a low-stakes positive reply can carry a controlled
`<giggle>` TTS tag, and the local microphone harness can play it as a
standalone latency cue when no first LLM text has appeared after **450 ms**.
The latter starts answer generation in parallel but holds playback until the
giggle ends, so the caller does not hear overlapping streams. It is rendered
by the TTS adapter—not emitted as model text such as `[giggle]`.

Giggle latency fill is **currently disabled in production VoiceLink** because
the cue-plus-pause experience was worse than a short natural silence. It
should never be used for a complaint, cancellation, refund, conflict, urgent
request or sensitive message.

**Clear throat** is a latency cue. A rule-based timer begins when Relay has
committed a caller turn and requested the LLM answer. If no first LLM text
chunk has arrived by the configured threshold, production can synthesize a
brief clear-throat audio cue. The current configured threshold is **600 ms**.
When answer text arrives, the cue is allowed to finish cleanly; then the TTS
for the actual reply begins. This avoids overlapping telephony streams.

```text
Caller finishes speaking
  → turn is committed
  → LLM begins streaming
  → no first text by 600 ms? brief clear-throat cue
  → first LLM text arrives
  → cue finishes, if started
  → spoken answer streams to caller
```

The cue does not improve LLM TTFT. It only makes a slow turn feel less like
dead air. Its activation rate must be measured: a low rate is healthy; a high
rate points to an endpointing, queueing, LLM or TTS problem that should be
fixed rather than hidden.

### 3. Personal-agent conversation behaviour

The assistant is configured as a female personal phone assistant in casual
Roman-script Hinglish. It:

- discloses that it is an AI assistant on the first reply of each call;
- greets only once, then continues naturally instead of restarting the
  introduction;
- treats short confirmations such as `haan`, `theek hai`, `bhej do` and
  `nahi` as follow-ups to the immediate topic;
- captures messages and confirms them once before claiming any action;
- never claims a message was sent, an event changed, or an external task was
  completed unless an approved tool actually completed it;
- keeps private information, OTPs, financial data, locations and account data
  protected by default.

### 4. Memory designed for phone calls

Relay does not dump a long conversation history into every model request.

- **Working memory** holds the current call’s active facts and immediate
  dialogue state.
- **Episodic memory** stores a durable, per-caller summary after a call.
- **Structured facts** preserve useful facts without treating old transcript
  text as an instruction.
- **Customer context window** preloads small caller-scoped information such as
  mock grocery orders where configured.
- **Memory recall** is deliberate: the assistant should retrieve prior context
  only when the caller asks to remember or revisit something, rather than
  injecting old details into a new, unrelated call.

This gives the agent continuity without making old context dominate a live
conversation or fill the context window.

### 5. Configurable prompt tree

Prompting is modular rather than one giant string.

- Small instruction nodes describe identity, safety, language, behaviour,
  tools and scenario-specific rules.
- A resolver builds the active prompt plan.
- Redis can cache resolved plans so SQL prompt retrieval is not on the
  real-time voice critical path.
- The dashboard is intended to make these instructions configurable by a user
  later, including user-level instructions and tool configuration.

### 6. Tools and approved account context

Relay supports tool adapters with a clear safety boundary.

#### Google Calendar: availability without calendar leakage

Google Calendar is connected through owner-scoped OAuth. Relay stores the
authorised token securely and uses it to create a small **presence snapshot**:
for example, available now, busy until a time, or in a meeting. It is not
designed to paste the whole calendar into the LLM context.

The important latency decision is that the snapshot is refreshed outside the
caller’s waiting path. At call setup, the agent reads the most recent local
presence snapshot instead of making a live Google API request while the caller
is on the line. That means a caller can ask, “Are they free right now?” without
the call having to wait for OAuth or Google Calendar network latency.

The prompt determines the disclosure level. A safe response is:

> “Woh abhi meeting mein hain. Main aapka message un tak pahucha deti hoon.”

It should not reveal private titles, attendee lists, locations, meeting links,
or the owner’s entire schedule. If the snapshot is unavailable or stale,
Relay says it cannot verify availability instead of inventing it.

#### Plane work items: an active, but constrained status assistant

Plane is connected as an approved work-item adapter. The flow is deliberately
specific:

1. The inbound caller number is normalised and checked against the owner’s
   trusted-caller allowlist.
2. The caller explicitly asks about a task/work item/status and supplies a
   concrete identifier, such as `SC-42`.
3. Relay invokes Plane for that one item, not the entire workspace.
4. The returned status/title/allowed summary is rendered into verified context
   for the current LLM turn.
5. Relay speaks the status in natural Hinglish and can capture a requested
   follow-up.

This makes Plane active in a useful way: it can answer a colleague directly
from the live source of truth rather than asking the owner to remember an old
status. It also prevents the dangerous version of the feature—an unauthorised
caller browsing a company’s project data.

For write-like work, such as “Please add this comment to SC-42,” Relay creates
a pending action in its own app. The owner reviews and approves it first; only
then is it sent to Plane. That approval queue turns the voice agent into an
assistant rather than an unbounded automation bot.

This is the platform direction: plug in tools such as Plane, Asana, calendars,
CRMs or internal APIs, but keep identity, approval and data access under the
account owner’s control.

### 7. Retail-intent research and classifier work

Before the personal-agent path, a multilingual MiniLM was fine-tuned for
intent classification using IndicJevBench sources:

- `intent_massive`
- `fintech_banking77`
- `hinglish_lid`
- `synthetic_enterprise`

The project also created a 60,000-row synthetic retail dataset for intents:

`track_order`, `return_item`, `cancel_order`, `refund_status`, `wrong_item`,
`size_issue`, `payment_issue`, `delivery_address`, and `other`.

The MiniLM is retained as a switchable/shadow classifier. In the grocery-store
persona, it becomes the fast intent route into the right order/return/refund
flow. In the corporate persona, the generic agent can remain the primary route
while Plane and Calendar are explicitly tool-gated. In either persona, real
conversations are logged, reviewed and later used to improve routing instead
of blindly trusting synthetic validation accuracy.

### 8. Conversation analytics and human-in-the-loop improvement

Every call can produce a complete transcript pairing:

```text
caller STT text → agent reply → caller STT text → agent reply → …
```

After the call:

- transcript segments and replies are stored in PostgreSQL;
- a Bedrock judge can judge the intent from STT text only;
- conversation-flow analysis can inspect STT + assistant replies for patterns
  such as missed clarification, repeated greetings, loops or poor hand-offs;
- classifier predictions can be reviewed and corrected;
- reviewed data becomes a safer training signal for future classifier work.

The intent judge and conversation-quality judge are deliberately separate: the
first should not infer intent from the assistant’s wording, while the second
needs both sides of the conversation to detect a bad interaction.

### 9. Operational dashboard and mobile path

A React mobile-first dashboard was built for a Flutter WebView wrapper.

It is intended to show:

- active-call state and agent identity;
- live operational events;
- per-call caller/agent transcript pairs;
- MiniLM review queue;
- conversation quality records;
- latency and voice-evaluation data;
- prompt and tool configuration surfaces.

A Flutter shell provides the route to install Relay as a mobile app while
keeping the product UI web-driven and faster to iterate.

### 10. Deployment and operations

The production path is automated:

```text
Git push to production branch
  → GitHub Actions regression tests
  → Docker image build
  → Artifact Registry
  → secure deployment to GCP VM through IAP
  → container health check
  → public HTTPS /healthz verification
```

Runtime components:

- GCP VM for the API and real-time voice process.
- Docker Compose for API, PostgreSQL and Redis.
- Artifact Registry for immutable images.
- GitHub Actions + GCP workload identity for deploy automation.
- Cloud DNS/domain routing for `voice.sidecut.me`.
- protected API key for dashboard and management APIs.

The deployment workflow protects the current service: it validates container
health and restores the previous image if the new one is unhealthy.

## Latency and quality engineering

The project did not stop at "the call connected." It added instrumented,
turn-level measurements to understand where a voice conversation waits.

### Current measured production metrics

The table below is the first corrected production sample after metric-schema
version 2 was deployed. It contains six measured turns; it is a diagnostic
baseline, not a mature public benchmark.

| Metric | Current value |
|---|---:|
| Model TTFT p50 | 579 ms |
| Model TTFT p95 | 824 ms |
| End-to-end TTFT p50 | 1,065 ms |
| End-to-end TTFT p95 | 1,748 ms |
| Queue delay p50 | 470 ms |
| Queue delay p95 | 993 ms |
| First outbound audio p50 | 924 ms |

Definitions matter:

- **Model TTFT**: first LLM token after the turn begins processing. It does
  not include endpointing or a waiting turn queue.
- **End-to-end TTFT**: committed STT transcript to the first LLM token. It
  includes turn queueing.
- **Queue delay**: committed transcript waiting before processing begins.
- **First outbound audio**: first media frame handed from Relay to VoiceLink.
  It is a server-side proxy, not a claim that the caller’s handset has already
  played sound.

This distinction was earned through debugging. The first dashboard version
incorrectly combined queued-turn delay with model TTFT and could use a stale
VAD end point as the audio latency anchor. Version 2 separates those timings
and excludes old incompatible records instead of mixing them into p95.

### What the measurements revealed

1. The LLM was not actually taking eight seconds. A reported 8.49-second
   legacy p95 contained time waiting behind an earlier active turn.
2. Turn queueing is now visible as its own problem: current p95 is 993 ms.
3. A direct production TTS probe showed Gnani can emit a small first PCM chunk
   quickly, then leave a material chunk gap. This can cause outbound playback
   starvation and is separate from LLM TTFT.
4. The VoiceLink sender itself correctly paces 20 ms A-law frames once PCM is
   available.
5. Barge-in hand-off is quick at the server boundary, but a carrier cannot
   confirm exact handset stop time, so Relay does not overclaim it.

### Current engineering backlog

- Separate `interrupted_before_first_audio` from a true silent turn. A caller
  interruption should not be reported as a failed silent response.
- Reduce turn queueing by cancelling/superseding stale generation safely after
  a barge-in.
- Add an adaptive outbound jitter buffer for short initial TTS chunks without
  imposing a fixed delay on every reply.
- Measure at least 30 clean, labelled turns before publishing p95/p99 claims.
- Expand labelled noisy Hinglish STT evaluation: fan noise, traffic, TV audio,
  code switching, pauses and overlapping speakers.

## A suggested launch-video story

### Opening — the missed-call problem

Show a phone ringing while the owner is busy. The caller needs to leave an
important message or ask for a work status. The owner misses the context.

**Voiceover:** “Calls are still where urgent things happen. But you cannot
always pick up.”

### The reveal — Relay answers

Show the call arriving at Relay, then a clean mobile transcript.

**Voiceover:** “Relay is my AI phone agent. It answers for me, in Hinglish,
understands what the caller needs, and gives me the context—not just a missed
call notification.”

### The conversation — natural, interruptible speech

Show the real-time chain visually: caller speaks → STT text appears → LLM
streams → voice reply starts. Demonstrate a caller interrupting the agent and
asking a different question.

**Voiceover:** “It streams. It knows when I am still speaking. And when I
interrupt it, it stops. If a response is late, it uses a tiny natural audio
cue instead of leaving dead air.”

### The proof — tools and memory with boundaries

Show both persona cards. First, show a grocery customer checking an order and
the MiniLM routing to the correct delivery flow. Then show a trusted colleague
asking about `SC-42`, a safe calendar availability answer, and the approval
queue for a proposed Plane comment.

**Voiceover:** “For a local grocery store, it understands orders and returns.
For work, it can use the tools I approve—calendar, work items, internal
systems—without exposing everything to every caller.”

### The builder story — not a demo wrapper

Animate the production pipeline and dashboard diagnostics.

**Voiceover:** “This is not a voice-note bot. It is a production voice stack:
telephony, streaming STT, smart turn detection, streaming LLM, streaming TTS,
memory, tools, analytics, and deployment automation.”

### The engineering receipt

Show the metrics card above with the qualifier `early production baseline —
six measured turns`. Emphasise the latency decomposition rather than claiming
finished perfection.

**Voiceover:** “Every turn is measured. We can separate model speed from
queueing and from the carrier hand-off—and use the data to improve the next
call.”

### Closing

Return to the owner’s mobile dashboard with a concise call summary.

**Voiceover:** “Relay makes missed calls useful.”

## Launch claims that are safe today

- “A real-time personal AI phone agent for authorised calls.”
- “Built for natural Hinglish conversations.”
- “Streaming STT, LLM and TTS with interruption handling.”
- “Configurable prompts, memory and approved tool integrations.”
- “Private dashboard, call transcripts and review workflow.”
- “Deployed with automated tests, immutable containers and health checks.”

## Claims to avoid until further evidence exists

- Do not claim handset-playback latency: the current measurement ends when
  Relay hands a frame to VoiceLink.
- Do not claim p95/p99 production performance as a stable benchmark from six
  turns.
- Do not claim perfect STT or work-item recognition; identifier transcription
  remains a live robustness issue.
- Do not claim that tools can take irreversible actions without owner approval.
