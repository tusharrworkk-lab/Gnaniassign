# Gnani Assignment

An intent-routed enquiry assistant. The current implementation accepts text or completed STT phrases, classifies the query with Laya through Vercel AI Gateway, and uses one focused small-model agent to produce a short answer and a compressed memory summary. Audio STT and TTS adapters are reserved for later integration.

## Run

Python 3.12 or newer is required. Install with `uv sync --extra local-laya`, then fill in `AI_GATEWAY_API_KEY` in the root `.env` file. The answer subagents use `meta/llama-3.1-8b` through Vercel AI Gateway by default; `SUBAGENT_MODEL` optionally overrides that model ID, and `SUBAGENT_GATEWAY_URL` optionally overrides the OpenAI-compatible endpoint. The app loads `.env` automatically and does not override variables already set in your shell. `LAYA_GATEWAY_URL` and `LAYA_MODEL` configure the hosted Laya endpoint and model. `.env` is ignored by Git; `.env.example` records the expected variable names.

To run intent classification locally on an NVIDIA GPU, install `uv sync --extra local-laya` and set `INTENT_PROVIDER=local_laya`, `LOCAL_LAYA_DEVICE=cuda`, and `LOCAL_LAYA_BACKEND=eager` in `.env`. The first startup downloads the public `convaiinnovations/laya` checkpoint. Startup also warms the model; later sessions in the same process reuse it. CUDA is fail-fast: the app refuses Laya's silent CPU fallback when CUDA was requested. Run `uv run python -m gnaniassignment.pipeline.laya_diagnostic` before startup; `nvidia-smi` and that diagnostic must report a visible GPU. `INTENT_PROVIDER=laya` keeps the hosted Vercel route. The Vercel answer subagent remains hosted in either mode.

## Native laya.cpp router (recommended local experiment)

`laya.cpp` runs Laya through native ggml/CUDA and exposes the same JEV-compatible `/v1/systemone` API. It keeps one checkpoint resident, so our Python process only makes a loopback HTTP call per intent. Do **not** invoke `laya-cli` per user turn.

For the GTX 1650 Ti (SM 75), use the upstream Linux CUDA 13 release: it contains an SM 75 kernel and avoids installing a CUDA compiler. Download its latest `linux-amd64-cuda13` executable into an ignored `local/laya.cpp/` directory, make it executable, then fetch the English checkpoint from the cloned upstream repository.

```bash
mkdir -p local/laya.cpp
git clone --recurse-submodules https://github.com/lkarlslund/laya.cpp.git local/laya.cpp/source
cd local/laya.cpp/source
python3 scripts/download_model.py --variant english
# Put the downloaded laya-rNNNN-linux-amd64-cuda13 executable at ../laya-cli.
chmod +x ../laya-cli
../laya-cli --server --host 127.0.0.1 --port 8080 \
  --model models/laya --variant english --tensor-core-fp32 --flash-fp32 \
  --batch-wait-ms 0
```

`--bf16` is not applicable to the GTX 1650 Ti: it needs compute capability 8.0 or newer. The optimized FP32 flags above still use the native CUDA Tensor Core path. After `/health` reports ready, set `INTENT_PROVIDER=laya_cpp` in `.env`; `LAYA_CPP_URL`, `LAYA_CPP_MODEL` (normally `laya`), and an optional `LAYA_CPP_API_KEY` are defined in `.env.example`. Keep the terminal that hosts the server running, then measure the end-to-end split with:

```bash
uv run python -m gnaniassignment.pipeline.check --runs 10 "Where do you operate?"
```

The native route removes Python framework overhead, but it cannot guarantee a 25 ms single-query result on a GTX 1650 Ti. Measure its warm median against the current ~516 ms Python baseline before adopting it.

## Local Gemma 3 1B with vLLM

The retail BERT intent router remains local and chooses the specialist agent. Gemma is the local streamed answer model. vLLM exposes an OpenAI-compatible endpoint, so the pipeline measures its TTFT in exactly the same way as the Vercel fallback.

First accept the Gemma licence on the [Google Gemma 3 1B model page](https://huggingface.co/google/gemma-3-1b-it), then authenticate Hugging Face in the same shell (or set `HF_TOKEN` in the ignored `.env`). Install the optional server dependency and start one persistent server:

```bash
uv sync --extra vllm
uv run --extra vllm vllm serve google/gemma-3-1b-it \
  --host 127.0.0.1 --port 8000 --api-key local-gemma \
  --dtype float16 --gpu-memory-utilization 0.75 \
  --max-model-len 2048 --max-num-seqs 1 --enforce-eager
```

This vLLM configuration requires a BF16-capable GPU (Ampere or newer). Gemma 3 is not supported in FP16 by vLLM due to numerical instability. The GTX 1650 Ti is a Turing GPU without BF16 support, so vLLM falls back to FP32 and cannot co-host the GPU retail BERT router; its experimental GGUF route also crashes in vLLM EngineCore on this hardware. Keep `LLM_PROVIDER=vercel` on this machine. To run Gemma locally on the GTX 1650 Ti, use a Q4 GGUF model with llama.cpp instead of vLLM. Keep any server process alive; loading per turn destroys the latency benefit.

After the server reports ready, measure a warm end-to-end Hindi retail turn:

```bash
uv run python -m gnaniassignment.pipeline.check --runs 5 "मेरा ऑर्डर अभी तक डिलीवर नहीं हुआ"
```

The output separates the retail BERT routing time, Gemma first text chunk (TTFT), remaining stream time, and complete turn time. Only compare runs 2–5: the first request warms CUDA kernels and is not representative.

## Gnani realtime STT

Install the raw WebSocket client when you are ready to attach microphone or call audio:

```bash
uv sync --extra gnani-stt
```

Set `GNANI_API_KEY` in `.env`. The configured pipeline is `en-IN` at `16000` Hz: signed-16-bit little-endian mono PCM, exactly 1,024 bytes every 32 ms. The raw WebSocket adapter sends `x-vad-*` headers, which the official Python SDK does not expose. `GNANI_MIN_SILENCE_MS=100` is the fastest final-segment setting; raise it to `300`–`600` if callers are split at natural pauses. Do not burst buffered frames.

`GnaniRealtimeStt` yields a final `GnaniTranscript` for each VAD-closed segment. `GnaniTurnBridge.handle()` immediately starts Laya and the selected subagent by default—the lowest-latency path. Use `handle_async()` from the WebSocket receiver so the synchronous LangGraph work does not block audio/event I/O. For a workflow that intentionally combines segments, construct the bridge with `auto_end_turn=False`, then call `end_turn()` explicitly. Gnani's `processing` event is useful to update the UI (for example, “transcribing…”), but it is not text and is never sent to Laya.

Before sending telephony audio, verify that Gnani actually accepted the requested sample rate. This opens a connection but sends no PCM and appends the result to an ignored local log:

```bash
uv run python -m gnaniassignment.pipeline.gnani_stt_diagnostic
```

The expected result is `acknowledged_sample_rate: 16000`. Do not send audio at a different rate or cadence than the rate Gnani acknowledges.

For a live Linux microphone test (no audio file is saved), run:

```bash
uv run python -m gnaniassignment.pipeline.gnani_microphone --seconds 8 --device plughw:1,0
```

Speak while it records. The runner coalesces arbitrary ALSA reads into Gnani's required 1,024-byte frames and prints only final transcript segments and STT latency.

For an open, call-length connection, omit `--seconds` and end it with `Ctrl+C`:

```bash
uv run python -m gnaniassignment.pipeline.gnani_microphone --device plughw:1,0
```

Production telephony/WebRTC adapters should follow the same lifecycle: open one `GnaniRealtimeStt` context when the call begins, forward every incoming customer-audio frame through it, and close it only when the call ends. Each VAD-closed utterance is delivered as a separate `GnaniTranscript` on that persistent connection.

To run the full live chain—microphone → Gnani STT → retail intent classifier → streamed LLM—add `--with-llm`:

```bash
uv run python -m gnaniassignment.pipeline.gnani_microphone --device plughw:1,0 --with-llm
```

Every VAD-final segment triggers an answer. With the 100 ms latency-first VAD profile, keep utterances short and pause only when you want the assistant to answer.

## Reusable decision models

`gnaniassignment.decision` is the low-latency replacement for zero-shot routing when a voice workflow has a fixed taxonomy. Each decision point has a versioned catalog, a separately versioned labelled JSONL dataset, and one generated model bundle. The engine is reusable across flows; do not mix unrelated labels (for example, enquiry and call-control intents) into one classifier.

Each JSONL row requires `text` and `intent`; `language`, `source`, and `group_id` are strongly recommended. `group_id` keeps near-duplicate paraphrases in the same train/validation/test split. See `data/decision/enquiry-v1.example.jsonl` for the schema, but do not train production models from its tiny seed set. Collect roughly 300–500 genuine Hindi, Hinglish, English, and STT-noise examples per label, including a strong `other` class.

Train a model bundle after labelling data:

```bash
uv sync --extra decision-model
uv run python -m gnaniassignment.decision.train \
  --catalog configs/decision/enquiry-v1.json \
  --data data/decision/enquiry-v1.jsonl \
  --output artifacts/enquiry-v1.joblib \
  --threshold 0.65
```

The baseline combines word 1–2 grams with character 3–5 grams and uses multinomial logistic regression. It is designed for CPU inference in a few milliseconds and returns calibrated-enough probabilities for a clarification fallback. Evaluate against a held-out, group-separated test set before replacing the Laya route.

For a device-verified Laya-only measurement, run `uv run --extra local-laya python -m gnaniassignment.pipeline.laya_benchmark --runs 10`. It verifies that model parameters are on the same CUDA device as Laya and synchronizes CUDA before and after every prediction, so the reported warm latency is not an asynchronous-timing artifact. For the fused CUDA path, install `uv sync --extra local-laya-fast`, then benchmark `--backend tilelang`. This is the preferred acceleration experiment for the Turing GPU; keep `eager` unless TileLang loads successfully and wins the warm median. For a GTX 1650 Ti, set `TILELANG_DEFAULT_TARGET='{"kind":"cuda","arch":"sm_75"}'` if automatic target detection fails. Do not use `compile` for this latency-sensitive single-request path because its start-up compilation can be substantial.

```bash
uv run gnaniassignment
uv run python -m gnaniassignment.pipeline.check "What does the product do?"
```

The CLI uses placeholder knowledge and will say when company or product facts are unavailable. Pass approved knowledge or an intent-scoped retriever through `create_session()` in `app/container.py` to enable factual answers.

## Amazon Bedrock: Gemma 3 4B in Mumbai

The Bedrock provider uses `ConverseStream` with the direct Mumbai endpoint, not an APAC or global inference profile. Install its optional dependency and use a Bedrock API key (or, alternatively, the standard AWS SDK credential chain):

```bash
uv sync --extra retail-classifier --extra bedrock
```

```env
LLM_PROVIDER=bedrock
BEDROCK_REGION=ap-south-1
BEDROCK_MODEL=google.gemma-3-4b-it
BEDROCK_MAX_OUTPUT_TOKENS=32
BEDROCK_API_KEY=your_bedrock_api_key
```

Generate the key in the Bedrock console while Mumbai is selected. A short-term key is limited to your console session (up to 12 hours), while a long-term key should be used only for development. The key needs `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` permissions for this model. Then benchmark the real Mumbai path:

To use IAM credentials instead of an API key, choose the IAM mode. The SDK then uses a named AWS profile, AWS SSO, environment credentials, or an attached compute role. IAM access needs `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` for `google.gemma-3-4b-it`.

```bash
aws configure sso --profile bedrock-mumbai
aws sso login --profile bedrock-mumbai
```

```env
LLM_PROVIDER=bedrock
BEDROCK_AUTH_MODE=iam
AWS_PROFILE=bedrock-mumbai
BEDROCK_REGION=ap-south-1
BEDROCK_MODEL=google.gemma-3-4b-it
```

```bash
uv run python -m gnaniassignment.pipeline.check --runs 3 "मेरा ऑर्डर अभी तक डिलीवर नहीं हुआ"
```

## Cerebras Llama 3.3 70B

Cerebras exposes an OpenAI-compatible streaming endpoint. Set the key in the ignored `.env` and select the provider:

```env
LLM_PROVIDER=cerebras
CEREBRAS_API_KEY=your_key_here
CEREBRAS_MODEL=llama-3.3-70b
CEREBRAS_MAX_OUTPUT_TOKENS=32
```

No extra package is needed because the project already uses the OpenAI Python SDK. The pipeline reports Cerebras first-text TTFT through the same `pipeline.check` command. Gemini remains implemented as an inactive alternative; switch `LLM_PROVIDER` back to `vertex_gemini` to use it.

## Groq Llama 3.3 70B

Groq uses the same OpenAI-compatible client already present in the project. Configure it in the ignored `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MAX_OUTPUT_TOKENS=32
```

The public Groq API does not publish a customer-selectable India endpoint. Deploy the application gateway near your users, then benchmark actual Groq TTFT from that gateway before selecting it for production.

## Simplismart Llama 3.1 8B

Simplismart's shared endpoint is OpenAI-compatible and uses the following active configuration:

```env
LLM_PROVIDER=simplismart
SIMPLISMART_API_KEY=your_key_here
SIMPLISMART_BASE_URL=https://api.simplismart.live
SIMPLISMART_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
SIMPLISMART_MAX_OUTPUT_TOKENS=32
```

After adding the key, use `pipeline.check --runs 3` to measure cold and warm TTFT with the same intent-routed Hindi query.

## Gemini 2.5 Flash on Vertex AI

The pipeline can stream `gemini-2.5-flash` through the official Google Gen AI SDK. It implements the same `ChatModel` interface as Vercel, so TTFT and complete-stream timing remain visible in `pipeline.check`.

Install the provider alongside the active retail classifier:

```bash
uv sync --extra retail-classifier --extra gemini
```

Choose exactly one authentication mode in `.env`:

```env
# Vertex AI Express Mode: API-key billing, Google selects request routing.
LLM_PROVIDER=vertex_gemini
GEMINI_AUTH_MODE=vertex_api_key
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

```env
# Standard regional Vertex AI: use this when Mumbai residency is required.
LLM_PROVIDER=vertex_gemini
GEMINI_AUTH_MODE=vertex_adc
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=asia-south1
# Authenticate with Application Default Credentials or GOOGLE_APPLICATION_CREDENTIALS.
```

`asia-south1` is the Mumbai Vertex AI region and supports Gemini 2.5 Flash. API-key Express Mode and regional Vertex AI are different Google Cloud products: an API key alone does not provide a regional-residency guarantee. Use `vertex_adc` with `GOOGLE_CLOUD_LOCATION=asia-south1` for a Mumbai-pinned serverless endpoint. Keep the API key and service-account credentials out of Git; `.env` is ignored.

For the voice response path, `GEMINI_THINKING_BUDGET=0` is the default. Gemini 2.5 Flash otherwise allocates an automatic thinking budget before producing the short JSON answer, increasing TTFT and potentially exhausting a small response-token cap.

After credentials are configured, run:

```bash
uv run python -m gnaniassignment.pipeline.check --runs 3 "मेरा ऑर्डर अभी तक डिलीवर नहीं हुआ"
```

The CLI and `pipeline.check` stream only the spoken answer, not the internal JSON summary. For application code, call `session.ask_stream(query, on_chunk)` or `session.feed_end(on_chunk=on_chunk)` after final STT phrases. The callback receives decoded answer fragments; the returned result and session history are complete only after the stream ends. `pipeline.check` reports Vercel Gateway TTFT from the subagent request, plus time from turn start to the first answer text.

To compare cold and reused-client latency, run `uv run python -m gnaniassignment.pipeline.check --runs 3 "What does the product do?"`. Each run starts with fresh conversation history but reuses the same Laya router and Vercel Gateway client. The report separates Laya classification, Vercel time to first text chunk and stream completion, and remaining local pipeline work. In hosted mode it also reports time to Laya response headers.

To compare Vercel's automatic provider choice with fixed Bedrock and Novita routes, run `uv run python -m gnaniassignment.pipeline.provider_benchmark --runs 2`. This diagnostic uses the same product-enquiry prompt for each route and reports time to response headers, first stream event, first text, the gateway-reported provider attempt (when available), and completion. It does not change production routing. Gateway calls may incur usage charges.

## LLM flow

1. `TurnSession` collects final STT phrases and waits for end of turn. Text callers can use `ask(query)`.
2. Laya receives structured state with the current query and at most two compressed recent turns through Vercel's TypeSafe-compatible decision endpoint. One `choice` question selects among six intents (`user_core`, `user_operations`, `user_timeline`, `product_what`, `product_why`, `product_when`) and `other`.
3. If Laya selects `other` or its Choice confidence is below `INTENT_THRESHOLD` (default `0.5`), the graph asks a clarification without a chat-model call. Tune this threshold using labeled queries from the actual use case.
4. The selected agent gets a maximum of four relevant memory summaries and up to four approved retrieval chunks from its own intent collection. Its single chat completion produces both the spoken answer and the new summary.

Conversation history is in memory per `TurnSession` and is capped at 100 turns. There is no persistent cross-session memory or concrete vector database adapter yet. The `IntentScopedRetriever` interface accepts one store per intent and can be backed by collections or namespaces in a vector database.

## Test

```bash
uv run python -m unittest discover -s tests -v
```
