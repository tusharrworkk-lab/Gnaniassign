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

## Gnani realtime STT

Install the official streaming client when you are ready to attach microphone or call audio:

```bash
uv sync --extra gnani-stt
```

Set `GNANI_API_KEY` in `.env`. `GNANI_STT_LANGUAGE` defaults to `en-IN`; use `GNANI_STT_SAMPLE_RATE=8000` for narrowband telephony or `16000` for wideband audio. The adapter accepts only 1,024-byte signed-16-bit little-endian mono PCM frames, matching Gnani's realtime protocol. Send them at their natural realtime cadence rather than in bursts.

`GnaniRealtimeStt` yields a final `GnaniTranscript` for each VAD-closed segment. `GnaniTurnBridge.handle()` immediately starts Laya and the selected subagent by default—the lowest-latency path. Use `handle_async()` from the WebSocket receiver so the synchronous LangGraph work does not block audio/event I/O. For a workflow that intentionally combines segments, construct the bridge with `auto_end_turn=False`, then call `end_turn()` explicitly. Gnani's `processing` event is useful to update the UI (for example, “transcribing…”), but it is not text and is never sent to Laya.

For a device-verified Laya-only measurement, run `uv run --extra local-laya python -m gnaniassignment.pipeline.laya_benchmark --runs 10`. It verifies that model parameters are on the same CUDA device as Laya and synchronizes CUDA before and after every prediction, so the reported warm latency is not an asynchronous-timing artifact. For the fused CUDA path, install `uv sync --extra local-laya-fast`, then benchmark `--backend tilelang`. This is the preferred acceleration experiment for the Turing GPU; keep `eager` unless TileLang loads successfully and wins the warm median. For a GTX 1650 Ti, set `TILELANG_DEFAULT_TARGET='{"kind":"cuda","arch":"sm_75"}'` if automatic target detection fails. Do not use `compile` for this latency-sensitive single-request path because its start-up compilation can be substantial.

```bash
uv run gnaniassignment
uv run python -m gnaniassignment.pipeline.check "What does the product do?"
```

The CLI uses placeholder knowledge and will say when company or product facts are unavailable. Pass approved knowledge or an intent-scoped retriever through `create_session()` in `app/container.py` to enable factual answers.

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
