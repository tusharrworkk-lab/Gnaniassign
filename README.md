# Gnani Assignment

An intent-routed enquiry assistant. The current implementation accepts text or completed STT phrases, classifies the query with Laya through Vercel AI Gateway, and uses one focused small-model agent to produce a short answer and a compressed memory summary. Audio STT and TTS adapters are reserved for later integration.

## Run

Python 3.12 or newer is required. Install with `uv sync --extra local-laya`, then fill in `AI_GATEWAY_API_KEY` in the root `.env` file. The answer subagents use `meta/llama-3.1-8b` through Vercel AI Gateway by default; `SUBAGENT_MODEL` optionally overrides that model ID, and `SUBAGENT_GATEWAY_URL` optionally overrides the OpenAI-compatible endpoint. The app loads `.env` automatically and does not override variables already set in your shell. `LAYA_GATEWAY_URL` and `LAYA_MODEL` configure the hosted Laya endpoint and model. `.env` is ignored by Git; `.env.example` records the expected variable names.

To run intent classification locally on an NVIDIA GPU, install `uv sync --extra local-laya` and set `INTENT_PROVIDER=local_laya`, `LOCAL_LAYA_DEVICE=cuda`, and `LOCAL_LAYA_BACKEND=eager` in `.env`. The first startup downloads the public `convaiinnovations/laya` checkpoint. Startup also warms the model; later sessions in the same process reuse it. CUDA is fail-fast: the app refuses Laya's silent CPU fallback when CUDA was requested. Run `uv run python -m gnaniassignment.pipeline.laya_diagnostic` before startup; `nvidia-smi` and that diagnostic must report a visible GPU. `INTENT_PROVIDER=laya` keeps the hosted Vercel route. The Vercel answer subagent remains hosted in either mode.

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
