# Gnani Assignment

An intent-routed enquiry assistant. The current implementation accepts text or completed STT phrases, classifies the query with Laya through Vercel AI Gateway, and uses one focused small-model agent to produce a short answer and a compressed memory summary. Audio STT and TTS adapters are reserved for later integration.

## Run

Python 3.12 or newer is required. Install with `uv sync`, then set `AI_GATEWAY_API_KEY` for Laya and `OPENROUTER_API_KEY` for the response subagents in your environment. The subagents use `meta-llama/llama-3.1-8b-instruct` through OpenRouter by default. `SUBAGENT_MODEL` optionally overrides that model ID.

```bash
uv run gnaniassignment
uv run python -m gnaniassignment.pipeline.check "What does the product do?"
```

The CLI uses placeholder knowledge and will say when company or product facts are unavailable. Pass approved knowledge or an intent-scoped retriever through `create_session()` in `app/container.py` to enable factual answers.

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
