# Gnani Assignment

An intent-routed enquiry assistant. The current implementation accepts text or completed STT phrases, classifies the query with Jev through OpenRouter, and uses one focused small-model agent to produce a short answer and a compressed memory summary. Audio STT and TTS adapters are reserved for later integration.

## Run

Python 3.12 or newer is required. Install with `uv sync`, then set `OPENROUTER_API_KEY` and `SUBAGENT_MODEL` in your environment. `SUBAGENT_MODEL` must be an OpenRouter chat model ID available to your account.

```bash
uv run gnaniassignment
uv run python -m gnaniassignment.pipeline.check "What does the product do?"
```

The CLI uses placeholder knowledge and will say when company or product facts are unavailable. Pass approved knowledge or an intent-scoped retriever through `create_session()` in `app/container.py` to enable factual answers.

## LLM flow

1. `TurnSession` collects final STT phrases and waits for end of turn. Text callers can use `ask(query)`.
2. Jev receives the current query plus at most two compressed recent turns and selects one of six intents: `user_core`, `user_operations`, `user_timeline`, `product_what`, `product_why`, or `product_when`.
3. If the strongest intent score is below `INTENT_THRESHOLD` (default `0.5`), the graph asks a clarification without a chat-model call.
4. The selected agent gets a maximum of four relevant memory summaries and up to four approved retrieval chunks from its own intent collection. Its single chat completion produces both the spoken answer and the new summary.

Conversation history is in memory per `TurnSession` and is capped at 100 turns. There is no persistent cross-session memory or concrete vector database adapter yet. The `IntentScopedRetriever` interface accepts one store per intent and can be backed by collections or namespaces in a vector database.

## Test

```bash
uv run python -m unittest discover -s tests -v
```
