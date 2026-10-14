"""Time a complete text enquiry through Laya and the selected subagent."""

from __future__ import annotations

import argparse
import time

from gnaniassignment.app.config import Settings, load_environment
from gnaniassignment.app.container import create_intent_router
from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.chat import VercelGatewayChatModel
from gnaniassignment.llm.local_laya import LocalLayaRouter
from gnaniassignment.llm.router import LayaGatewayRouter
from gnaniassignment.orchestration.graph import build_graph
from gnaniassignment.orchestration.session import TurnSession


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one or more streamed enquiry turns")
    parser.add_argument("query", nargs="*", help="Question to send through the pipeline")
    parser.add_argument("--runs", type=int, default=1, help="Repeat with fresh conversation history and reused API clients")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    load_environment()
    query = " ".join(args.query).strip() or "Where do you operate?"
    router = create_intent_router(Settings.from_environment())
    if isinstance(router, LocalLayaRouter):
        print(f"Local Laya load: {router.load_ms:.0f} ms (outside turn timing)")
    model = VercelGatewayChatModel()
    graph = build_graph(router, enquiry_agents(model))
    for run in range(args.runs):
        if args.runs > 1:
            print(f"run {run + 1}/{args.runs}")
        session = TurnSession(graph)
        _run_turn(session, router, model, query)


def _run_turn(session: TurnSession, router: LayaGatewayRouter | LocalLayaRouter, model: VercelGatewayChatModel, query: str) -> None:
    model.last_first_token_ms = None
    model.last_stream_ms = None
    started = time.perf_counter()
    first_answer_ms: float | None = None

    def show_chunk(chunk: str) -> None:
        nonlocal first_answer_ms
        if first_answer_ms is None:
            first_answer_ms = (time.perf_counter() - started) * 1000
        print(chunk, end="", flush=True)

    print("answer: ", end="", flush=True)
    result = session.ask_stream(query, show_chunk)
    print()
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"intent: {result['intent']}  topic: {result['topic']}")
    print(f"Laya {'local GPU' if isinstance(router, LocalLayaRouter) else 'gateway'}: {router.last_ms:.0f} ms")
    if isinstance(router, LayaGatewayRouter) and router.last_headers_ms is not None:
        print(f"  until response headers: {router.last_headers_ms:.0f} ms")
        print(f"  body + JSON validation: {max(0, router.last_ms - router.last_headers_ms):.0f} ms")
    if model.last_first_token_ms is not None:
        print(f"Vercel first text chunk: {model.last_first_token_ms:.0f} ms")
    if model.last_stream_ms is not None:
        if model.last_first_token_ms is not None:
            print(f"Vercel rest of stream: {model.last_stream_ms - model.last_first_token_ms:.0f} ms")
        print(f"Vercel total: {model.last_stream_ms:.0f} ms")
        print(f"other pipeline work: {max(0, elapsed_ms - router.last_ms - model.last_stream_ms):.0f} ms")
    if first_answer_ms is not None:
        print(f"first answer text: {first_answer_ms:.0f} ms")
    print(f"complete turn: {elapsed_ms:.0f} ms")
    print(f"memory: {session.history[-1].summary}")


if __name__ == "__main__":
    main()
