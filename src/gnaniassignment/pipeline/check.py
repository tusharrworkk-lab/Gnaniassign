"""Time a complete text enquiry through Jev and the selected subagent."""

from __future__ import annotations

import sys
import time

from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.chat import OpenRouterChatModel
from gnaniassignment.llm.router import JevRouter
from gnaniassignment.orchestration.graph import build_graph
from gnaniassignment.orchestration.session import TurnSession


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or "Where do you operate?"
    router = JevRouter()
    session = TurnSession(build_graph(router, enquiry_agents(OpenRouterChatModel())))
    started = time.perf_counter()
    result = session.ask(query)
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"intent: {result['intent']}  topic: {result['topic']}")
    print(f"Jev: {router.last_ms:.0f} ms  complete turn: {elapsed_ms:.0f} ms")
    print(f"answer: {result['spoken']}")
    print(f"memory: {session.history[-1].summary}")


if __name__ == "__main__":
    main()
