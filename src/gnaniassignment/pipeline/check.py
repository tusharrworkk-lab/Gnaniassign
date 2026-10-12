"""Time one simulated call. Jev is the decision model. Speech is already final."""

from __future__ import annotations

import time

from gnaniassignment.pipeline.agents import default_agents
from gnaniassignment.pipeline.graph import INTENT_KEYS, build_graph
from gnaniassignment.pipeline.jev import JevRouter
from gnaniassignment.pipeline.session import TurnSession


def _show(label: str, result: dict, elapsed_ms: float, jev_ms: float) -> None:
    draft = result.get("draft") or {}
    answers = result.get("answers") or {}
    print(f"\n{label}  graph {elapsed_ms:.0f} ms  jev {jev_ms:.0f} ms")
    print(f"  text: {draft.get('text')}")
    for name in INTENT_KEYS:
        if name in answers:
            print(f"  {name}: {float(answers[name]['noul']):.3f}")
    print(f"  intents: {draft.get('intents')}  slots: {draft.get('slots')}")
    print(f"  reused: {result.get('reused')}")
    for reply in result.get("replies") or []:
        print(f"  {reply['agent']}: {reply['text']}")
    print(f"  spoken: {result.get('spoken')}")


def _step(session: TurnSession, router: JevRouter, label: str, call) -> float:
    started = time.perf_counter()
    result = call()
    elapsed_ms = (time.perf_counter() - started) * 1000
    _show(label, result, elapsed_ms, router.last_ms)
    return elapsed_ms


def main() -> None:
    router = JevRouter()
    session = TurnSession(build_graph(router, default_agents()))
    total = 0.0
    total += _step(session, router, "transcript 1", lambda: session.feed_final("book a meeting"))
    total += _step(
        session,
        router,
        "transcript 2",
        lambda: session.feed_final("on Thursday with Priya"),
    )
    total += _step(session, router, "pause", session.feed_end)
    print(f"\ncall total {total:.0f} ms")


if __name__ == "__main__":
    main()
