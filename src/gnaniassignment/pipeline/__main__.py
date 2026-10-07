"""Two locked phrases, then the pause. Uses the local Laya checkpoint."""

from __future__ import annotations

from gnaniassignment.pipeline.agents import default_agents
from gnaniassignment.pipeline.graph import build_graph, load_laya
from gnaniassignment.pipeline.session import TurnSession


def _show(label: str, result: dict) -> None:
    draft = result.get("draft") or {}
    print(
        f"{label}: text={draft.get('text')!r} "
        f"intents={draft.get('intents')} slots={draft.get('slots')} "
        f"reused={result.get('reused')} spoken={result.get('spoken')!r}"
    )


def main() -> None:
    agents = default_agents()
    session = TurnSession(build_graph(load_laya(), agents))

    _show("final 1", session.feed_final("book a meeting"))
    _show("final 2", session.feed_final("on Thursday"))
    _show("pause", session.feed_end())
    print(f"meeting agent calls: {agents['book_meeting'].calls}")


if __name__ == "__main__":
    main()
