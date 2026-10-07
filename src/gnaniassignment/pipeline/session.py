"""Feeds locked speech-to-text phrases into the graph.

Partials are ignored. Each final phrase is appended and the graph runs on the
whole sentence so far. The pause at the end of the turn is the only time a
reply is spoken.
"""

from __future__ import annotations

from typing import Any


class TurnSession:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.phrases: list[str] = []
        self.draft: dict[str, Any] | None = None

    def feed_final(self, phrase: str) -> dict[str, Any]:
        self.phrases.append(phrase.strip())
        return self._invoke("final")

    def feed_end(self) -> dict[str, Any]:
        if not self.phrases:
            return {"spoken": None, "draft": self.draft, "reused": False, "replies": []}
        result = self._invoke("end")
        self.phrases = []
        self.draft = None
        return result

    def _invoke(self, event: str) -> dict[str, Any]:
        result = self.graph.invoke(
            {
                "text": " ".join(self.phrases),
                "event": event,
                "draft": self.draft,
                "answers": {},
                "intents": [],
                "slots": {},
                "reused": False,
                "replies": [],
                "spoken": None,
            }
        )
        self.draft = result["draft"]
        return result
