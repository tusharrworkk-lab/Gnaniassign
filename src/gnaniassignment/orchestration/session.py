"""Session with bounded, intent-aware episodic memory."""

from __future__ import annotations

from typing import Any

from gnaniassignment.llm.memory import ChatTurn


MAX_HISTORY_TURNS = 100


class TurnSession:
    def __init__(self, graph: Any) -> None:
        self.graph: Any = graph
        self.phrases: list[str] = []
        self.history: list[ChatTurn] = []

    def feed_final(self, phrase: str) -> dict[str, Any]:
        if phrase.strip():
            self.phrases.append(phrase.strip())
        return {"spoken": None, "pending_text": " ".join(self.phrases)}

    def feed_end(self) -> dict[str, Any]:
        if not self.phrases:
            return {"spoken": None, "replies": []}
        text = " ".join(self.phrases)
        result = self.graph.invoke({"text": text, "answers": {}, "intent": "", "topic": "", "history": self.history, "memory": "", "replies": [], "spoken": None})
        reply_data = result["replies"][0]
        reply = reply_data["text"]
        self.history.append(ChatTurn(query=text, answer=reply, summary=reply_data["summary"], intent=result["intent"], topic=result["topic"]))
        if len(self.history) > MAX_HISTORY_TURNS:
            del self.history[:-MAX_HISTORY_TURNS]
        self.phrases = []
        return result

    def ask(self, query: str) -> dict[str, Any]:
        """Convenience entry point for one complete text-chat query."""
        self.feed_final(query)
        return self.feed_end()
