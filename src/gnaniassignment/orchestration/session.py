"""Session with bounded, intent-aware episodic memory."""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from gnaniassignment.orchestration.episodic_memory import EpisodicMemoryStore
from gnaniassignment.llm.memory import ChatTurn
from gnaniassignment.orchestration.working_memory import update_working_memory


MAX_HISTORY_TURNS = 100


class TurnSession:
    def __init__(self, graph: Any, memory_store: EpisodicMemoryStore | None = None, customer_id: str | None = None) -> None:
        self.graph: Any = graph
        self.phrases: list[str] = []
        self.memory_store = memory_store
        self.customer_id = customer_id
        self.session_id = str(uuid4())
        self.history: list[ChatTurn] = memory_store.load_recent(customer_id, MAX_HISTORY_TURNS) if memory_store and customer_id else []
        self.working_memory: dict[str, str] = {}

    def feed_final(self, phrase: str) -> dict[str, Any]:
        if phrase.strip():
            self.phrases.append(phrase.strip())
        return {"spoken": None, "pending_text": " ".join(self.phrases)}

    def feed_end(self, on_chunk: Callable[[str], None] | None = None) -> dict[str, Any]:
        if not self.phrases:
            return {"spoken": None, "replies": []}
        text = " ".join(self.phrases)
        config = {"configurable": {"on_answer_chunk": on_chunk}} if on_chunk is not None else None
        result = self.graph.invoke({"text": text, "answers": {}, "intent": "", "topic": "", "history": self.history, "working_memory": self.working_memory, "memory": "", "replies": [], "spoken": None, "metrics": {}}, config=config)
        self.working_memory = result["working_memory"]
        reply_data = result["replies"][0]
        reply = reply_data["text"]
        turn = ChatTurn(query=text, answer=reply, summary=reply_data["summary"], intent=result["intent"], topic=result["topic"])
        self.history.append(turn)
        if len(self.history) > MAX_HISTORY_TURNS:
            del self.history[:-MAX_HISTORY_TURNS]
        if self.memory_store is not None and self.customer_id is not None:
            self.memory_store.append(self.customer_id, self.session_id, turn)
        self.phrases = []
        return result

    def ask(self, query: str) -> dict[str, Any]:
        """Convenience entry point for one complete text-chat query."""
        self.feed_final(query)
        return self.feed_end()

    def ask_stream(self, query: str, on_chunk: Callable[[str], None]) -> dict[str, Any]:
        """Deliver answer text incrementally; return the completed turn."""
        self.feed_final(query)
        return self.feed_end(on_chunk=on_chunk)
