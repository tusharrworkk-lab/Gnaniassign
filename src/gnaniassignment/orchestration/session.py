"""Session with bounded, intent-aware episodic memory."""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from gnaniassignment.orchestration.episodic_memory import EpisodicMemoryStore
from gnaniassignment.llm.memory import ChatTurn
from gnaniassignment.orchestration.working_memory import update_working_memory
from gnaniassignment.conversation_analytics import AcilConversationTracer, ConversationJudge, ConversationRecord, PostgresConversationStore, flow_flags


MAX_HISTORY_TURNS = 100
MAX_PRIOR_CALL_SUMMARIES = 3
MAX_CALL_SUMMARY_CHARS = 900


class TurnSession:
    def __init__(self, graph: Any, memory_store: EpisodicMemoryStore | None = None, customer_id: str | None = None, conversation_store: PostgresConversationStore | None = None, tracer: AcilConversationTracer | None = None, call_judge: ConversationJudge | None = None, account_context_provider: Callable[[str, dict[str, str]], str] | None = None, session_id: str | None = None) -> None:
        self.graph: Any = graph
        self.phrases: list[str] = []
        self.memory_store = memory_store
        self.customer_id = customer_id
        # Telephony adapters provide a durable provider stream ID. Reuse it so
        # call metrics, transcripts, reviews and dashboard detail all share a
        # single, queryable call identity.
        self.session_id = session_id or str(uuid4())
        # Raw turns are retained in PostgreSQL for audit, but never replayed
        # wholesale to a new call.  A fresh call starts with a few concise,
        # completed-call summaries only.
        if memory_store and customer_id:
            summaries = getattr(memory_store, "load_call_summaries", None)
            self.prior_call_context = summaries(customer_id, MAX_PRIOR_CALL_SUMMARIES) if summaries else memory_store.load_recent(customer_id, MAX_HISTORY_TURNS)
        else:
            self.prior_call_context = []
        self.history: list[ChatTurn] = list(self.prior_call_context)
        self.working_memory: dict[str, str] = {}
        self.conversation_store = conversation_store
        self.tracer = tracer
        self.call_judge = call_judge
        self.call_turns: list[ChatTurn] = []
        self.stt_segments: list[dict[str, object]] = []
        self.account_context_provider = account_context_provider
        self.account_context = account_context_provider("", self.working_memory) if account_context_provider is not None else "No verified account context is available."

    def record_stt_segment(self, text: str, segment_id: str | None, latency_ms: int | None) -> None:
        """Keep the exact final STT segments for the end-of-call transcript."""
        if text.strip():
            self.stt_segments.append({"text": text.strip(), "segment_id": segment_id, "latency_ms": latency_ms})

    def feed_final(self, phrase: str) -> dict[str, Any]:
        if phrase.strip():
            clean_phrase = phrase.strip()
            if not self.phrases and self.conversation_store is not None:
                flags = self.conversation_store.attach_followup(self.session_id, clean_phrase)
                if self.tracer is not None:
                    self.tracer.emit_followup(self.session_id, clean_phrase, flags)
            self.phrases.append(clean_phrase)
        return {"spoken": None, "pending_text": " ".join(self.phrases)}

    def feed_end(self, on_chunk: Callable[[str], None] | None = None) -> dict[str, Any]:
        if not self.phrases:
            return {"spoken": None, "replies": []}
        text = " ".join(self.phrases)
        if self.account_context_provider is not None:
            self.account_context = self.account_context_provider(text, self.working_memory)
        config = {"configurable": {"on_answer_chunk": on_chunk}} if on_chunk is not None else None
        result = self.graph.invoke({"session_id": self.session_id, "text": text, "answers": {}, "intent": "", "topic": "", "history": self.history, "prior_call_context": self.prior_call_context, "current_call_history": self.call_turns, "working_memory": self.working_memory, "account_context": self.account_context, "memory": "", "replies": [], "spoken": None, "metrics": {}}, config=config)
        self.working_memory = result["working_memory"]
        reply_data = result["replies"][0]
        reply = reply_data["text"]
        turn = ChatTurn(query=text, answer=reply, summary=reply_data["summary"], intent=result["intent"], topic=result["topic"])
        self.history.append(turn)
        self.call_turns.append(turn)
        if len(self.history) > MAX_HISTORY_TURNS:
            del self.history[:-MAX_HISTORY_TURNS]
        if self.memory_store is not None and self.customer_id is not None:
            self.memory_store.append(self.customer_id, self.session_id, turn)
        if self.conversation_store is not None:
            decision = result.get("answers", {}).get("intent", {})
            classifier_intent = decision.get("choice") if isinstance(decision.get("choice"), str) and decision.get("choice") != "generic" else None
            classifier_confidence = float(decision["confidence"]) if classifier_intent and isinstance(decision.get("confidence"), (int, float)) else None
            record = ConversationRecord(
                event_id=str(uuid4()), session_id=self.session_id, turn_number=len(self.history),
                customer_id=self.customer_id, user_text=text, response_text=reply,
                routed_intent=result["intent"], classifier_intent=classifier_intent,
                classifier_confidence=classifier_confidence, working_memory=dict(self.working_memory),
                flow_flags=flow_flags(reply, self.working_memory),
            )
            self.conversation_store.append(record)
            if self.tracer is not None:
                self.tracer.emit_turn(record)
        self.phrases = []
        return result

    def finalize_call(self) -> dict[str, object] | None:
        """Persist and judge one complete call after microphone capture has ended."""
        if not self.call_turns:
            return None
        self._persist_call_summary()
        if self.conversation_store is None:
            return None
        transcript = [
            {"turn_number": index, "user": turn.query, "assistant": turn.answer, "routed_intent": turn.intent}
            for index, turn in enumerate(self.call_turns, start=1)
        ]
        try:
            judge_result = self.call_judge.judge(self.stt_segments, transcript) if self.call_judge is not None else None
        except Exception as error:
            judge_result = {"judge_error": type(error).__name__, "message": str(error)[:500]}
        self.conversation_store.finalize_call(self.session_id, self.customer_id, self.stt_segments, transcript, judge_result)
        if self.tracer is not None and judge_result is not None:
            self.tracer.emit_call_judgment(self.session_id, judge_result)
        return judge_result

    def _persist_call_summary(self) -> None:
        """Store a small deterministic hand-off summary for the next call.

        No extra model call is made after the phone call; per-turn summaries
        already produced by the agent are compacted so finalization cannot add
        voice latency or background LLM cost.
        """
        if self.memory_store is None or self.customer_id is None:
            return
        append_summary = getattr(self.memory_store, "append_call_summary", None)
        if append_summary is None:
            return
        lines: list[str] = []
        seen: set[str] = set()
        for turn in self.call_turns[-3:]:
            item = turn.summary.strip()
            if item and item not in seen:
                lines.append(item)
                seen.add(item)
        if not lines:
            return
        summary = "Previous call: " + " ".join(lines)
        append_summary(
            self.customer_id,
            self.session_id,
            summary[:MAX_CALL_SUMMARY_CHARS],
            self.call_turns[-1].intent,
            self.call_turns[-1].topic,
        )

    def ask(self, query: str) -> dict[str, Any]:
        """Convenience entry point for one complete text-chat query."""
        self.feed_final(query)
        return self.feed_end()

    def ask_stream(self, query: str, on_chunk: Callable[[str], None]) -> dict[str, Any]:
        """Deliver answer text incrementally; return the completed turn."""
        self.feed_final(query)
        return self.feed_end(on_chunk=on_chunk)
