"""Session with bounded, intent-aware episodic memory."""

from __future__ import annotations

import logging
from typing import Any, Callable
from uuid import uuid4

from gnaniassignment.orchestration.episodic_memory import EpisodicMemoryStore
from gnaniassignment.llm.memory import ChatTurn
from gnaniassignment.orchestration.working_memory import update_working_memory
from gnaniassignment.conversation_analytics import AcilConversationTracer, ConversationJudge, ConversationRecord, PostgresConversationStore, flow_flags


MAX_HISTORY_TURNS = 100
MAX_PRIOR_CALL_SUMMARIES = 3
MAX_CALL_SUMMARY_CHARS = 900
MAX_CALL_MEMO_CHARS = 320
logger = logging.getLogger(__name__)


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
        # Generated answers remain in the audit transcript, but only audio
        # handed to the caller may become assistant context for another turn.
        self.call_context_turns: list[ChatTurn] = []
        self.delivery_statuses: list[str] = []
        self.handed_off_answers: list[str] = []
        self.call_memo = ""
        self._deferred_records: dict[int, ConversationRecord | None] = {}
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

    def feed_end(self, on_chunk: Callable[[str], None] | None = None, *, defer_delivery: bool = False) -> dict[str, Any]:
        if not self.phrases:
            return {"spoken": None, "replies": []}
        text = " ".join(self.phrases)
        self.phrases = []
        try:
            if self.account_context_provider is not None:
                self.account_context = self.account_context_provider(text, self.working_memory)
            config = {"configurable": {"on_answer_chunk": on_chunk}} if on_chunk is not None else None
            result = self.graph.invoke({"session_id": self.session_id, "text": text, "answers": {}, "intent": "", "topic": "", "history": self.history, "prior_call_context": self.prior_call_context, "current_call_history": self.call_context_turns, "call_memo": self.call_memo or "No call memo yet.", "working_memory": self.working_memory, "account_context": self.account_context, "memory": "", "replies": [], "spoken": None, "metrics": {}}, config=config)
        except Exception:
            if not defer_delivery:
                self.phrases = [text]
            raise
        self.working_memory = result["working_memory"]
        reply_data = result["replies"][0]
        reply = reply_data["text"]
        turn = ChatTurn(query=text, answer=reply, summary=reply_data["summary"], intent=result["intent"], topic=result["topic"])
        self.history.append(turn)
        self.call_turns.append(turn)
        self.delivery_statuses.append("generated")
        self.handed_off_answers.append("")
        if len(self.history) > MAX_HISTORY_TURNS:
            del self.history[:-MAX_HISTORY_TURNS]
        record: ConversationRecord | None = None
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
        if defer_delivery:
            # The voice worker persists this only after the audio result is
            # known, keeping PostgreSQL writes off the first-audio path.
            self._deferred_records[len(self.call_turns) - 1] = record
        else:
            self._persist_turn(turn, record)
        if not defer_delivery:
            self.mark_delivery("delivered")
        return result

    def _persist_turn(self, turn: ChatTurn, record: ConversationRecord | None) -> None:
        if self.memory_store is not None and self.customer_id is not None:
            self.memory_store.append(self.customer_id, self.session_id, turn)
        if record is not None and self.conversation_store is not None:
            self.conversation_store.append(record)
            if self.tracer is not None:
                self.tracer.emit_turn(record)

    def mark_delivery(self, status: str, *, handed_off_text: str = "") -> bool:
        """Set the latest generated turn's delivery outcome for prompt memory."""
        if status not in {"delivered", "audio_sent", "interrupted", "failed"}:
            raise ValueError("Unsupported delivery status")
        pending = next((index for index in range(len(self.delivery_statuses) - 1, -1, -1)
                        if self.delivery_statuses[index] == "generated"), None)
        if pending is None:
            return False
        turn = self.call_turns[pending]
        self.delivery_statuses[pending] = status
        if status in {"delivered", "audio_sent"}:
            self.handed_off_answers[pending] = turn.answer
            self.call_context_turns.append(turn)
        elif handed_off_text.strip():
            spoken = handed_off_text.strip()
            self.handed_off_answers[pending] = spoken
            self.call_context_turns.append(ChatTurn(
                query=turn.query,
                answer=spoken,
                summary=(
                    f"Caller said: {turn.query[:220]}. Assistant audio handed off before {status}: "
                    f"{spoken[:180]}. The rest of the generated reply was not handed off."
                ),
                intent=turn.intent,
                topic=turn.topic,
            ))
        else:
            self.record_unanswered_request(turn.query, status=status, intent=turn.intent, topic=turn.topic)
        self._update_call_memo(turn, status)
        if pending in self._deferred_records:
            record = self._deferred_records.pop(pending)
            try:
                self._persist_turn(turn, record)
            except Exception:
                # A storage outage must not make an already-sent voice answer
                # disappear from this call's in-memory conversation context.
                logger.exception("Could not persist voice turn for session %s", self.session_id)
        return True

    def _update_call_memo(self, turn: ChatTurn, status: str) -> None:
        """Replace the rolling semantic state using the current model response."""
        memo = " ".join(turn.summary.split())
        if not memo:
            memo = f"Caller wants: {' '.join(turn.query.split())[:240]}"
        if status in {"failed", "interrupted"}:
            # Plain-text fallback summaries contain the generated answer. It
            # must not enter memory when that answer was not fully delivered.
            memo = memo.partition(". Answer:")[0]
            memo += " Assistant reply delivery was incomplete."
        self.call_memo = memo[:MAX_CALL_MEMO_CHARS]

    def record_unanswered_request(self, text: str, *, status: str = "failed", intent: str = "generic", topic: str = "general") -> None:
        """Keep the caller's request while avoiding an unheard assistant reply."""
        self.call_context_turns.append(ChatTurn(
            query=text, answer="",
            summary=f"Caller said: {text[:220]}. Assistant reply was {status}; do not assume the caller heard it.",
            intent=intent, topic=topic,
        ))

    def finalize_call(self) -> dict[str, object] | None:
        """Persist and judge one complete call after microphone capture has ended."""
        if not self.call_turns:
            return None
        self._persist_call_summary()
        if self.conversation_store is None:
            return None
        transcript = [
            {"turn_number": index, "user": turn.query,
             "assistant": self.handed_off_answers[index - 1],
             "generated_assistant": turn.answer,
             "routed_intent": turn.intent, "summary": turn.summary,
             "delivery_status": self.delivery_statuses[index - 1]}
            for index, turn in enumerate(self.call_turns, start=1)
        ]
        # The flow judge must evaluate the conversation the caller could hear,
        # while the audit record retains the generated text separately.
        judge_transcript = [{key: value for key, value in turn.items() if key not in {"generated_assistant", "summary"}}
                            for turn in transcript]
        try:
            judge_result = self.call_judge.judge(self.stt_segments, judge_transcript) if self.call_judge is not None else None
        except Exception as error:
            judge_result = {"judge_error": type(error).__name__, "message": str(error)[:500]}
        self.conversation_store.finalize_call(self.session_id, self.customer_id, self.stt_segments, transcript, judge_result)
        if self.tracer is not None and judge_result is not None:
            self.tracer.emit_call_judgment(self.session_id, judge_result)
        return judge_result

    def _persist_call_summary(self) -> None:
        """Store a small deterministic hand-off summary for the next call.

        No extra model call is made after the phone call. The latest rolling
        memo already represents the complete call state.
        """
        if self.memory_store is None or self.customer_id is None:
            return
        append_summary = getattr(self.memory_store, "append_call_summary", None)
        if append_summary is None:
            return
        if not self.call_memo or not self.call_context_turns:
            return
        summary = "Previous call: " + self.call_memo
        append_summary(
            self.customer_id,
            self.session_id,
            summary[:MAX_CALL_SUMMARY_CHARS],
            self.call_context_turns[-1].intent,
            self.call_context_turns[-1].topic,
        )

    def ask(self, query: str) -> dict[str, Any]:
        """Convenience entry point for one complete text-chat query."""
        self.feed_final(query)
        return self.feed_end()

    def ask_stream(self, query: str, on_chunk: Callable[[str], None], *, defer_delivery: bool = False) -> dict[str, Any]:
        """Deliver answer text incrementally; return the completed turn."""
        self.feed_final(query)
        return self.feed_end(on_chunk=on_chunk, defer_delivery=defer_delivery)
