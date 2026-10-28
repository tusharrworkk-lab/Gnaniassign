"""LLM-backed agents, one per classified enquiry intent."""

from __future__ import annotations

import json
import re
from typing import Callable, Mapping

from gnaniassignment.llm.agents.base import AgentResponse, SubAgent
from gnaniassignment.llm.chat import ChatModel
from gnaniassignment.llm.prompts.enquiry_agents import PROMPTS
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever
from gnaniassignment.observability.langsmith import trace_llm_call, trace_llm_stream

MAX_KNOWLEDGE_CHARS = 6000
MAX_SUMMARY_CHARS = 320
ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"')


class _AnswerStream:
    """Decode only the answer JSON string as chunks arrive."""

    def __init__(self) -> None:
        self.raw = ""
        self.position: int | None = None
        self.raw_answer = ""
        self.emitted = ""
        self.escaped = False
        self.finished = False

    def feed(self, chunk: str) -> str:
        self.raw += chunk
        if self.finished:
            return ""
        if self.position is None:
            match = ANSWER_FIELD.search(self.raw)
            if match is None:
                return ""
            self.position = match.end()
        while self.position < len(self.raw):
            char = self.raw[self.position]
            self.position += 1
            if self.escaped:
                self.raw_answer += char
                self.escaped = False
            elif char == "\\":
                self.raw_answer += char
                self.escaped = True
            elif char == '"':
                self.finished = True
                break
            else:
                self.raw_answer += char
        try:
            decoded = json.loads('"' + self.raw_answer + '"')
        except json.JSONDecodeError:
            return ""
        # Wait for a matching low surrogate before emitting a Unicode escape pair.
        if decoded and 0xD800 <= ord(decoded[-1]) <= 0xDBFF:
            decoded = decoded[:-1]
        delta = decoded[len(self.emitted):]
        self.emitted = decoded
        return delta


class EnquiryAgent(SubAgent):
    def __init__(self, name: str, model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.", retriever: IntentScopedRetriever | EmptyRetriever | None = None, prompts: Mapping[str, str] = PROMPTS) -> None:
        super().__init__()
        self.name, self.model, self.knowledge, self.retriever = name, model, knowledge, retriever or EmptyRetriever()
        self.system_prompt = prompts[name]

    def reply(self, text: str, slots: dict[str, str], memory: str = "") -> AgentResponse:
        system_prompt, user_prompt = self._prompts(text, memory)
        raw = trace_llm_call(
            agent=self.name,
            model=getattr(self.model, "model", None),
            system=system_prompt,
            user=user_prompt,
            invoke=lambda: self.model.complete(system_prompt, user_prompt),
        )
        return self._parse_response(raw, text)

    def run_stream(self, text: str, slots: dict[str, str], memory: str, on_chunk: Callable[[str], None]) -> AgentResponse:
        self.calls += 1
        system_prompt, user_prompt = self._prompts(text, memory)
        stream_complete = getattr(self.model, "stream_complete", None)
        if stream_complete is None:
            output = self._parse_response(self.model.complete(system_prompt, user_prompt), text)
            on_chunk(output.answer)
            return output
        answer_stream = _AnswerStream()
        raw_parts: list[str] = []
        structured: bool | None = None
        plain_answer_streamed = False
        for chunk in trace_llm_stream(
            agent=self.name,
            model=getattr(self.model, "model", None),
            system=system_prompt,
            user=user_prompt,
            invoke=lambda: stream_complete(system_prompt, user_prompt),
        ):
            raw_parts.append(chunk)
            if structured is None:
                joined = "".join(raw_parts).lstrip()
                if not joined:
                    continue
                structured = joined.startswith("{")
            if structured:
                answer_delta = answer_stream.feed(chunk)
                if answer_delta:
                    on_chunk(answer_delta)
            else:
                on_chunk(chunk)
                plain_answer_streamed = True
        output = self._parse_response("".join(raw_parts), text)
        if not answer_stream.emitted and not plain_answer_streamed:
            on_chunk(output.answer)
        return output

    def _prompts(self, text: str, memory: str) -> tuple[str, str]:
        retrieved = self.retriever.retrieve(self.name, text, limit=4)
        knowledge_parts = [self.knowledge, *retrieved]
        knowledge = "\n\n".join(part for part in knowledge_parts if part and part != "No approved knowledge has been loaded yet.")
        knowledge = (knowledge or "No approved knowledge has been loaded yet.")[:MAX_KNOWLEDGE_CHARS]
        provided = []
        if "- order_id:" in memory:
            provided.append("order ID")
        if "- payment_reference:" in memory:
            provided.append("payment reference")
        slot_rule = ""
        if provided:
            slot_rule = f"\n\nMANDATORY: The following fields are already provided in Working memory: {', '.join(provided)}. Do not ask the customer for any of them again."
        verified_order = re.search(r"Verified matching account order window.*?order ID ([A-Za-z0-9-]+)", memory, flags=re.IGNORECASE | re.DOTALL)
        if verified_order:
            slot_rule += f"\n\nMANDATORY: A verified account record for order ID {verified_order.group(1)} is already selected for this turn. Use its supplied status directly. Do not ask for an order ID."
        user_prompt = f"Approved knowledge:\n{knowledge}\n\nRelevant conversation:\n{memory}\n\nCurrent user query:\n{text}{slot_rule}"
        return self.system_prompt, user_prompt

    @staticmethod
    def _parse_response(raw: str, text: str) -> AgentResponse:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str) or not isinstance(payload.get("memory_summary"), str):
                raise ValueError("Invalid response fields")
            answer = payload["answer"].strip()
            summary = payload["memory_summary"].strip()[:MAX_SUMMARY_CHARS]
            if answer and summary:
                return AgentResponse(answer=answer, memory_summary=summary)
        except (json.JSONDecodeError, ValueError):
            if raw.lstrip().startswith("{"):
                raise ValueError("Subagent returned malformed JSON") from None
        # Some small models may return plain text despite the requested format.
        answer = raw.strip()
        if not answer:
            raise ValueError("Subagent returned an empty answer")
        summary = f"User asked: {text[:120]}. Answer: {answer[:180]}"
        return AgentResponse(answer=answer, memory_summary=summary[:MAX_SUMMARY_CHARS])


def enquiry_agents(model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.", retriever: IntentScopedRetriever | EmptyRetriever | None = None) -> dict[str, SubAgent]:
    return {name: EnquiryAgent(name, model, knowledge, retriever) for name in PROMPTS}
