"""LLM-backed agents, one per classified enquiry intent."""

from __future__ import annotations

import json
import re
from typing import Callable, Mapping

from gnaniassignment.llm.agents.base import AgentResponse, SubAgent
from gnaniassignment.llm.chat import ChatModel
from gnaniassignment.llm.prompts.enquiry_agents import PROMPTS
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever

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
        return self._parse_response(self.model.complete(system_prompt, user_prompt), text)

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
        for chunk in stream_complete(system_prompt, user_prompt):
            raw_parts.append(chunk)
            answer_delta = answer_stream.feed(chunk)
            if answer_delta:
                on_chunk(answer_delta)
        output = self._parse_response("".join(raw_parts), text)
        if not answer_stream.emitted:
            on_chunk(output.answer)
        return output

    def _prompts(self, text: str, memory: str) -> tuple[str, str]:
        retrieved = self.retriever.retrieve(self.name, text, limit=4)
        knowledge = ("\n\n".join(retrieved) or self.knowledge)[:MAX_KNOWLEDGE_CHARS]
        user_prompt = f"Approved knowledge:\n{knowledge}\n\nRelevant conversation:\n{memory}\n\nCurrent user query:\n{text}"
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
