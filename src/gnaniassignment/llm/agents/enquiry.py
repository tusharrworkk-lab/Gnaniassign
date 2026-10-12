"""LLM-backed agents, one per classified enquiry intent."""

from __future__ import annotations

import json

from gnaniassignment.llm.agents.base import AgentResponse, SubAgent
from gnaniassignment.llm.chat import ChatModel
from gnaniassignment.llm.prompts.enquiry_agents import PROMPTS
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever

MAX_KNOWLEDGE_CHARS = 6000
MAX_SUMMARY_CHARS = 320


class EnquiryAgent(SubAgent):
    def __init__(self, name: str, model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.", retriever: IntentScopedRetriever | EmptyRetriever | None = None) -> None:
        super().__init__()
        self.name, self.model, self.knowledge, self.retriever = name, model, knowledge, retriever or EmptyRetriever()
        self.system_prompt = PROMPTS[name]

    def reply(self, text: str, slots: dict[str, str], memory: str = "") -> AgentResponse:
        retrieved = self.retriever.retrieve(self.name, text, limit=4)
        knowledge = ("\n\n".join(retrieved) or self.knowledge)[:MAX_KNOWLEDGE_CHARS]
        user_prompt = f"Approved knowledge:\n{knowledge}\n\nRelevant conversation:\n{memory}\n\nCurrent user query:\n{text}"
        raw = self.model.complete(self.system_prompt, user_prompt)
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
