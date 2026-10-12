"""LLM-backed agents, one per classified enquiry intent."""

from __future__ import annotations

import json

from gnaniassignment.llm.agents.base import AgentResponse, SubAgent
from gnaniassignment.llm.chat import ChatModel
from gnaniassignment.llm.prompts.enquiry_agents import PROMPTS
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever


class EnquiryAgent(SubAgent):
    def __init__(self, name: str, model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.", retriever: IntentScopedRetriever | EmptyRetriever | None = None) -> None:
        super().__init__()
        self.name, self.model, self.knowledge, self.retriever = name, model, knowledge, retriever or EmptyRetriever()
        self.system_prompt = PROMPTS[name]

    def reply(self, text: str, slots: dict[str, str], memory: str = "") -> AgentResponse:
        retrieved = self.retriever.retrieve(self.name, text)
        knowledge = "\n\n".join(retrieved) or self.knowledge
        user_prompt = f"Approved knowledge:\n{knowledge}\n\nRelevant conversation:\n{memory}\n\nCurrent user query:\n{text}"
        raw = self.model.complete(self.system_prompt, user_prompt)
        try:
            payload = json.loads(raw)
            answer = str(payload["answer"]).strip()
            summary = str(payload["memory_summary"]).strip()
            if answer and summary:
                return AgentResponse(answer=answer, memory_summary=summary)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        # A malformed model response still remains useful, while ensuring the
        # stored context is shorter than the original request/response pair.
        answer = raw.strip()
        return AgentResponse(answer=answer, memory_summary=f"User asked: {text}. Answer: {answer[:280]}")


def enquiry_agents(model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.", retriever: IntentScopedRetriever | EmptyRetriever | None = None) -> dict[str, SubAgent]:
    return {name: EnquiryAgent(name, model, knowledge, retriever) for name in PROMPTS}
