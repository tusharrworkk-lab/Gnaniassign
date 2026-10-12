"""LLM-backed agents, one per classified enquiry intent."""

from __future__ import annotations

from gnaniassignment.llm.agents.base import SubAgent
from gnaniassignment.llm.chat import ChatModel
from gnaniassignment.llm.prompts.enquiry_agents import PROMPTS


class EnquiryAgent(SubAgent):
    def __init__(self, name: str, model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.") -> None:
        super().__init__()
        self.name, self.model, self.knowledge = name, model, knowledge
        self.system_prompt = PROMPTS[name]

    def reply(self, text: str, slots: dict[str, str], memory: str = "") -> str:
        user_prompt = f"Approved knowledge:\n{self.knowledge}\n\nRelevant conversation:\n{memory}\n\nCurrent user query:\n{text}"
        return self.model.complete(self.system_prompt, user_prompt)


def enquiry_agents(model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.") -> dict[str, SubAgent]:
    return {name: EnquiryAgent(name, model, knowledge) for name in PROMPTS}
