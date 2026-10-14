"""Agents for the published Hinglish retail intent taxonomy."""

from __future__ import annotations

from gnaniassignment.llm.agents.enquiry import EnquiryAgent
from gnaniassignment.llm.chat import ChatModel
from gnaniassignment.llm.prompts.retail_agents import PROMPTS
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever


class RetailAgent(EnquiryAgent):
    def __init__(self, name: str, model: ChatModel, knowledge: str, retriever: IntentScopedRetriever | EmptyRetriever | None) -> None:
        super().__init__(name, model, knowledge, retriever, prompts=PROMPTS)


def retail_agents(model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.", retriever: IntentScopedRetriever | EmptyRetriever | None = None):
    return {name: RetailAgent(name, model, knowledge, retriever) for name in PROMPTS}
