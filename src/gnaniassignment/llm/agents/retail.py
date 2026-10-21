"""Agents for the published Hinglish retail intent taxonomy."""

from __future__ import annotations

from gnaniassignment.llm.agents.enquiry import EnquiryAgent
from gnaniassignment.llm.chat import ChatModel
from gnaniassignment.llm.prompts.retail_agents import CLARIFICATION_PROMPT, PROMPTS
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever


class RetailAgent(EnquiryAgent):
    def __init__(self, name: str, model: ChatModel, knowledge: str, retriever: IntentScopedRetriever | EmptyRetriever | None) -> None:
        super().__init__(name, model, knowledge, retriever, prompts=PROMPTS)


def retail_agents(model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.", retriever: IntentScopedRetriever | EmptyRetriever | None = None):
    return {name: RetailAgent(name, model, knowledge, retriever) for name in PROMPTS}


def retail_clarification_agent(
    model: ChatModel,
    knowledge: str = "No approved knowledge has been loaded yet.",
    retriever: IntentScopedRetriever | EmptyRetriever | None = None,
) -> EnquiryAgent:
    """Create the LLM-backed fallback agent without adding it to the intent taxonomy."""
    return EnquiryAgent(
        "clarify",
        model,
        knowledge,
        retriever,
        prompts={"clarify": CLARIFICATION_PROMPT},
    )
