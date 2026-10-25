"""Retail response agents matching the nine-label MiniLM taxonomy."""

from __future__ import annotations

from gnaniassignment.llm.agents.enquiry import EnquiryAgent
from gnaniassignment.llm.chat import ChatModel
from gnaniassignment.llm.prompts.retail_agents import BASE, CLARIFICATION_PROMPT
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever


PROMPTS = {
    "track_order": BASE + "\nFocus: order tracking, shipment status, and required order details.",
    "return_item": BASE + "\nFocus: return requests, eligibility, and the next approved step.",
    "cancel_order": BASE + "\nFocus: whether and how an order can be cancelled.",
    "refund_status": BASE + "\nFocus: refund progress, timing, and required order details.",
    "wrong_item": BASE + "\nFocus: incorrect-item reports and the next approved support step.",
    "size_issue": BASE + "\nFocus: size or fit problems and the approved exchange or return step.",
    "payment_issue": BASE + "\nFocus: failed, pending, duplicated, or disputed payments.",
    "delivery_address": BASE + "\nFocus: delivery-address corrections and required order details.",
}


def retail_minilm_agents(model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.", retriever: IntentScopedRetriever | EmptyRetriever | None = None) -> dict[str, EnquiryAgent]:
    return {name: EnquiryAgent(name, model, knowledge, retriever, prompts=PROMPTS) for name in PROMPTS}


def retail_minilm_clarification_agent(model: ChatModel, knowledge: str = "No approved knowledge has been loaded yet.", retriever: IntentScopedRetriever | EmptyRetriever | None = None) -> EnquiryAgent:
    return EnquiryAgent("clarify", model, knowledge, retriever, prompts={"clarify": CLARIFICATION_PROMPT})
