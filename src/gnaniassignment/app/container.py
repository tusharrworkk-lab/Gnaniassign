"""Single composition point for production dependencies."""

from __future__ import annotations

from gnaniassignment.app.config import Settings, load_environment
from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.chat import VercelGatewayChatModel
from gnaniassignment.llm.local_laya import LocalLayaRouter
from gnaniassignment.llm.router import LayaGatewayRouter
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever
from gnaniassignment.orchestration.graph import build_graph
from gnaniassignment.orchestration.session import TurnSession


def create_intent_router(settings: Settings) -> LayaGatewayRouter | LocalLayaRouter:
    if settings.intent_provider == "local_laya":
        return LocalLayaRouter()
    if settings.intent_provider == "laya":
        return LayaGatewayRouter()
    raise ValueError("INTENT_PROVIDER must be 'local_laya' or 'laya'.")


def create_session(
    settings: Settings | None = None,
    retriever: IntentScopedRetriever | EmptyRetriever | None = None,
    knowledge: str = "No approved knowledge has been loaded yet.",
) -> TurnSession:
    load_environment()
    settings = settings or Settings.from_environment()
    router = create_intent_router(settings)
    agents = enquiry_agents(VercelGatewayChatModel(), knowledge=knowledge, retriever=retriever)
    return TurnSession(build_graph(router, agents, threshold=settings.intent_threshold))
