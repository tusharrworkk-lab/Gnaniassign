"""Single composition point for production dependencies."""

from __future__ import annotations

from gnaniassignment.app.config import Settings, load_environment
from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.chat import OpenRouterChatModel
from gnaniassignment.llm.router import LayaGatewayRouter
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever
from gnaniassignment.orchestration.graph import build_graph
from gnaniassignment.orchestration.session import TurnSession


def create_session(
    settings: Settings | None = None,
    retriever: IntentScopedRetriever | EmptyRetriever | None = None,
    knowledge: str = "No approved knowledge has been loaded yet.",
) -> TurnSession:
    load_environment()
    settings = settings or Settings.from_environment()
    if settings.intent_provider == "laya":
        router = LayaGatewayRouter()
    else:
        raise ValueError("INTENT_PROVIDER must be 'laya'.")
    agents = enquiry_agents(OpenRouterChatModel(), knowledge=knowledge, retriever=retriever)
    return TurnSession(build_graph(router, agents, threshold=settings.intent_threshold))
