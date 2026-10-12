"""Single composition point for production dependencies."""

from __future__ import annotations

from gnaniassignment.app.config import Settings
from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.chat import OpenRouterChatModel
from gnaniassignment.llm.router import JevRouter
from gnaniassignment.orchestration.graph import build_graph
from gnaniassignment.orchestration.session import TurnSession


def create_session(settings: Settings | None = None) -> TurnSession:
    settings = settings or Settings.from_environment()
    if settings.intent_provider == "jev":
        router = JevRouter()
    else:
        raise ValueError("INTENT_PROVIDER must be 'jev'.")
    return TurnSession(build_graph(router, enquiry_agents(OpenRouterChatModel()), threshold=settings.intent_threshold))
