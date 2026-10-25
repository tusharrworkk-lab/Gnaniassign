"""Single composition point for production dependencies."""

from __future__ import annotations

import os

from gnaniassignment.app.config import Settings, load_environment
from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.agents.retail import retail_agents, retail_clarification_agent
from gnaniassignment.llm.agents.retail_minilm import retail_minilm_agents, retail_minilm_clarification_agent
from gnaniassignment.llm.chat import BedrockChatModel, CerebrasChatModel, ChatModel, GroqChatModel, SimplismartChatModel, VercelGatewayChatModel, VertexGeminiChatModel, VllmGemmaChatModel
from gnaniassignment.llm.laya_cpp import LayaCppRouter
from gnaniassignment.llm.laya_lambda import LayaLambdaRouter
from gnaniassignment.llm.local_laya import LocalLayaRouter
from gnaniassignment.llm.retail_classifier import HinglishRetailRouter
from gnaniassignment.llm.retail_minilm import RetailMiniLMRouter
from gnaniassignment.llm.router import LayaGatewayRouter
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever, PostgresKnowledgeStore
from gnaniassignment.orchestration.graph import build_graph
from gnaniassignment.orchestration.episodic_memory import PostgresEpisodicMemoryStore
from gnaniassignment.orchestration.session import TurnSession


def create_intent_router(settings: Settings) -> LayaGatewayRouter | LocalLayaRouter | LayaCppRouter | LayaLambdaRouter | HinglishRetailRouter | RetailMiniLMRouter:
    if settings.intent_provider == "hinglish_retail":
        router = HinglishRetailRouter()
        router.warmup()
        return router
    if settings.intent_provider == "retail_minilm":
        router = RetailMiniLMRouter()
        router.warmup()
        return router
    if settings.intent_provider == "local_laya":
        return LocalLayaRouter()
    if settings.intent_provider == "laya_cpp":
        return LayaCppRouter()
    if settings.intent_provider == "laya_lambda":
        return LayaLambdaRouter()
    if settings.intent_provider == "laya":
        return LayaGatewayRouter()
    raise ValueError("INTENT_PROVIDER must be 'retail_minilm', 'hinglish_retail', 'local_laya', 'laya_cpp', 'laya_lambda', or 'laya'.")


def create_chat_model(settings: Settings) -> ChatModel:
    if settings.llm_provider == "bedrock":
        return BedrockChatModel()
    if settings.llm_provider == "simplismart":
        return SimplismartChatModel()
    if settings.llm_provider == "groq":
        return GroqChatModel()
    if settings.llm_provider == "cerebras":
        return CerebrasChatModel()
    if settings.llm_provider == "vertex_gemini":
        return VertexGeminiChatModel()
    if settings.llm_provider == "vllm_gemma":
        return VllmGemmaChatModel()
    if settings.llm_provider == "vercel":
        return VercelGatewayChatModel()
    raise ValueError("LLM_PROVIDER must be 'bedrock', 'simplismart', 'groq', 'cerebras', 'vertex_gemini', 'vllm_gemma', or 'vercel'.")


def create_episodic_memory_store(customer_id: str | None = None) -> tuple[PostgresEpisodicMemoryStore | None, str | None]:
    """Build persistent memory only when both a database and customer ID exist."""
    resolved_customer_id = customer_id or os.getenv("EPISODIC_MEMORY_CUSTOMER_ID")
    database_url = os.getenv("EPISODIC_MEMORY_DATABASE_URL")
    return (PostgresEpisodicMemoryStore(database_url) if database_url and resolved_customer_id else None, resolved_customer_id)


def create_knowledge_retriever(intents: list[str] | tuple[str, ...]) -> IntentScopedRetriever | None:
    if os.getenv("KNOWLEDGE_BASE_ENABLED", "false").lower() != "true":
        return None
    database_url = os.getenv("KNOWLEDGE_BASE_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
    if not database_url:
        raise ValueError("Set KNOWLEDGE_BASE_DATABASE_URL or EPISODIC_MEMORY_DATABASE_URL when KNOWLEDGE_BASE_ENABLED=true")
    return IntentScopedRetriever({intent: PostgresKnowledgeStore(intent, database_url) for intent in intents})


def create_session(
    settings: Settings | None = None,
    retriever: IntentScopedRetriever | EmptyRetriever | None = None,
    knowledge: str = "No approved knowledge has been loaded yet.",
    customer_id: str | None = None,
) -> TurnSession:
    load_environment()
    settings = settings or Settings.from_environment()
    router = create_intent_router(settings)
    model = create_chat_model(settings)
    if settings.intent_provider == "hinglish_retail":
        retriever = retriever or create_knowledge_retriever(tuple(retail_agents(model, knowledge=knowledge).keys()))
        agents = retail_agents(model, knowledge=knowledge, retriever=retriever)
        clarification_agent = retail_clarification_agent(model, knowledge=knowledge, retriever=retriever)
    elif settings.intent_provider == "retail_minilm":
        retriever = retriever or create_knowledge_retriever(tuple(retail_minilm_agents(model, knowledge=knowledge).keys()))
        agents = retail_minilm_agents(model, knowledge=knowledge, retriever=retriever)
        clarification_agent = retail_minilm_clarification_agent(model, knowledge=knowledge, retriever=retriever)
    else:
        agents = enquiry_agents(model, knowledge=knowledge, retriever=retriever)
        clarification_agent = None
    memory_store, resolved_customer_id = create_episodic_memory_store(customer_id)
    return TurnSession(
        build_graph(router, agents, threshold=settings.intent_threshold, clarification_agent=clarification_agent),
        memory_store=memory_store,
        customer_id=resolved_customer_id,
    )
