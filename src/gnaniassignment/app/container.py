"""Single composition point for production dependencies."""

from __future__ import annotations

import os

from gnaniassignment.app.config import Settings, load_environment
from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.agents.generic import generic_agents
from gnaniassignment.llm.agents.retail import retail_agents, retail_clarification_agent
from gnaniassignment.llm.agents.retail_minilm import retail_minilm_agents, retail_minilm_clarification_agent
from gnaniassignment.llm.chat import BedrockChatModel, BedrockGptOssChatModel, CerebrasChatModel, ChatModel, GroqChatModel, SimplismartChatModel, VercelGatewayChatModel, VertexGeminiChatModel, VllmGemmaChatModel
from gnaniassignment.llm.laya_cpp import LayaCppRouter
from gnaniassignment.llm.laya_lambda import LayaLambdaRouter
from gnaniassignment.llm.local_laya import LocalLayaRouter
from gnaniassignment.llm.retail_classifier import HinglishRetailRouter
from gnaniassignment.llm.retail_minilm import RetailMiniLMRouter
from gnaniassignment.llm.generic_router import GenericRouter
from gnaniassignment.llm.router import LayaGatewayRouter
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever, PostgresKnowledgeStore
from gnaniassignment.orchestration.graph import build_graph
from gnaniassignment.orchestration.episodic_memory import PostgresEpisodicMemoryStore
from gnaniassignment.orchestration.session import TurnSession
from gnaniassignment.order_data import PostgresCustomerOrderStore, customer_order_window
from gnaniassignment.intent_feedback import PostgresIntentFeedbackStore
from gnaniassignment.conversation_analytics import AcilConversationTracer, BedrockConversationJudge, ConversationJudge, PostgresConversationStore


def create_intent_router(settings: Settings, customer_id: str | None = None, conversation_store: PostgresConversationStore | None = None) -> LayaGatewayRouter | LocalLayaRouter | LayaCppRouter | LayaLambdaRouter | HinglishRetailRouter | RetailMiniLMRouter | GenericRouter:
    if settings.intent_provider == "generic":
        if os.getenv("SHADOW_INTENT_ENABLED", "false").lower() == "true":
            store = PostgresIntentFeedbackStore()
            shadow_router = RetailMiniLMRouter()
            return GenericRouter(
                shadow_router=shadow_router,
                on_shadow_prediction=lambda query, prediction, session_id: _record_shadow_prediction(store, conversation_store, customer_id, session_id, query, prediction),
            )
        return GenericRouter()
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
    raise ValueError("INTENT_PROVIDER must be 'generic', 'retail_minilm', 'hinglish_retail', 'local_laya', 'laya_cpp', 'laya_lambda', or 'laya'.")


def create_chat_model(settings: Settings) -> ChatModel:
    if settings.llm_provider == "bedrock_gpt_oss":
        return BedrockGptOssChatModel()
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
    raise ValueError("LLM_PROVIDER must be 'bedrock_gpt_oss', 'bedrock', 'simplismart', 'groq', 'cerebras', 'vertex_gemini', 'vllm_gemma', or 'vercel'.")


def create_episodic_memory_store(customer_id: str | None = None) -> tuple[PostgresEpisodicMemoryStore | None, str | None]:
    """Build persistent memory only when both a database and customer ID exist."""
    resolved_customer_id = customer_id or os.getenv("EPISODIC_MEMORY_CUSTOMER_ID")
    database_url = os.getenv("EPISODIC_MEMORY_DATABASE_URL")
    return (PostgresEpisodicMemoryStore(database_url) if database_url and resolved_customer_id else None, resolved_customer_id)


def create_customer_order_context(customer_id: str | None) -> str | None:
    """Load verified mock orders only when the optional store is enabled."""
    if not customer_id or os.getenv("ORDER_DATA_ENABLED", "false").lower() != "true":
        return None
    return PostgresCustomerOrderStore().render_customer_orders(customer_id)


def create_customer_order_window(customer_id: str | None):
    """Preload a caller-scoped order window before the call, then refresh per turn."""
    if not customer_id or os.getenv("ORDER_DATA_ENABLED", "false").lower() != "true":
        return None
    return customer_order_window(customer_id)


def create_conversation_store() -> PostgresConversationStore | None:
    return PostgresConversationStore() if os.getenv("CONVERSATION_LOGGING_ENABLED", "false").lower() == "true" else None


def create_acil_tracer() -> AcilConversationTracer | None:
    return AcilConversationTracer() if os.getenv("ACIL_TRACE_ENABLED", "false").lower() == "true" else None


def create_call_judge() -> ConversationJudge | None:
    return BedrockConversationJudge() if os.getenv("CONVERSATION_JUDGE_ENABLED", "false").lower() == "true" else None


def _record_shadow_prediction(store: PostgresIntentFeedbackStore, conversation_store: PostgresConversationStore | None, customer_id: str | None, session_id: str | None, query: str, prediction: dict) -> None:
    store.record_prediction(customer_id, query, prediction, session_id)
    decision = prediction.get("answers", {}).get("intent", {})
    if conversation_store is not None and isinstance(decision.get("choice"), str) and isinstance(decision.get("confidence"), (int, float)):
        conversation_store.attach_classifier_prediction(session_id, query, decision["choice"], float(decision["confidence"]))


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
    memory_store, resolved_customer_id = create_episodic_memory_store(customer_id)
    conversation_store = create_conversation_store()
    router = create_intent_router(settings, resolved_customer_id, conversation_store)
    model = create_chat_model(settings)
    if settings.intent_provider == "generic":
        guidance = conversation_store.active_guidance() if conversation_store is not None else ""
        retriever = retriever or create_knowledge_retriever(("generic",))
        agents = generic_agents(model, knowledge=knowledge, retriever=retriever, guidance=guidance)
        clarification_agent = None
    elif settings.intent_provider == "hinglish_retail":
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
    return TurnSession(
        build_graph(router, agents, threshold=settings.intent_threshold, clarification_agent=clarification_agent),
        memory_store=memory_store,
        customer_id=resolved_customer_id,
        conversation_store=conversation_store,
        tracer=create_acil_tracer(),
        call_judge=create_call_judge(),
        account_context_provider=create_customer_order_window(resolved_customer_id),
    )
