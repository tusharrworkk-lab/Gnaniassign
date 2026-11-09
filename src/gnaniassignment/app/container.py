"""Single composition point for production dependencies."""

from __future__ import annotations

import os
from collections.abc import Callable

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
from gnaniassignment.prompts.tree import PostgresPromptTreeStore, PromptPlanResolver, RedisPromptPlanCache
from gnaniassignment.work_items import PlaneClient, WorkItemProviderError, render_work_item_voice_context, requested_work_item_identifier


# Storage implementations live behind the existing protocols. Add a factory
# here to introduce Redis, a vector database, or a merchant API without
# changing the agent graph or session code.
EPISODIC_MEMORY_FACTORIES: dict[str, Callable[[str], PostgresEpisodicMemoryStore]] = {
    "postgres": PostgresEpisodicMemoryStore,
}
KNOWLEDGE_STORE_FACTORIES: dict[str, Callable[[str, str], PostgresKnowledgeStore]] = {
    "postgres": lambda intent, database_url: PostgresKnowledgeStore(intent, database_url),
}
ORDER_CONTEXT_FACTORIES: dict[str, Callable[[str], PostgresCustomerOrderStore]] = {
    "postgres": PostgresCustomerOrderStore,
}


def _configured_provider(variable: str, factories: dict[str, object]) -> str:
    provider = os.getenv(variable, "postgres").strip().lower()
    if provider == "none":
        return provider
    if provider not in factories:
        raise ValueError(f"Unsupported {variable} {provider!r}. Available: none, {', '.join(sorted(factories))}")
    return provider


def create_intent_router(settings: Settings, customer_id: str | None = None, conversation_store: PostgresConversationStore | None = None, shadow_observer: Callable[[dict], None] | None = None, shadow_error_observer: Callable[[Exception], None] | None = None) -> LayaGatewayRouter | LocalLayaRouter | LayaCppRouter | LayaLambdaRouter | HinglishRetailRouter | RetailMiniLMRouter | GenericRouter:
    if settings.intent_provider == "generic":
        if os.getenv("SHADOW_INTENT_ENABLED", "false").lower() == "true":
            store = PostgresIntentFeedbackStore()
            shadow_router = RetailMiniLMRouter()
            def on_shadow_prediction(query: str, prediction: dict, session_id: str | None) -> None:
                _record_shadow_prediction(store, conversation_store, customer_id, session_id, query, prediction)
                if shadow_observer is not None:
                    shadow_observer(prediction)

            return GenericRouter(
                shadow_router=shadow_router,
                on_shadow_prediction=on_shadow_prediction,
                on_shadow_error=shadow_error_observer,
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
    provider = _configured_provider("EPISODIC_MEMORY_PROVIDER", EPISODIC_MEMORY_FACTORIES)
    if provider == "none" or not database_url or not resolved_customer_id:
        return None, resolved_customer_id
    return EPISODIC_MEMORY_FACTORIES[provider](database_url), resolved_customer_id


def create_customer_order_context(customer_id: str | None) -> str | None:
    """Load verified mock orders only when the optional store is enabled."""
    provider = _configured_provider("ORDER_CONTEXT_PROVIDER", ORDER_CONTEXT_FACTORIES)
    if provider == "none" or not customer_id or os.getenv("ORDER_DATA_ENABLED", "false").lower() != "true":
        return None
    database_url = os.getenv("ORDER_DATA_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
    return ORDER_CONTEXT_FACTORIES[provider](database_url).render_customer_orders(customer_id)


def create_customer_order_window(customer_id: str | None):
    """Preload a caller-scoped order window before the call, then refresh per turn."""
    provider = _configured_provider("ORDER_CONTEXT_PROVIDER", ORDER_CONTEXT_FACTORIES)
    if provider == "none" or not customer_id or os.getenv("ORDER_DATA_ENABLED", "false").lower() != "true":
        return None
    database_url = os.getenv("ORDER_DATA_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
    return customer_order_window(customer_id, ORDER_CONTEXT_FACTORIES[provider](database_url))


def create_calendar_presence_window():
    """Read the pre-refreshed private calendar presence snapshot.

    This deliberately performs no Google request while a caller is waiting;
    the API background task refreshes the snapshot ahead of calls.
    """
    if os.getenv("GOOGLE_CALENDAR_ENABLED", "false").lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        from gnaniassignment.google_calendar import GoogleCalendarSettings, GoogleCalendarStore, calendar_presence_context
        settings = GoogleCalendarSettings.from_environment()
        store = GoogleCalendarStore(settings)
    except (RuntimeError, ValueError):
        # Calendar setup must never stop the voice agent from answering.
        return None

    def load(_query: str, _working_memory: dict[str, str]) -> str:
        try:
            return calendar_presence_context(store, settings.owner_id)
        except RuntimeError:
            return "Calendar presence: unavailable. Do not claim the owner's availability."

    return load


def _normalised_phone(value: str | None) -> str:
    """Compare phone identities without leaking full numbers into prompts."""
    digits = "".join(char for char in (value or "") if char.isdigit())
    return digits[-10:] if len(digits) >= 10 else ""


def create_plane_work_item_window(
    customer_id: str | None,
    client_factory: Callable[[], PlaneClient] = PlaneClient,
):
    """Fetch one Plane item only for an explicitly requested trusted caller.

    No Plane lookup happens at call start or on ordinary turns. This protects
    call latency and keeps work data absent unless a caller supplies a concrete
    item identifier and their inbound caller ID is allowlisted.
    """
    if os.getenv("PLANE_VOICE_STATUS_ENABLED", "false").lower() not in {"1", "true", "yes", "on"}:
        return None
    caller = _normalised_phone(customer_id.removeprefix("caller:") if customer_id else None)
    trusted = {
        number for raw in os.getenv("PLANE_VOICE_TRUSTED_CALLERS", "").split(",")
        if (number := _normalised_phone(raw))
    }
    if not caller or caller not in trusted:
        return None

    def load(query: str, _working_memory: dict[str, str]) -> str:
        identifier = requested_work_item_identifier(query)
        if identifier is None:
            return ""
        try:
            item = client_factory().find_work_item(identifier)
        except (ValueError, WorkItemProviderError):
            return "Verified Plane work-item context: lookup is temporarily unavailable. Do not invent a status."
        if item is None:
            return f"Verified Plane work-item context: no item matching {identifier} was found. Do not invent a status."
        return render_work_item_voice_context(item)

    return load


def combine_account_context_providers(*providers):
    active = tuple(provider for provider in providers if provider is not None)
    if not active:
        return None

    def load(query: str, working_memory: dict[str, str]) -> str:
        return "\n\n".join(provider(query, working_memory) for provider in active)

    return load


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
    provider = _configured_provider("KNOWLEDGE_BASE_PROVIDER", KNOWLEDGE_STORE_FACTORIES)
    if provider == "none":
        return None
    database_url = os.getenv("KNOWLEDGE_BASE_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
    if not database_url:
        raise ValueError("Set KNOWLEDGE_BASE_DATABASE_URL or EPISODIC_MEMORY_DATABASE_URL when KNOWLEDGE_BASE_ENABLED=true")
    return IntentScopedRetriever({intent: KNOWLEDGE_STORE_FACTORIES[provider](intent, database_url) for intent in intents})


def create_prompt_resolver() -> PromptPlanResolver | None:
    """Build the optional tree resolver; L1 cache keeps it off the turn path."""
    if os.getenv("PROMPT_TREE_ENABLED", "false").lower() != "true":
        return None
    cache = None
    if os.getenv("PROMPT_CACHE_PROVIDER", "none").lower() == "redis":
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            raise ValueError("Set REDIS_URL when PROMPT_CACHE_PROVIDER=redis")
        cache = RedisPromptPlanCache(redis_url)
    return PromptPlanResolver(
        PostgresPromptTreeStore(),
        cache=cache,
        ttl_seconds=int(os.getenv("PROMPT_PLAN_CACHE_TTL_SECONDS", "300")),
    )


def resolve_prompt(
    resolver: PromptPlanResolver | None,
    agent_id: str,
    fallback: str,
    customer_id: str | None = None,
) -> str:
    if resolver is None:
        return fallback
    plan = resolver.resolve(
        os.getenv("TENANT_ID", "default"),
        agent_id,
    )
    if plan is None:
        return fallback
    prompt = plan.render({
        "agent": agent_id,
        "channel": os.getenv("AGENT_CHANNEL", "voice"),
        "language": os.getenv("AGENT_LANGUAGE", "hinglish"),
        "customer_id": customer_id or "",
    })
    return prompt or fallback


def create_session(
    settings: Settings | None = None,
    retriever: IntentScopedRetriever | EmptyRetriever | None = None,
    knowledge: str = "No approved knowledge has been loaded yet.",
    customer_id: str | None = None,
    session_id: str | None = None,
    shadow_observer: Callable[[dict], None] | None = None,
    shadow_error_observer: Callable[[Exception], None] | None = None,
) -> TurnSession:
    load_environment()
    settings = settings or Settings.from_environment()
    memory_store, resolved_customer_id = create_episodic_memory_store(customer_id)
    conversation_store = create_conversation_store()
    prompt_resolver = create_prompt_resolver()
    router = create_intent_router(
        settings,
        resolved_customer_id,
        conversation_store,
        shadow_observer=shadow_observer,
        shadow_error_observer=shadow_error_observer,
    )
    model = create_chat_model(settings)
    if settings.intent_provider == "generic":
        guidance = conversation_store.active_guidance() if conversation_store is not None else ""
        retriever = retriever or create_knowledge_retriever(("generic",))
        agents = generic_agents(
            model,
            knowledge=knowledge,
            retriever=retriever,
            guidance=guidance,
            prompt_override=resolve_prompt(
                prompt_resolver, "generic", "", customer_id=resolved_customer_id
            ),
        )
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
        account_context_provider=combine_account_context_providers(
            create_customer_order_window(resolved_customer_id),
            create_calendar_presence_window(),
            create_plane_work_item_window(resolved_customer_id),
        ),
        session_id=session_id,
    )
