"""Single composition point for production dependencies."""

from __future__ import annotations

from gnaniassignment.app.config import Settings, load_environment
from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.agents.retail import retail_agents, retail_clarification_agent
from gnaniassignment.llm.chat import BedrockChatModel, CerebrasChatModel, ChatModel, GroqChatModel, SimplismartChatModel, VercelGatewayChatModel, VertexGeminiChatModel, VllmGemmaChatModel
from gnaniassignment.llm.laya_cpp import LayaCppRouter
from gnaniassignment.llm.laya_lambda import LayaLambdaRouter
from gnaniassignment.llm.local_laya import LocalLayaRouter
from gnaniassignment.llm.retail_classifier import HinglishRetailRouter
from gnaniassignment.llm.router import LayaGatewayRouter
from gnaniassignment.llm.retrieval import EmptyRetriever, IntentScopedRetriever
from gnaniassignment.orchestration.graph import build_graph
from gnaniassignment.orchestration.session import TurnSession


def create_intent_router(settings: Settings) -> LayaGatewayRouter | LocalLayaRouter | LayaCppRouter | LayaLambdaRouter | HinglishRetailRouter:
    if settings.intent_provider == "hinglish_retail":
        router = HinglishRetailRouter()
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
    raise ValueError("INTENT_PROVIDER must be 'hinglish_retail', 'local_laya', 'laya_cpp', 'laya_lambda', or 'laya'.")


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


def create_session(
    settings: Settings | None = None,
    retriever: IntentScopedRetriever | EmptyRetriever | None = None,
    knowledge: str = "No approved knowledge has been loaded yet.",
) -> TurnSession:
    load_environment()
    settings = settings or Settings.from_environment()
    router = create_intent_router(settings)
    model = create_chat_model(settings)
    if settings.intent_provider == "hinglish_retail":
        agents = retail_agents(model, knowledge=knowledge, retriever=retriever)
        clarification_agent = retail_clarification_agent(model, knowledge=knowledge, retriever=retriever)
    else:
        agents = enquiry_agents(model, knowledge=knowledge, retriever=retriever)
        clarification_agent = None
    return TurnSession(build_graph(router, agents, threshold=settings.intent_threshold, clarification_agent=clarification_agent))
