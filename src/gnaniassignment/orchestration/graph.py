"""Low-latency LangGraph workflow for a completed enquiry turn."""

from __future__ import annotations

from math import isfinite
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from gnaniassignment.domain.ports import IntentRouter
from gnaniassignment.llm.agents import AgentResponse, SubAgent
from gnaniassignment.llm.memory import ChatTurn, render_memory, select_memory, topic_from_query
from gnaniassignment.llm.prompts.intents import INTENT_KEYS, QUESTIONS

THRESHOLD = 0.5


class TurnState(TypedDict):
    text: str
    answers: dict[str, Any]
    intent: str
    topic: str
    history: list[ChatTurn]
    memory: str
    replies: list[dict[str, str]]
    spoken: str | None


def select_intent(answers: dict[str, Any], threshold: float = THRESHOLD, allowed_intents: tuple[str, ...] = INTENT_KEYS) -> str:
    """Route Laya's Choice result, using its separate confidence value."""
    if not 0 <= threshold <= 1:
        raise ValueError("Intent threshold must be between 0 and 1")
    try:
        decision = answers["intent"]
        choice = decision["choice"]
        confidence = float(decision["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Laya response is missing a valid intent choice or confidence") from exc
    if not isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Laya intent confidence must be between 0 and 1")
    if choice not in (*allowed_intents, "other"):
        raise ValueError(f"Intent router returned an unknown intent choice: {choice!r}")
    return choice if choice != "other" and confidence >= threshold else "clarify"


def build_graph(
    router: IntentRouter,
    agents: dict[str, SubAgent],
    threshold: float = THRESHOLD,
    clarification_agent: SubAgent | None = None,
):
    if not agents:
        raise ValueError("At least one intent agent is required")
    if not 0 <= threshold <= 1:
        raise ValueError("Intent threshold must be between 0 and 1")

    def classify(state: TurnState) -> dict[str, Any]:
        decision_context = {
            "recent_conversation": [turn.summary[:320] for turn in state["history"][-2:]],
            "current_query": state["text"],
        }
        answers = router.predict(decision_context, QUESTIONS)["answers"]
        intent = select_intent(answers, threshold, tuple(agents))
        topic = topic_from_query(state["text"])
        if topic == "general" and state["history"] and state["history"][-1].intent == intent:
            topic = state["history"][-1].topic
        return {"answers": answers, "intent": intent, "topic": topic, "memory": render_memory(select_memory(state["history"], intent, topic))}

    def route(state: TurnState) -> str:
        return state["intent"]

    def settle(state: TurnState) -> dict[str, Any]:
        return {"spoken": state["replies"][0]["text"]}

    def clarify(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        on_chunk = config.get("configurable", {}).get("on_answer_chunk")
        if clarification_agent is None:
            # Kept for callers that construct a graph without a clarification
            # subagent. Production composition always supplies one.
            answer = "Aap order, delivery, payment, product, return, refund ya cancellation ke baare mein kya jaanna chahte hain?"
            if on_chunk is not None:
                on_chunk(answer)
            return {"replies": [{"agent": "clarify", "text": answer, "summary": "The question was unclear; awaiting clarification."}]}
        slots = {"intent": "clarify", "topic": state["topic"]}
        if on_chunk is None:
            output = clarification_agent.run(state["text"], slots, state["memory"])
        else:
            output = clarification_agent.run_stream(state["text"], slots, state["memory"], on_chunk)
        if isinstance(output, AgentResponse):
            return {"replies": [{"agent": clarification_agent.name, "text": output.answer, "summary": output.memory_summary}]}
        return {"replies": [{"agent": clarification_agent.name, "text": output, "summary": f"User asked: {state['text']}. Clarification: {output[:280]}"}]}

    graph = StateGraph(TurnState)
    graph.add_node("classify", classify)
    graph.add_node("clarify", clarify)
    graph.add_node("settle", settle)
    graph.add_edge("clarify", "settle")
    for name, agent in agents.items():
        graph.add_node(name, _agent_node(agent))
        graph.add_edge(name, "settle")
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route, [*agents, "clarify"])
    graph.add_edge("settle", END)
    return graph.compile()


def _agent_node(agent: SubAgent):
    def node(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        on_chunk = config.get("configurable", {}).get("on_answer_chunk")
        slots = {"intent": state["intent"], "topic": state["topic"]}
        if on_chunk is None:
            output = agent.run(state["text"], slots, state["memory"])
        else:
            output = agent.run_stream(state["text"], slots, state["memory"], on_chunk)
        if isinstance(output, AgentResponse):
            return {"replies": [{"agent": agent.name, "text": output.answer, "summary": output.memory_summary}]}
        return {"replies": [{"agent": agent.name, "text": output, "summary": f"User asked: {state['text']}. Answer: {output[:280]}"}]}

    return node
