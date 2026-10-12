"""Low-latency LangGraph workflow for a completed enquiry turn."""

from __future__ import annotations

from math import isfinite
from typing import Any, TypedDict

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


def select_intent(answers: dict[str, Any], threshold: float = THRESHOLD) -> str:
    """Choose one route; ask for clarification if Jev is uncertain."""
    if not 0 <= threshold <= 1:
        raise ValueError("Intent threshold must be between 0 and 1")
    try:
        scores = {name: float(answers[name]["noul"]) for name in INTENT_KEYS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Jev response is missing a valid score for an intent") from exc
    if any(not isfinite(score) or not 0 <= score <= 1 for score in scores.values()):
        raise ValueError("Jev intent scores must be finite numbers between 0 and 1")
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] >= threshold else "clarify"


def build_graph(router: IntentRouter, agents: dict[str, SubAgent], threshold: float = THRESHOLD):
    missing = set(INTENT_KEYS) - set(agents)
    if missing:
        raise ValueError(f"Missing agents for intents: {', '.join(sorted(missing))}")
    if not 0 <= threshold <= 1:
        raise ValueError("Intent threshold must be between 0 and 1")

    def classify(state: TurnState) -> dict[str, Any]:
        recent = render_memory(state["history"][-2:])
        decision_context = f"Recent conversation:\n{recent}\nCurrent user query: {state['text']}"
        answers = router.predict(decision_context, QUESTIONS, model="jev")["answers"]
        intent = select_intent(answers, threshold)
        topic = topic_from_query(state["text"])
        if topic == "general" and state["history"] and state["history"][-1].intent == intent:
            topic = state["history"][-1].topic
        return {"answers": answers, "intent": intent, "topic": topic, "memory": render_memory(select_memory(state["history"], intent, topic))}

    def route(state: TurnState) -> str:
        return state["intent"]

    def settle(state: TurnState) -> dict[str, Any]:
        return {"spoken": state["replies"][0]["text"]}

    def clarify(state: TurnState) -> dict[str, Any]:
        answer = "Could you clarify whether you mean the company or a product, and what you would like to know?"
        return {"replies": [{"agent": "clarify", "text": answer, "summary": "The question was unclear; awaiting clarification."}]}

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
    def node(state: TurnState) -> dict[str, Any]:
        output = agent.run(state["text"], {"intent": state["intent"], "topic": state["topic"]}, state["memory"])
        if isinstance(output, AgentResponse):
            return {"replies": [{"agent": agent.name, "text": output.answer, "summary": output.memory_summary}]}
        return {"replies": [{"agent": agent.name, "text": output, "summary": f"User asked: {state['text']}. Answer: {output[:280]}"}]}

    return node
