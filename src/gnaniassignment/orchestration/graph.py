"""Low-latency LangGraph workflow for a completed enquiry turn."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from gnaniassignment.domain.ports import IntentRouter
from gnaniassignment.llm.agents import SubAgent
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
    """Pick exactly one route, so a query triggers only one subagent call."""
    scores = {name: float(answers[name]["noul"]) for name in INTENT_KEYS}
    return max(scores, key=scores.__getitem__)


def build_graph(router: IntentRouter, agents: dict[str, SubAgent], threshold: float = THRESHOLD):
    missing = set(INTENT_KEYS) - set(agents)
    if missing:
        raise ValueError(f"Missing agents for intents: {', '.join(sorted(missing))}")

    def classify(state: TurnState) -> dict[str, Any]:
        answers = router.predict(f"user query: {state['text']}", QUESTIONS, model="jev")["answers"]
        intent = select_intent(answers, threshold)
        topic = topic_from_query(state["text"])
        return {"answers": answers, "intent": intent, "topic": topic, "memory": render_memory(select_memory(state["history"], intent, topic))}

    def route(state: TurnState) -> str:
        return state["intent"]

    def settle(state: TurnState) -> dict[str, Any]:
        return {"spoken": state["replies"][0]["text"]}

    graph = StateGraph(TurnState)
    graph.add_node("classify", classify)
    graph.add_node("settle", settle)
    for name, agent in agents.items():
        graph.add_node(name, _agent_node(agent))
        graph.add_edge(name, "settle")
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route, list(agents))
    graph.add_edge("settle", END)
    return graph.compile()


def _agent_node(agent: SubAgent):
    def node(state: TurnState) -> dict[str, Any]:
        answer = agent.run(state["text"], {"intent": state["intent"], "topic": state["topic"]}, state["memory"])
        return {"replies": [{"agent": agent.name, "text": answer}]}

    return node
