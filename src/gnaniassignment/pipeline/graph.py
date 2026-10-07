"""One LangGraph turn step.

A final phrase, or the pause at the end of the turn, is one invoke.
Laya reads the whole locked sentence. The subagent for each intent drafts
a reply. If the intent and the slots are unchanged, the previous draft is kept.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from gnaniassignment.pipeline.agents import SubAgent, default_agents

INTENT_KEYS = ("book_meeting", "take_note", "about_me")
THRESHOLD = 0.5
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_PERSON = re.compile(r"\bwith\s+([A-Za-z]+)")
_PERSON_SKIP = {"the", "a", "me", "us", "them", "you", "him", "her"}

QUESTIONS = {
    "book_meeting": {
        "type": "noul",
        "instructions": "Does the caller want to book, move, or cancel a meeting?",
    },
    "take_note": {
        "type": "noul",
        "instructions": "Should this turn be saved as a note?",
    },
    "about_me": {
        "type": "noul",
        "instructions": "Is the caller asking who the user is or what they do?",
    },
}


class TurnState(TypedDict):
    text: str
    event: str
    draft: dict[str, Any] | None
    answers: dict[str, Any]
    intents: list[str]
    slots: dict[str, str]
    reused: bool
    replies: Annotated[list[dict[str, str]], operator.add]
    spoken: str | None


def slots_from_text(text: str) -> dict[str, str]:
    lowered = text.lower()
    day = next((name for name in DAYS if name in lowered), "none")
    person = "none"
    match = _PERSON.search(text)
    if match and match.group(1).lower() not in _PERSON_SKIP:
        person = match.group(1).lower()
    return {"day": day, "person": person}


def same_decision(draft: dict[str, Any] | None, intents: list[str], slots: dict[str, str]) -> bool:
    if not draft:
        return False
    return draft["intents"] == intents and draft["slots"] == slots


def build_graph(router: Any, agents: dict[str, SubAgent] | None = None, threshold: float = THRESHOLD):
    agents = agents or default_agents()

    def decide(state: TurnState) -> dict[str, Any]:
        result = router.predict(f"caller: {state['text']}", QUESTIONS, model="multilingual")
        answers = result["answers"]
        intents = [
            name
            for name in INTENT_KEYS
            if float(answers[name]["noul"]) >= threshold
        ]
        if not intents:
            intents = ["general"]
        return {
            "answers": answers,
            "intents": intents,
            "slots": slots_from_text(state["text"]),
        }

    def route(state: TurnState) -> str | list[Send]:
        if same_decision(state["draft"], state["intents"], state["slots"]):
            return "reuse"
        return [Send(name, state) for name in state["intents"]]

    def reuse(state: TurnState) -> dict[str, Any]:
        previous = state["draft"] or {}
        return {"replies": list(previous.get("replies", [])), "reused": True}

    def settle(state: TurnState) -> dict[str, Any]:
        draft = {
            "intents": list(state["intents"]),
            "slots": dict(state["slots"]),
            "replies": list(state["replies"]),
            "text": state["text"],
        }
        spoken = None
        if state["event"] == "end":
            spoken = " ".join(item["text"] for item in state["replies"])
        return {"draft": draft, "spoken": spoken}

    graph = StateGraph(TurnState)
    graph.add_node("decide", decide)
    graph.add_node("reuse", reuse)
    graph.add_node("settle", settle)
    for name, agent in agents.items():
        graph.add_node(name, _agent_node(agent))

    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide",
        route,
        ["reuse", *agents.keys()],
    )
    graph.add_edge("reuse", "settle")
    for name in agents:
        graph.add_edge(name, "settle")
    graph.add_edge("settle", END)
    return graph.compile()


def _agent_node(agent: SubAgent):
    def node(state: TurnState) -> dict[str, Any]:
        text = agent.run(state["text"], state["slots"])
        return {"replies": [{"agent": agent.name, "text": text}]}

    return node


def load_laya():
    """One multilingual checkpoint. FP16 so it fits the 1650 Ti."""
    import torch
    from laya import Router

    device = "cuda" if torch.cuda.is_available() else "cpu"
    router = Router(device=device, max_loaded=1, default="multilingual")
    router.preload(["multilingual"])
    if device == "cuda":
        router._agents["multilingual"].model.half()
    return router
