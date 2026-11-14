"""Low-latency LangGraph workflow for a completed enquiry turn."""

from __future__ import annotations

import time
from math import isfinite
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from gnaniassignment.domain.ports import IntentRouter
from gnaniassignment.llm.agents import AgentResponse, SubAgent
from gnaniassignment.llm.memory import ChatTurn, render_memory, topic_from_query
from gnaniassignment.orchestration.call_facts import render_call_facts
from gnaniassignment.orchestration.working_memory import render_working_memory, update_working_memory
from gnaniassignment.llm.prompts.intents import INTENT_KEYS, QUESTIONS

THRESHOLD = 0.5


class TurnState(TypedDict):
    session_id: str
    text: str
    answers: dict[str, Any]
    intent: str
    topic: str
    history: list[ChatTurn]
    prior_call_context: list[ChatTurn]
    current_call_history: list[ChatTurn]
    call_memo: str
    call_facts: dict[str, dict[str, Any]]
    working_memory: dict[str, str]
    account_context: str
    memory: str
    replies: list[dict[str, Any]]
    spoken: str | None
    metrics: dict[str, float]


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
        started = time.perf_counter()
        decision_context = {
            # Immediate call context always accompanies the classifier. Short
            # follow-ups such as "haan, bhej do" have no lexical overlap with
            # the preceding message but must resolve against it.
            "recent_conversation": [turn.summary[:320] for turn in state["current_call_history"][-2:]],
            "current_query": state["text"],
            "conversation_session_id": state["session_id"],
        }
        answers = router.predict(decision_context, QUESTIONS)["answers"]
        intent = select_intent(answers, threshold, tuple(agents))
        topic = topic_from_query(state["text"])
        if topic == "general" and state["current_call_history"] and state["current_call_history"][-1].intent == intent:
            topic = state["current_call_history"][-1].topic
        working_memory = update_working_memory(state["working_memory"], state["text"], intent, topic)
        call_status = (
            "Call status: This is a continuing call; the caller has already received the greeting."
            if state["current_call_history"]
            else "Call status: This is the first response of the call."
        )
        # Never inject an older call into the model prompt. The generic agent
        # can access it only through its explicit read-only memory_recall tool.
        prior_memory: list[ChatTurn] = []
        # The rolling memo carries the complete call state. Keep only the
        # latest audible exchange beside it for immediate conversational cues
        # such as "haan" or a correction, avoiding repeated prompt history.
        active_memory = state["current_call_history"][-1:]
        recall_availability = (
            "Prior-call memory: available only through the memory_recall tool."
            if state["prior_call_context"]
            else "Prior-call memory: no completed-call summary is available."
        )
        return {
            "answers": answers,
            "intent": intent,
            "topic": topic,
            "working_memory": working_memory,
            "memory": render_working_memory(working_memory) + "\n\n" + render_call_facts(state["call_facts"]) + "\n\n" + state["account_context"] + "\n\n" + call_status + "\n\n" + recall_availability + "\n\nCall working memo (concise state of this open call):\n" + state["call_memo"] + "\n\nCurrent call context:\n" + render_memory(active_memory) + "\n\nRelevant prior-call context:\n" + render_memory(prior_memory),
            "metrics": {"intent_ms": (time.perf_counter() - started) * 1000},
        }

    def route(state: TurnState) -> str:
        return state["intent"]

    def settle(state: TurnState) -> dict[str, Any]:
        return {"spoken": state["replies"][0]["text"]}

    def clarify(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        on_chunk = config.get("configurable", {}).get("on_answer_chunk")
        if clarification_agent is None:
            # Kept for callers that construct a graph without a clarification
            # subagent. Production composition always supplies one.
            answer = "Aap order, delivery, payment, product, return, refund ya cancellation ke baare mein kya jaanna chahte hain?"
            if on_chunk is not None:
                on_chunk(answer)
            return {"replies": [{"agent": "clarify", "text": answer, "summary": "The question was unclear; awaiting clarification."}], "metrics": {**state["metrics"], "agent_ms": (time.perf_counter() - started) * 1000}}
        slots = {"intent": "clarify", "topic": state["topic"]}
        if on_chunk is None:
            output = clarification_agent.run(state["text"], slots, state["memory"])
        else:
            output = clarification_agent.run_stream(state["text"], slots, state["memory"], on_chunk)
        if isinstance(output, AgentResponse):
            reply = {
                "agent": clarification_agent.name,
                "text": output.answer,
                "summary": output.memory_summary,
                "memory_operations": list(output.memory_operations),
            }
        else:
            reply = {"agent": clarification_agent.name, "text": output, "summary": f"User asked: {state['text']}. Clarification: {output[:280]}"}
        return {"replies": [reply], "metrics": _agent_metrics(state["metrics"], clarification_agent, started)}

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
        started = time.perf_counter()
        on_chunk = config.get("configurable", {}).get("on_answer_chunk")
        slots = {"intent": state["intent"], "topic": state["topic"]}
        if agent.name == "generic" and state["prior_call_context"]:
            # The callable is intentionally kept out of the prompt. The model
            # sees old summaries only after it emits a native tool request.
            slots["memory_recall_tool"] = lambda: {
                "summaries": [turn.summary[:320] for turn in state["prior_call_context"][-2:]],
            }
        if on_chunk is None:
            output = agent.run(state["text"], slots, state["memory"])
        else:
            output = agent.run_stream(state["text"], slots, state["memory"], on_chunk)
        if isinstance(output, AgentResponse):
            reply = {
                "agent": agent.name,
                "text": output.answer,
                "summary": output.memory_summary,
                "memory_operations": list(output.memory_operations),
            }
        else:
            reply = {"agent": agent.name, "text": output, "summary": f"User asked: {state['text']}. Answer: {output[:280]}"}
        return {"replies": [reply], "metrics": _agent_metrics(state["metrics"], agent, started)}

    return node


def _agent_metrics(existing: dict[str, float], agent: SubAgent, started: float) -> dict[str, float]:
    """Attach provider timing exposed by streaming models without coupling ports."""
    metrics = {**existing, "agent_ms": (time.perf_counter() - started) * 1000}
    first_token = getattr(getattr(agent, "model", None), "last_first_token_ms", None)
    stream_total = getattr(getattr(agent, "model", None), "last_stream_ms", None)
    if isinstance(first_token, (int, float)):
        metrics["llm_ttft_ms"] = float(first_token)
    if isinstance(stream_total, (int, float)):
        metrics["llm_stream_ms"] = float(stream_total)
    return metrics
