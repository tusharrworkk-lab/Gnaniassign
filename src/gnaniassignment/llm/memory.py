"""Intent-aware memory selection with a bounded prompt budget."""

from __future__ import annotations

from dataclasses import dataclass
import re

MAX_MEMORY_TURNS = 4
MAX_SUMMARY_CHARS = 320


@dataclass(frozen=True)
class ChatTurn:
    query: str
    answer: str
    summary: str
    intent: str
    topic: str


def topic_from_query(query: str) -> str:
    """Small, fast topic key until semantic retrieval is introduced.

    It makes follow-ups such as 'where do you operate?' and 'which regions?'
    share a memory bucket without another model call.
    """
    tokens = set(re.findall(r"[\w]+", query.casefold()))
    groups = {
        "coverage": {"where", "region", "regions", "country", "countries", "operate", "location", "locations"},
        "timeline": {"when", "launch", "available", "delivery", "roadmap", "date"},
        "operations": {"how", "process", "work", "support", "implement"},
        "value": {"why", "benefit", "benefits", "value", "choose", "comparison"},
        "product": {"product", "products", "feature", "features", "capability", "price", "plan"},
    }
    return next((name for name, words in groups.items() if tokens & words), "general")


def select_memory(history: list[ChatTurn], intent: str, topic: str) -> list[ChatTurn]:
    """Return only relevant history: same topic first, then the latest turns.

    A run of same-intent turns needs no duplicate context: its last three turns
    are both the episodic and long-term context. Otherwise use two relevant
    turns plus two most recent turns, capped at four total.
    """
    same_intent_run: list[ChatTurn] = []
    for turn in reversed(history):
        if turn.intent != intent:
            break
        same_intent_run.append(turn)
    if len(same_intent_run) >= 3:
        return list(reversed(same_intent_run[:3]))

    selected_indices: set[int] = set()
    for index in range(len(history) - 1, -1, -1):
        turn = history[index]
        if turn.intent == intent and turn.topic == topic:
            selected_indices.add(index)
        if len(selected_indices) == 2:
            break
    for index in range(len(history) - 1, -1, -1):
        selected_indices.add(index)
        if len(selected_indices) == MAX_MEMORY_TURNS:
            break
    return [history[index] for index in sorted(selected_indices)]


def render_memory(turns: list[ChatTurn]) -> str:
    if not turns:
        return "No prior conversation is relevant."
    return "\n".join(f"Memory: {turn.summary[:MAX_SUMMARY_CHARS]}" for turn in turns)
