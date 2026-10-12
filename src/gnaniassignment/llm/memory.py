"""Intent-aware memory selection with a bounded prompt budget."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTurn:
    query: str
    answer: str
    intent: str
    topic: str


def topic_from_query(query: str) -> str:
    """Small, fast topic key until semantic retrieval is introduced.

    It makes follow-ups such as 'where do you operate?' and 'which regions?'
    share a memory bucket without another model call.
    """
    lowered = query.lower()
    groups = {
        "coverage": ("where", "region", "country", "operate", "location"),
        "timeline": ("when", "launch", "available", "delivery", "roadmap", "date"),
        "operations": ("how", "process", "work", "support", "implement"),
        "value": ("why", "benefit", "value", "choose", "comparison"),
        "product": ("product", "feature", "capability", "price", "plan"),
    }
    return next((name for name, words in groups.items() if any(word in lowered for word in words)), "general")


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

    selected: list[ChatTurn] = []
    for turn in reversed(history):
        if turn.intent == intent and turn.topic == topic:
            selected.append(turn)
        if len(selected) == 2:
            break
    for turn in reversed(history):
        if turn not in selected:
            selected.append(turn)
        if len(selected) == 4:
            break
    return list(reversed(selected))


def render_memory(turns: list[ChatTurn]) -> str:
    if not turns:
        return "No prior conversation is relevant."
    return "\n".join(f"User: {turn.query}\nAssistant: {turn.answer}" for turn in turns)
