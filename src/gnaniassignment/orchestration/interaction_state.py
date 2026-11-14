"""Validation and delivery-aware storage for LLM-produced dialogue state.

This module deliberately does not classify caller language. The generic LLM
interprets the current utterance in its single normal response; Python only
keeps the resulting compact state after the associated audio is heard.
"""

from __future__ import annotations

from typing import Any


_ACTS = {
    "greeting", "acknowledgement", "confirmation", "correction", "challenge",
    "closing", "explicit_message_request", "private_information_request",
    "request", "other",
}
_RESOLUTIONS = {
    "answered", "acknowledged", "message_captured", "privacy_boundary",
    "clarification_requested", "clarification_resolved", "closed", "limitation",
    "pending", "other",
}
_NEXT_EXPECTED = {
    "none", "correction_or_closing", "confirmation_or_correction",
    "message_or_callback_detail", "caller_response", "other",
}


def normalize_dialogue_state(value: object) -> dict[str, Any]:
    """Accept only a small, non-authoritative state projection from the LLM."""
    if not isinstance(value, dict):
        return {}
    act = value.get("act")
    resolution = value.get("resolution")
    next_expected_act = value.get("next_expected_act")
    clarification = value.get("clarification")
    if act not in _ACTS or resolution not in _RESOLUTIONS:
        return {}
    if next_expected_act not in _NEXT_EXPECTED:
        next_expected_act = "other"
    if clarification not in {"not_needed", "asked", "resolved"}:
        clarification = "not_needed"
    return {
        "act": act,
        "resolution": resolution,
        "clarification": clarification,
        "next_expected_act": next_expected_act,
    }


def merge_delivered_interaction_state(
    previous: dict[str, Any],
    proposed: object,
    delivered_answer: str,
) -> dict[str, Any]:
    """Promote a model state only after its corresponding audio was delivered."""
    state = normalize_dialogue_state(proposed)
    if not state:
        return {}
    prior_count = int(previous.get("clarification_count", 0))
    if state["clarification"] == "asked":
        clarification_count = min(2, prior_count + 1)
    else:
        clarification_count = 0
    return {
        **state,
        "clarification_count": clarification_count,
        "last_audible_answer": " ".join(delivered_answer.split())[:240],
    }


def render_interaction_state(state: dict[str, Any]) -> str:
    """Render last-turn state as context, never as a command over new speech."""
    if not state:
        return ""
    lines = [
        "Previous audible interaction state (context only; the current caller words take priority):",
        f"- previous caller act: {state.get('act', 'other')}",
        f"- previous resolution: {state.get('resolution', 'other')}",
        f"- likely next caller move: {state.get('next_expected_act', 'other')}",
        f"- clarification turns already used for that need: {int(state.get('clarification_count', 0))}",
    ]
    if state.get("clarification_count", 0):
        lines.append("- do not ask that clarification again; resolve, state a limitation, or offer one useful alternative")
    if state.get("last_audible_answer"):
        lines.append(f"- previous audible assistant answer: {state['last_audible_answer']}")
    return "\n".join(lines)


def partial_delivery_context(previous: dict[str, Any], delivered_answer: str) -> dict[str, Any]:
    """Keep partial audio wording without promoting an unheard final state."""
    if not delivered_answer.strip():
        return previous
    return {
        **previous,
        "last_audible_answer": " ".join(delivered_answer.split())[:240],
    }
