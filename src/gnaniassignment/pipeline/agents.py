"""Subagents. Each one is created once and keeps its own system prompt."""

from __future__ import annotations


class SubAgent:
    name: str
    system_prompt: str

    def __init__(self) -> None:
        self.calls = 0

    def reply(self, text: str, slots: dict[str, str]) -> str:
        raise NotImplementedError

    def run(self, text: str, slots: dict[str, str]) -> str:
        self.calls += 1
        return self.reply(text, slots)


class MeetingAgent(SubAgent):
    name = "book_meeting"
    system_prompt = (
        "You only schedule meetings. Use the day and the person when they are "
        "known. If the day is missing, ask for it. Do not take notes and do not "
        "describe the user."
    )

    def reply(self, text: str, slots: dict[str, str]) -> str:
        day = slots.get("day", "none")
        person = slots.get("person", "none")
        if day == "none":
            return "I can book that. Which day?"
        who = f" with {person}" if person != "none" else ""
        return f"I can hold {day}{who}. I will confirm before it is on the calendar."


class NoteAgent(SubAgent):
    name = "take_note"
    system_prompt = (
        "You only write a note of what the caller just said. Do not book a meeting."
    )

    def reply(self, text: str, slots: dict[str, str]) -> str:
        return f"Noted: {text}"


class AboutAgent(SubAgent):
    name = "about_me"
    system_prompt = (
        "You answer from this card only: a cybersecurity software engineer, "
        "open to a short introductory call. Do not invent details."
    )

    def reply(self, text: str, slots: dict[str, str]) -> str:
        return "They are a cybersecurity software engineer, and a short introductory call is fine."


class GeneralAgent(SubAgent):
    name = "general"
    system_prompt = "You take a message when no specific intent is clear. Ask what they need."

    def reply(self, text: str, slots: dict[str, str]) -> str:
        return "I can take a message or book a time. What should I pass on?"


def default_agents() -> dict[str, SubAgent]:
    agents = (MeetingAgent(), NoteAgent(), AboutAgent(), GeneralAgent())
    return {agent.name: agent for agent in agents}
