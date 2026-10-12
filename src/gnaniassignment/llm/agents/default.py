"""Deterministic response agents; replaceable by generative LLM-backed agents."""

from __future__ import annotations

from gnaniassignment.llm.agents.base import SubAgent
from gnaniassignment.llm.prompts.agents import ABOUT, GENERAL, MEETING, NOTE


class MeetingAgent(SubAgent):
    name = "book_meeting"
    system_prompt = MEETING

    def reply(self, text: str, slots: dict[str, str], memory: str = "") -> str:
        day, person = slots.get("day", "none"), slots.get("person", "none")
        if day == "none":
            return "I can book that. Which day?"
        who = f" with {person}" if person != "none" else ""
        return f"I can hold {day}{who}. I will confirm before it is on the calendar."


class NoteAgent(SubAgent):
    name = "take_note"
    system_prompt = NOTE

    def reply(self, text: str, slots: dict[str, str], memory: str = "") -> str:
        return f"Noted: {text}"


class AboutAgent(SubAgent):
    name = "about_me"
    system_prompt = ABOUT

    def reply(self, text: str, slots: dict[str, str], memory: str = "") -> str:
        return "They are a cybersecurity software engineer, and a short introductory call is fine."


class GeneralAgent(SubAgent):
    name = "general"
    system_prompt = GENERAL

    def reply(self, text: str, slots: dict[str, str], memory: str = "") -> str:
        return "I can take a message or book a time. What should I pass on?"


def default_agents() -> dict[str, SubAgent]:
    agents = (MeetingAgent(), NoteAgent(), AboutAgent(), GeneralAgent())
    return {agent.name: agent for agent in agents}
