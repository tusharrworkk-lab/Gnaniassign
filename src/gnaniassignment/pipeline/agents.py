"""Compatibility exports; use :mod:`gnaniassignment.llm.agents` in new code."""

from gnaniassignment.llm.agents.default import AboutAgent, GeneralAgent, MeetingAgent, NoteAgent, default_agents
from gnaniassignment.llm.agents.base import SubAgent

__all__ = ("AboutAgent", "GeneralAgent", "MeetingAgent", "NoteAgent", "SubAgent", "default_agents")
