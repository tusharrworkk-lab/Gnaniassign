"""Base class for focused response agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    memory_summary: str


class SubAgent:
    name: str
    system_prompt: str

    def __init__(self) -> None:
        self.calls = 0

    def reply(self, text: str, slots: dict[str, str], memory: str = "") -> str | AgentResponse:
        raise NotImplementedError

    def run(self, text: str, slots: dict[str, str], memory: str = "") -> str | AgentResponse:
        self.calls += 1
        return self.reply(text, slots, memory)
