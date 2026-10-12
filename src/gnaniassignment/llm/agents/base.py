"""Base class for focused response agents."""

from __future__ import annotations


class SubAgent:
    name: str
    system_prompt: str

    def __init__(self) -> None:
        self.calls = 0

    def reply(self, text: str, slots: dict[str, str], memory: str = "") -> str:
        raise NotImplementedError

    def run(self, text: str, slots: dict[str, str], memory: str = "") -> str:
        self.calls += 1
        return self.reply(text, slots, memory)
