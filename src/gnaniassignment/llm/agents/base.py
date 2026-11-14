"""Base class for focused response agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    memory_summary: str
    memory_operations: tuple[dict[str, Any], ...] = ()


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

    def run_stream(self, text: str, slots: dict[str, str], memory: str, on_chunk: Callable[[str], None]) -> str | AgentResponse:
        """Default streaming behavior for agents without a streaming model."""
        output = self.run(text, slots, memory)
        on_chunk(output.answer if isinstance(output, AgentResponse) else output)
        return output
