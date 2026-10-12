"""Provider-independent objects used throughout a conversation turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TurnEvent(StrEnum):
    FINAL = "final"
    END = "end"


@dataclass(frozen=True)
class TurnInput:
    text: str
    event: TurnEvent


@dataclass(frozen=True)
class AgentReply:
    agent: str
    text: str


@dataclass
class TurnResult:
    draft: dict[str, Any] | None = None
    answers: dict[str, Any] = field(default_factory=dict)
    replies: list[AgentReply] = field(default_factory=list)
    spoken: str | None = None
    reused: bool = False
