"""Opt-in LangSmith spans for custom model adapters."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import TypeVar


T = TypeVar("T")


def _enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "false").strip().lower() in {"1", "true", "yes"} and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


def trace_llm_call(
    *,
    agent: str,
    model: str | None,
    system: str,
    user: str,
    invoke: Callable[[], T],
) -> T:
    """Record exact prompt input and provider output when tracing is enabled."""
    if not _enabled():
        return invoke()
    from langsmith import traceable

    @traceable(name=f"llm:{agent}", run_type="llm", metadata={"model": model or "unknown"})
    def traced(payload: dict[str, str]) -> T:
        return invoke()

    return traced({"system": system, "user": user})


def trace_llm_stream(
    *,
    agent: str,
    model: str | None,
    system: str,
    user: str,
    invoke: Callable[[], Iterator[str]],
) -> Iterator[str]:
    """Keep token streaming intact while recording the final streamed output."""
    if not _enabled():
        yield from invoke()
        return
    from langsmith import traceable

    @traceable(name=f"llm:{agent}", run_type="llm", metadata={"model": model or "unknown"})
    def traced(payload: dict[str, str]) -> Iterator[str]:
        yield from invoke()

    yield from traced({"system": system, "user": user})
