"""Interfaces that let providers change without changing turn logic."""

from __future__ import annotations

from typing import Any, Protocol


class IntentRouter(Protocol):
    def predict(self, state: str | dict[str, Any], questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        """Return typed answers for the supplied state and questions."""
