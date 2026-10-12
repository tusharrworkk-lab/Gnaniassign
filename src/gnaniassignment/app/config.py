"""Runtime settings. Secrets are read only from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    intent_provider: str = "jev"
    intent_threshold: float = 0.5

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            intent_provider=os.getenv("INTENT_PROVIDER", "jev").lower(),
            intent_threshold=float(os.getenv("INTENT_THRESHOLD", "0.5")),
        )
