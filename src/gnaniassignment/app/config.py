"""Runtime settings. Secrets are read only from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


def load_environment() -> None:
    """Load a nearby .env while preserving variables set by the host."""
    load_dotenv(find_dotenv(usecwd=True), override=False)


@dataclass(frozen=True)
class Settings:
    intent_provider: str = "laya"
    intent_threshold: float = 0.5
    llm_provider: str = "vercel"

    @classmethod
    def from_environment(cls) -> "Settings":
        load_environment()
        intent_provider = os.getenv("INTENT_PROVIDER", "laya").lower()
        # Turning off MiniLM bypasses *all* intent classification. Every turn
        # then goes directly to one general retail-support agent.
        if intent_provider == "retail_minilm" and os.getenv("RETAIL_MINILM_ENABLED", "true").lower() not in {"1", "true", "yes", "on"}:
            intent_provider = "generic"
        threshold_default = "0.12" if intent_provider == "hinglish_retail" else "0.60" if intent_provider == "retail_minilm" else "0.5"
        return cls(
            intent_provider=intent_provider,
            intent_threshold=float(os.getenv("RETAIL_MINILM_THRESHOLD", os.getenv("RETAIL_INTENT_THRESHOLD", os.getenv("INTENT_THRESHOLD", threshold_default)))),
            llm_provider=os.getenv("LLM_PROVIDER", "vercel").lower(),
        )
