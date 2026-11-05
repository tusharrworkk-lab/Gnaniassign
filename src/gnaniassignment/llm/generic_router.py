"""Router used when intent classification is deliberately disabled."""

from __future__ import annotations

import logging
from threading import Thread
from typing import Any, Callable

from gnaniassignment.domain.ports import IntentRouter


logger = logging.getLogger(__name__)


class GenericRouter:
    """Send every turn to the single general-support agent without inference."""

    def __init__(
        self,
        shadow_router: IntentRouter | None = None,
        on_shadow_prediction: Callable[[str, dict[str, Any], str | None], None] | None = None,
        on_shadow_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.shadow_router = shadow_router
        self.on_shadow_prediction = on_shadow_prediction
        self.on_shadow_error = on_shadow_error
        self.last_ms = 0.0
        self.last_headers_ms: float | None = None

    def predict(
        self, state: str | dict[str, Any], questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        if self.shadow_router is not None and self.on_shadow_prediction is not None:
            query = state.get("current_query", "") if isinstance(state, dict) else state
            if isinstance(query, str) and query.strip():
                # Shadow inference must never delay or alter the live generic response.
                session_id = state.get("conversation_session_id") if isinstance(state, dict) else None
                Thread(target=self._record_shadow, args=(query, state, questions, session_id), daemon=True).start()
        return {"answers": {"intent": {"type": "choice", "choice": "generic", "confidence": 1.0}}}

    def _record_shadow(self, query: str, state: str | dict[str, Any], questions: dict[str, Any], session_id: str | None) -> None:
        try:
            prediction = self.shadow_router.predict(state, questions)
            self.on_shadow_prediction(query, prediction, session_id)
        except Exception as error:
            # Observability cannot make a customer-facing generic turn fail.
            logger.exception("Shadow MiniLM prediction was not recorded")
            if self.on_shadow_error is not None:
                self.on_shadow_error(error)
