"""Intent-scoped knowledge retrieval contracts.

Use one namespace/collection per intent. This keeps retrieval precise without
requiring six separate vector-database deployments.
"""

from __future__ import annotations

from typing import Protocol


class VectorStore(Protocol):
    def search(self, query: str, limit: int = 4) -> list[str]: ...


class IntentScopedRetriever:
    def __init__(self, stores: dict[str, VectorStore]) -> None:
        self.stores = dict(stores)

    def retrieve(self, intent: str, query: str, limit: int = 4) -> list[str]:
        if limit < 1:
            raise ValueError("Retrieval limit must be positive")
        store = self.stores.get(intent)
        return store.search(query, limit=limit)[:limit] if store else []


class EmptyRetriever:
    def retrieve(self, intent: str, query: str, limit: int = 4) -> list[str]:
        return []
