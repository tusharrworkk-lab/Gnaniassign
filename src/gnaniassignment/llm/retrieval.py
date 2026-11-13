"""Intent-scoped knowledge retrieval contracts.

Use one namespace/collection per intent. This keeps retrieval precise without
requiring six separate vector-database deployments.
"""

from __future__ import annotations

from typing import Protocol
import os
import re


_QUERY_FILLER = {
    "aap", "aapka", "aur", "bata", "batao", "haan", "hai", "hain", "hello", "kya",
    "main", "mera", "meri", "mujhe", "nahi", "please", "the", "this", "what",
    "when", "where", "with", "you",
}


def _knowledge_search_query(query: str) -> str:
    """Build a bounded OR query for short English and Roman-Hinglish turns."""
    words = re.findall(r"[^\W_]{3,}", query.casefold(), flags=re.UNICODE)
    terms = list(dict.fromkeys(word for word in words if word not in _QUERY_FILLER))[:8]
    return " OR ".join(terms)


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


class PostgresKnowledgeStore:
    """Intent-scoped approved knowledge chunks stored in PostgreSQL."""

    def __init__(self, intent: str, database_url: str | None = None) -> None:
        self.intent = intent
        self.database_url = database_url or os.getenv("KNOWLEDGE_BASE_DATABASE_URL")
        if not self.database_url:
            raise ValueError("Set KNOWLEDGE_BASE_DATABASE_URL to enable PostgreSQL knowledge retrieval")

    def search(self, query: str, limit: int = 4) -> list[str]:
        if limit < 1:
            raise ValueError("limit must be positive")
        search_query = _knowledge_search_query(query)
        if not search_query:
            return []
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Install PostgreSQL memory support with: uv sync --extra postgres-memory") from error
        with psycopg.connect(self.database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT content FROM approved_knowledge_chunks
                WHERE (%s = '*' OR intent = %s) AND status = 'approved'
                  AND to_tsvector('simple', title || ' ' || content)
                      @@ websearch_to_tsquery('simple', %s)
                ORDER BY ts_rank_cd(
                    to_tsvector('simple', title || ' ' || content),
                    websearch_to_tsquery('simple', %s)
                ) DESC, priority DESC, updated_at DESC, id DESC
                LIMIT %s
                """,
                (self.intent, self.intent, search_query, search_query, limit),
            )
            return [row[0] for row in cursor.fetchall()]
