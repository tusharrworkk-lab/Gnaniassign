"""Small PostgreSQL-backed customer-order context for local demonstrations."""

from __future__ import annotations

import os
import re
from collections.abc import Callable


_ORDER_ID = re.compile(r"\b(?:GRC|ORD)-[A-Z0-9-]{3,}\b", re.I)
_ORDER_REQUEST = re.compile(r"\b(?:order|refund|payment|delivery|return|cancel)\b", re.I)


class PostgresCustomerOrderStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("ORDER_DATA_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
        if not self.database_url:
            raise ValueError("Set ORDER_DATA_DATABASE_URL or EPISODIC_MEMORY_DATABASE_URL to enable order data")

    def render_order_window(self, customer_id: str, query: str = "", active_order_id: str | None = None, limit: int = 4) -> str:
        """Return a small account window, preferring an order named by the caller."""
        requested = [match.upper() for match in _ORDER_ID.findall(query)]
        if active_order_id:
            requested.extend(match.upper() for match in _ORDER_ID.findall(active_order_id))
        requested = list(dict.fromkeys(requested))
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Install PostgreSQL memory support with: uv sync --extra postgres-memory") from error
        with psycopg.connect(self.database_url) as connection, connection.cursor() as cursor:
            if requested:
                cursor.execute(
                    """
                    SELECT order_id, item_name, order_status, payment_status, delivery_note, refund_status
                    FROM customer_orders
                    WHERE customer_id = %s AND order_id = ANY(%s)
                    ORDER BY updated_at DESC, id DESC
                    """,
                    (customer_id, requested),
                )
            else:
                cursor.execute(
                    """
                    SELECT order_id, item_name, order_status, payment_status, delivery_note, refund_status
                    FROM customer_orders
                    WHERE customer_id = %s
                    ORDER BY updated_at DESC, id DESC LIMIT %s
                    """,
                    (customer_id, limit),
                )
            rows = cursor.fetchall()
        if not rows:
            return "No account order records are available. Do not invent an order record."
        records = []
        for order_id, item, status, payment, delivery, refund in rows:
            details = [f"order ID {order_id}", f"item {item}", f"order status {status}", f"payment {payment}"]
            if delivery:
                details.append(f"delivery note {delivery}")
            if refund:
                details.append(f"refund status {refund}")
            records.append("; ".join(details) + ".")
        selected = "matching" if requested else "recent"
        return f"Verified {selected} account order window (use only these records as facts):\n" + "\n".join(records)

    def render_customer_orders(self, customer_id: str) -> str:
        """Backward-compatible full account context for non-conversation callers."""
        return self.render_order_window(customer_id, limit=100)


def customer_order_window(customer_id: str, store: PostgresCustomerOrderStore | None = None) -> Callable[[str, dict[str, str]], str]:
    """Create the per-call account-window provider after caller identification."""
    resolved_store = store or PostgresCustomerOrderStore()
    cached_key: tuple[str, ...] | None = None
    cached_context = ""

    def load(query: str, working_memory: dict[str, str]) -> str:
        nonlocal cached_key, cached_context
        # An explicit ID replaces the older active ID. Ordinary follow-ups
        # reuse the verified window already fetched for this call.
        explicit = tuple(dict.fromkeys(match.upper() for match in _ORDER_ID.findall(query)))
        active = tuple(dict.fromkeys(match.upper() for match in _ORDER_ID.findall(working_memory.get("order_id", ""))))
        key = explicit or active
        if key != cached_key:
            cached_context = resolved_store.render_order_window(
                customer_id, query if explicit else "", None if explicit else (working_memory.get("order_id") or None),
            )
            cached_key = key
        if cached_context.startswith("No account order records") and not key and not _ORDER_REQUEST.search(query):
            return ""
        return cached_context

    return load
