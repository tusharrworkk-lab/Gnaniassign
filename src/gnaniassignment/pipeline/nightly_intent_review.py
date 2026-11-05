"""Classify unlabelled completed turns in a nightly MiniLM review batch.

This job deliberately runs outside the live voice path. Predictions remain
candidate labels until a person reviews them through review_conversations.py.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import dataclass

import requests

from gnaniassignment.app.config import load_environment
from gnaniassignment.llm.retail_minilm import RETAIL_MINILM_KEYS, RetailMiniLMRouter


@dataclass(frozen=True)
class ClassifiedTurn:
    turn_id: int
    text: str
    intent: str
    confidence: float


def _telegram_chunks(turns: list[ClassifiedTurn], max_items: int, max_chars: int = 3900) -> list[str]:
    """Format a bounded, human-readable private-channel review digest."""
    totals = Counter(turn.intent for turn in turns)
    header = "Nightly MiniLM review\n"
    header += f"Classified: {len(turns)} turns\n"
    header += "Labels: " + ", ".join(f"{label}={totals[label]}" for label in RETAIL_MINILM_KEYS if totals[label])
    selected = turns[:max_items]
    if not selected:
        return [header + "\n\nNo newly unclassified turns were found."]

    chunks: list[str] = []
    current = header + "\n\nReview cards (candidate labels only):"
    for turn in selected:
        text = " ".join(turn.text.split())
        card = f"\n\n#{turn.turn_id}  {turn.intent} ({turn.confidence:.1%})\n{text}"
        if len(current) + len(card) > max_chars and current:
            chunks.append(current)
            current = "Nightly MiniLM review (continued):" + card
        else:
            current += card
    chunks.append(current)
    if len(turns) > len(selected):
        chunks[-1] += f"\n\nOnly the first {len(selected)} cards were sent; {len(turns) - len(selected)} remain in PostgreSQL."
    return chunks


def _send_telegram(messages: list[str]) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_REVIEW_CHAT_ID")
    if not token or not chat_id:
        raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_REVIEW_CHAT_ID to send Telegram review cards")
    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    for message in messages:
        response = requests.post(endpoint, json={"chat_id": chat_id, "text": message}, timeout=15)
        response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly offline MiniLM intent classification and optional Telegram digest")
    parser.add_argument("--limit", type=int, default=500, help="Maximum previously unclassified turns to process")
    parser.add_argument("--telegram", action="store_true", help="Send private-channel review cards after classification")
    parser.add_argument("--telegram-max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Classify and print counts without updating PostgreSQL or Telegram")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    load_environment()
    if args.telegram_max_items is None:
        args.telegram_max_items = int(os.getenv("TELEGRAM_REVIEW_MAX_ITEMS", "20"))
    if args.telegram_max_items < 1:
        parser.error("--telegram-max-items must be positive")
    database_url = os.getenv("CONVERSATION_LOG_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
    if not database_url:
        parser.error("Set CONVERSATION_LOG_DATABASE_URL or EPISODIC_MEMORY_DATABASE_URL")
    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError("Install PostgreSQL support with: uv sync --extra postgres-memory") from error

    router = RetailMiniLMRouter()
    router.warmup()  # One model load per job, never once per stored turn.
    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, user_text
            FROM conversation_turns
            WHERE classifier_intent IS NULL
            ORDER BY created_at, id
            LIMIT %s
            """,
            (args.limit,),
        )
        source_rows = cursor.fetchall()
        classified: list[ClassifiedTurn] = []
        for turn_id, text in source_rows:
            result = router.predict(text, {})
            decision = result["answers"]["intent"]
            label, confidence = decision["choice"], float(decision["confidence"])
            if label not in RETAIL_MINILM_KEYS:
                raise RuntimeError(f"Unexpected MiniLM label: {label!r}")
            classified.append(ClassifiedTurn(turn_id, text, label, confidence))
        if not args.dry_run:
            cursor.executemany(
                "UPDATE conversation_turns SET classifier_intent = %s, classifier_confidence = %s WHERE id = %s",
                [(turn.intent, turn.confidence, turn.turn_id) for turn in classified],
            )
            connection.commit()

    counts = Counter(turn.intent for turn in classified)
    print(f"Classified {len(classified)} turns" + (" (dry run)" if args.dry_run else ""))
    print(", ".join(f"{label}={counts[label]}" for label in RETAIL_MINILM_KEYS if counts[label]) or "No pending turns")
    if args.telegram and not args.dry_run:
        _send_telegram(_telegram_chunks(classified, args.telegram_max_items))
        print("Telegram review digest sent")


if __name__ == "__main__":
    main()
