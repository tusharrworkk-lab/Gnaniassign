"""Review MiniLM shadow predictions before using them as training labels."""

from __future__ import annotations

import argparse
import os

from gnaniassignment.llm.retail_minilm import RETAIL_MINILM_KEYS


def main() -> None:
    parser = argparse.ArgumentParser(description="Label unreviewed MiniLM shadow predictions")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    database_url = os.getenv("INTENT_FEEDBACK_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
    if not database_url:
        parser.error("Set INTENT_FEEDBACK_DATABASE_URL or EPISODIC_MEMORY_DATABASE_URL")
    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError("Install PostgreSQL memory support with: uv sync --extra postgres-memory") from error
    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, query_text, predicted_intent, confidence
            FROM intent_shadow_observations
            WHERE reviewed_intent IS NULL
            ORDER BY created_at
            LIMIT %s
            """,
            (args.limit,),
        )
        rows = cursor.fetchall()
        for observation_id, query, predicted, confidence in rows:
            print(f"\n#{observation_id}  predicted={predicted}  confidence={confidence:.1%}\n{query}")
            print(" / ".join(f"{index + 1}:{label}" for index, label in enumerate(RETAIL_MINILM_KEYS)))
            response = input("Correct label (Enter=accept, s=skip): ").strip().lower()
            if response == "s":
                continue
            label = predicted if not response else RETAIL_MINILM_KEYS[int(response) - 1] if response.isdigit() and 1 <= int(response) <= len(RETAIL_MINILM_KEYS) else response
            if label not in RETAIL_MINILM_KEYS:
                print("Skipped: use a number, an exact label, Enter, or s.")
                continue
            cursor.execute(
                "UPDATE intent_shadow_observations SET reviewed_intent = %s, reviewed_at = now() WHERE id = %s",
                (label, observation_id),
            )
            print(f"Saved: {label}")
    print(f"Reviewed {len(rows)} queued observations.")


if __name__ == "__main__":
    main()
