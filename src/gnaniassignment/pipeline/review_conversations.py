"""Human-review voice conversations and assign final classifier labels."""

from __future__ import annotations

import argparse

from gnaniassignment.app.config import load_environment
from gnaniassignment.conversation_analytics import PostgresConversationStore
from gnaniassignment.llm.retail_minilm import RETAIL_MINILM_KEYS


def main() -> None:
    parser = argparse.ArgumentParser(description="Review recorded conversations for intent training")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--flagged-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    load_environment()
    store = PostgresConversationStore()
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, user_text, response_text, next_user_text, classifier_intent,
                   classifier_confidence, flow_flags
            FROM conversation_turns
            WHERE final_intent IS NULL AND (%s = false OR flow_status = 'needs_review')
            ORDER BY created_at
            LIMIT %s
            """,
            (args.flagged_only, args.limit),
        )
        rows = cursor.fetchall()
    for turn_id, user_text, response, followup, predicted, confidence, flags in rows:
        score = f" ({confidence:.1%})" if isinstance(confidence, float) else ""
        print(f"\n#{turn_id} predicted={predicted or 'none'}{score} flags={', '.join(flags) or 'none'}")
        print(f"user: {user_text}\nassistant: {response}")
        if followup:
            print(f"next user reply: {followup}")
        print(" / ".join(f"{index + 1}:{label}" for index, label in enumerate(RETAIL_MINILM_KEYS)))
        choice = input("Final label (Enter=accept prediction, s=skip): ").strip().lower()
        if choice == "s":
            continue
        label = predicted if not choice else RETAIL_MINILM_KEYS[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(RETAIL_MINILM_KEYS) else choice
        if label not in RETAIL_MINILM_KEYS:
            print("Skipped: enter a number, exact label, Enter, or s.")
            continue
        store.review(turn_id, label)
        print(f"Saved: {label}")
    print(f"Reviewed queue size: {len(rows)}")


if __name__ == "__main__":
    main()
