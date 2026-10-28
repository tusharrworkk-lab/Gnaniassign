"""List and explicitly activate reviewed conversation-pattern rules."""

from __future__ import annotations

import argparse

from gnaniassignment.app.config import load_environment
from gnaniassignment.conversation_analytics import PostgresConversationStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Bedrock conversation-pattern rule candidates")
    parser.add_argument("--activate", type=int, help="Activate this reviewed candidate rule")
    args = parser.parse_args()
    load_environment()
    store = PostgresConversationStore()
    with store._connect() as connection, connection.cursor() as cursor:
        if args.activate is not None:
            cursor.execute(
                "UPDATE conversation_pattern_rules SET status = 'active', activated_at = now() WHERE id = %s AND status = 'candidate'",
                (args.activate,),
            )
            if cursor.rowcount != 1:
                parser.error("Candidate rule was not found or is no longer pending")
        cursor.execute("SELECT id, rule_kind, rule_text, status FROM conversation_pattern_rules ORDER BY created_at DESC, id DESC")
        rows = cursor.fetchall()
    for rule_id, kind, text, status in rows:
        print(f"#{rule_id} [{status}] {kind.upper()}: {text}")


if __name__ == "__main__":
    main()
