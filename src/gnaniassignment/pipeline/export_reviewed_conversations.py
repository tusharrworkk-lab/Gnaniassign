"""Export final human-reviewed conversation labels for the Colab MiniLM trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gnaniassignment.app.config import load_environment
from gnaniassignment.conversation_analytics import PostgresConversationStore
from gnaniassignment.llm.retail_minilm import RETAIL_INSTRUCTION, RETAIL_MINILM_KEYS


def split_for(turn_id: int) -> str:
    value = int.from_bytes(hashlib.sha256(str(turn_id).encode()).digest()[:8], "big") / 2**64
    return "train" if value < 0.8 else "validation" if value < 0.9 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final reviewed conversations to trainer JSONL")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    load_environment()
    store = PostgresConversationStore()
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT id, user_text, final_intent FROM conversation_turns WHERE final_intent IS NOT NULL ORDER BY id")
        rows = cursor.fetchall()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for turn_id, user_text, label in rows:
            file.write(json.dumps({
                "id": f"conversation-reviewed-{turn_id}", "state": user_text,
                "question": {"type": "choice", "instructions": RETAIL_INSTRUCTION, "criteria": {}, "options": list(RETAIL_MINILM_KEYS)},
                "expected": RETAIL_MINILM_KEYS.index(label), "split": split_for(turn_id),
                "source": "human_reviewed_conversation", "group_id": f"conversation-reviewed-{turn_id}",
            }, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} reviewed conversation labels to {args.output}")


if __name__ == "__main__":
    main()
