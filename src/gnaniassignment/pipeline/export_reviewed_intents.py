"""Export human-reviewed shadow observations in the Colab trainer's JSONL format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from gnaniassignment.llm.retail_minilm import RETAIL_MINILM_KEYS, RETAIL_INSTRUCTION


def split_for(observation_id: int) -> str:
    value = int.from_bytes(hashlib.sha256(str(observation_id).encode()).digest()[:8], "big") / 2**64
    return "train" if value < 0.8 else "validation" if value < 0.9 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reviewed intent labels for Colab fine-tuning")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    database_url = os.getenv("INTENT_FEEDBACK_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
    if not database_url:
        parser.error("Set INTENT_FEEDBACK_DATABASE_URL or EPISODIC_MEMORY_DATABASE_URL")
    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError("Install PostgreSQL memory support with: uv sync --extra postgres-memory") from error
    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, query_text, reviewed_intent FROM intent_shadow_observations WHERE reviewed_intent IS NOT NULL ORDER BY id"
        )
        rows = cursor.fetchall()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for observation_id, query, label in rows:
            row = {
                "id": f"shadow-reviewed-{observation_id}", "state": query,
                "question": {"type": "choice", "instructions": RETAIL_INSTRUCTION, "criteria": {}, "options": list(RETAIL_MINILM_KEYS)},
                "expected": RETAIL_MINILM_KEYS.index(label), "split": split_for(observation_id),
                "source": "human_reviewed_shadow_feedback", "group_id": f"shadow-reviewed-{observation_id}",
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} human-reviewed examples to {args.output}")


if __name__ == "__main__":
    main()
