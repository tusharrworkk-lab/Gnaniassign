"""Score exported voice-turn metadata and optional labelled STT samples.

Example:
  uv run python -m gnaniassignment.pipeline.voice_eval_report \
    --turns evals/voice/turns.jsonl --stt-labels evals/voice/stt-labelled.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gnaniassignment.telephony.evaluation import summarize_turns, summarize_wer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=Path, required=True, help="Metadata-only exported turn records in JSONL format.")
    parser.add_argument("--stt-labels", type=Path, help="Optional local JSONL with reference_text and hypothesis_text.")
    parser.add_argument("--silent-after-ms", type=int, default=2500)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report.")
    args = parser.parse_args()
    report: dict[str, Any] = {"voice": summarize_turns(read_jsonl(args.turns), args.silent_after_ms)}
    if args.stt_labels:
        report["hinglish_stt"] = summarize_wer(read_jsonl(args.stt_labels))
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
