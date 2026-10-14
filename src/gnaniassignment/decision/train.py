"""Train a versioned decision bundle from JSONL examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnaniassignment.decision.model import DecisionCatalog, TfidfDecisionModel


def load_examples(path: str | Path, catalog: DecisionCatalog) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        text, label = str(row.get("text", "")).strip(), str(row.get("intent", ""))
        if not text or label not in catalog.labels:
            raise ValueError(f"Invalid example at {path}:{line_number}")
        texts.append(text)
        labels.append(label)
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    catalog = DecisionCatalog.from_json(args.catalog)
    texts, labels = load_examples(args.data, catalog)
    model = TfidfDecisionModel.train(texts, labels, catalog, args.threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"trained={catalog.name}:{catalog.version} examples={len(texts)} output={output}")


if __name__ == "__main__":
    main()
