"""Fine-tune ModernBERT for the versioned retail voice-intent contract.

The script deliberately avoids Hugging Face Trainer so it remains predictable
on a 4 GB GTX 1650 Ti: small batches, fp16 on CUDA, and gradient accumulation.
It saves a standard Hugging Face checkpoint directly loadable by the live
``HinglishRetailRouter``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gnaniassignment.decision.model import DecisionCatalog


DEFAULT_MODEL = "answerdotai/ModernBERT-base"


@dataclass(frozen=True)
class Example:
    text: str
    label: str
    group_id: str


def load_examples(path: str | Path, catalog: DecisionCatalog) -> list[Example]:
    examples: list[Example] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("text", "")).strip()
        label = str(row.get("intent", ""))
        group_id = str(row.get("group_id", "")).strip()
        if not text or label not in catalog.labels or not group_id:
            raise ValueError(f"Invalid example at {path}:{line_number}; text, intent, and group_id are required")
        examples.append(Example(text, label, group_id))
    if not examples:
        raise ValueError("Dataset is empty")
    return examples


def split_by_group(examples: list[Example], eval_fraction: float, seed: int) -> tuple[list[Example], list[Example]]:
    """Deterministically split complete source/paraphrase groups per label."""
    grouped: dict[str, dict[str, list[Example]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        grouped[example.label][example.group_id].append(example)
    train, evaluation = [], []
    for label, groups in grouped.items():
        group_ids = list(groups)
        random.Random(f"{seed}:{label}").shuffle(group_ids)
        evaluation_count = max(1, round(len(group_ids) * eval_fraction)) if len(group_ids) > 1 else 0
        evaluation_ids = set(group_ids[:evaluation_count])
        for group_id, rows in groups.items():
            (evaluation if group_id in evaluation_ids else train).extend(rows)
    if not train or not evaluation:
        raise ValueError("Need at least two distinct groups in at least one label to create train and evaluation splits")
    return train, evaluation


def macro_f1(predictions: list[int], labels: list[int], label_count: int) -> float:
    scores = []
    for label in range(label_count):
        true_positive = sum(pred == label and actual == label for pred, actual in zip(predictions, labels, strict=True))
        false_positive = sum(pred == label and actual != label for pred, actual in zip(predictions, labels, strict=True))
        false_negative = sum(pred != label and actual == label for pred, actual in zip(predictions, labels, strict=True))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return sum(scores) / label_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not 0 < args.eval_fraction < 0.5:
        parser.error("--eval-fraction must be between 0 and 0.5")
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install training dependencies with: uv sync --extra retail-classifier") from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested but CUDA is unavailable")

    catalog = DecisionCatalog.from_json(args.catalog)
    examples = load_examples(args.data, catalog)
    train_examples, eval_examples = split_by_group(examples, args.eval_fraction, args.seed)
    label_to_id = {label: index for index, label in enumerate(catalog.labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class IntentDataset(Dataset):
        def __init__(self, rows: list[Example]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.rows[index]
            encoded = tokenizer(row.text, truncation=True, max_length=args.max_length)
            encoded["labels"] = label_to_id[row.label]
            return encoded

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        labels = torch.tensor([row.pop("labels") for row in rows], dtype=torch.long)
        return {**tokenizer.pad(rows, padding=True, return_tensors="pt"), "labels": labels}

    torch.manual_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(catalog.labels), id2label=id_to_label, label2id=label_to_id
    ).to(args.device)
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(IntentDataset(train_examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(IntentDataset(eval_examples), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                loss = model(**batch).loss / args.grad_accumulation
            scaler.scale(loss).backward()
            if step % args.grad_accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        predicted, expected = [], []
        with torch.inference_mode():
            for batch in eval_loader:
                expected.extend(batch["labels"].tolist())
                batch = {key: value.to(args.device) for key, value in batch.items() if key != "labels"}
                predicted.extend(model(**batch).logits.argmax(dim=-1).cpu().tolist())
        accuracy = sum(pred == actual for pred, actual in zip(predicted, expected, strict=True)) / len(expected)
        print(f"epoch={epoch} eval_accuracy={accuracy:.4f} eval_macro_f1={macro_f1(predicted, expected, len(catalog.labels)):.4f}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "intent-training-metadata.json").write_text(json.dumps({
        "catalog": {"name": catalog.name, "version": catalog.version, "labels": list(catalog.labels)},
        "base_model": args.model,
        "examples": len(examples),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    print(f"saved={output} train_examples={len(train_examples)} eval_examples={len(eval_examples)}")


if __name__ == "__main__":
    main()
