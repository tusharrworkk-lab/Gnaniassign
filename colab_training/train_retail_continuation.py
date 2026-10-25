"""Continue fine-tuning a MiniLM dual encoder on retail JSONL data in Colab."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def load_splits(path: Path) -> dict[str, list[tuple[str, str, list[str], int]]]:
    splits = {"train": [], "validation": [], "test": []}
    group_splits: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("split")
        question = row.get("question", {})
        options, expected = question.get("options"), row.get("expected")
        if split not in splits or not isinstance(options, list) or not isinstance(expected, int):
            raise ValueError(f"Invalid row at {path}:{line_number}")
        if not 0 <= expected < len(options):
            raise ValueError(f"Invalid expected index at {path}:{line_number}")
        group_id = row.get("group_id")
        if group_id:
            prior = group_splits.setdefault(group_id, split)
            if prior != split:
                raise ValueError(f"Group {group_id!r} appears in both {prior} and {split}")
        splits[split].append((row["state"], question["instructions"], options, expected))
    if not all(splits.values()):
        raise ValueError("The data file must contain non-empty train, validation, and test splits.")
    return splits


def score(model, tokenizer, examples, device: torch.device, max_length: int) -> tuple[float, float]:
    model.eval()
    correct = 0
    loss_total = 0.0
    with torch.inference_mode():
        for state, instruction, options, target in tqdm(examples, desc="Evaluating", leave=False):
            state_inputs = tokenizer(state, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            choice_inputs = tokenizer([f"{instruction}: {option}" for option in options], return_tensors="pt",
                                      padding=True, truncation=True, max_length=max_length).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                state_embedding = F.normalize(model(**state_inputs).last_hidden_state[:, 0], dim=-1)
                choice_embeddings = F.normalize(model(**choice_inputs).last_hidden_state[:, 0], dim=-1)
                logits = (state_embedding @ choice_embeddings.T) * 20
                loss_total += F.cross_entropy(logits, torch.tensor([target], device=device)).item()
            correct += int(logits.argmax(dim=1).item() == target)
    model.train()
    return correct / len(examples), loss_total / len(examples)


def save(model, tokenizer, output: Path, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "retail-training-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="Existing systemone-indic-v1 model directory.")
    parser.add_argument("--data", required=True, type=Path, help="retail-intent-synthetic-v1.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8, help="Effective batch size via per-item gradient accumulation.")
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("--epochs, --batch-size, and --learning-rate must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("A Colab GPU is required. Select Runtime > Change runtime type > T4 GPU.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    splits = load_splits(args.data)
    print("Dataset:", {name: len(rows) for name, rows in splits.items()})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True, collate_fn=list)
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            for state, instruction, options, target in batch:
                state_inputs = tokenizer(state, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
                choice_inputs = tokenizer([f"{instruction}: {option}" for option in options], return_tensors="pt",
                                          padding=True, truncation=True, max_length=args.max_length).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    state_embedding = F.normalize(model(**state_inputs).last_hidden_state[:, 0], dim=-1)
                    choice_embeddings = F.normalize(model(**choice_inputs).last_hidden_state[:, 0], dim=-1)
                    loss = F.cross_entropy((state_embedding @ choice_embeddings.T) * 20,
                                           torch.tensor([target], device=device)) / len(batch)
                scaler.scale(loss).backward()
                train_loss += loss.detach().item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(loss=f"{train_loss / (progress.n + 1):.4f}")

        validation_accuracy, validation_loss = score(model, tokenizer, splits["validation"], device, args.max_length)
        metadata = {
            "base_model": str(args.model), "data": str(args.data), "epoch": epoch,
            "training_examples": len(splits["train"]), "validation_examples": len(splits["validation"]),
            "test_examples": len(splits["test"]), "learning_rate": args.learning_rate,
            "mean_training_loss": train_loss / len(loader), "validation_accuracy": validation_accuracy,
            "validation_cross_entropy_loss": validation_loss,
        }
        save(model, tokenizer, args.output / f"checkpoint-epoch-{epoch}", metadata)
        print(f"epoch={epoch} validation_accuracy={validation_accuracy:.2%} validation_loss={validation_loss:.4f}")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save(model, tokenizer, args.output, metadata)
            print(f"New best model saved to {args.output}")

    best_model = AutoModel.from_pretrained(args.output).to(device).train()
    test_accuracy, test_loss = score(best_model, tokenizer, splits["test"], device, args.max_length)
    report = {"best_validation_accuracy": best_accuracy, "test_accuracy": test_accuracy, "test_cross_entropy_loss": test_loss}
    (args.output / "retail-test-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"FINAL test_accuracy={test_accuracy:.2%} test_loss={test_loss:.4f}")


if __name__ == "__main__":
    main()
