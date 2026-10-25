"""Fine-tune Multilingual MiniLM on IndicJevBench in Google Colab.

The saved output is a standard Hugging Face AutoModel directory, compatible
with ``AutoModel.from_pretrained(<output-directory>)``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "microsoft/Multilingual-MiniLM-L12-H384"
DATASET_ID = "cmul8-hf/IndicJevBench"


def usable_rows(dataset):
    """Yield examples that have a valid expected answer and at least one choice."""
    for row in dataset:
        question = row["question"]
        options = question.get("options") or (
            ["no", "yes"] if question.get("type") == "noul" else []
        )
        expected = row["expected"]
        if options and 0 <= expected < len(options):
            yield row["state"], question["instructions"], options, expected


def freeze_encoder_layers(model: torch.nn.Module, trainable_layers: int) -> None:
    """Freeze all but MiniLM's last N transformer blocks; zero means all blocks."""
    if trainable_layers == 0:
        return
    layers = getattr(getattr(model, "encoder", None), "layer", None)
    if layers is None:
        raise ValueError("Expected a BERT-style encoder with encoder.layer.")
    if not 1 <= trainable_layers <= len(layers):
        raise ValueError(f"--trainable-layers must be between 0 and {len(layers)}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in layers[-trainable_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)


def save_model(model, tokenizer, destination: Path, metadata: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    (destination / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def is_validation_group(state: str, fraction: float) -> bool:
    """Assign every question sharing a state to the same deterministic split."""
    digest = hashlib.sha256(state.strip().casefold().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def evaluate(model, tokenizer, examples, device: torch.device, max_length: int) -> tuple[float, float]:
    """Return exact-choice accuracy and mean cross-entropy without updating weights."""
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.inference_mode():
        for state, instruction, options, target in tqdm(examples, desc="Validating", leave=False):
            state_inputs = tokenizer(state, return_tensors="pt", truncation=True,
                                     max_length=max_length).to(device)
            choice_inputs = tokenizer([f"{instruction}: {option}" for option in options],
                                      return_tensors="pt", padding=True, truncation=True,
                                      max_length=max_length).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                state_embedding = F.normalize(model(**state_inputs).last_hidden_state[:, 0], dim=-1)
                choice_embeddings = F.normalize(model(**choice_inputs).last_hidden_state[:, 0], dim=-1)
                scores = (state_embedding @ choice_embeddings.T) * 20
                total_loss += F.cross_entropy(scores, torch.tensor([target], device=device)).item()
            correct += int(scores.argmax(dim=1).item() == target)
    model.train()
    return correct / len(examples), total_loss / len(examples)


def save_validation_split(examples, path: Path) -> None:
    """Persist the exact held-out examples used for validation and later audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for state, instruction, options, expected in examples:
            file.write(json.dumps({
                "state": state,
                "instruction": instruction,
                "options": options,
                "expected": expected,
            }, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/content/drive/MyDrive/systemone-indic-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Examples per optimizer step; examples are processed one at a time to limit VRAM.")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--trainable-layers", type=int, default=0,
                        help="0 fine-tunes all 12 layers; 2 tunes only the final two.")
    parser.add_argument("--validation-fraction", type=float, default=0.10,
                        help="State-group-disjoint IndicJevBench fraction reserved for validation.")
    parser.add_argument(
        "--splits",
        default="all",
        help="Comma-separated IndicJevBench dataset splits, or 'all' (the default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.validation_fraction < 1:
        raise ValueError("--epochs and --batch-size must be positive; --validation-fraction must be between 0 and 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. In Colab select Runtime > Change runtime type > T4 GPU.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset_dict = load_dataset(DATASET_ID, "default")
    available_splits = list(dataset_dict.keys())
    selected_splits = available_splits if args.splits == "all" else [
        split.strip() for split in args.splits.split(",") if split.strip()
    ]
    missing_splits = sorted(set(selected_splits) - set(available_splits))
    if missing_splits:
        raise ValueError(f"Unknown split(s): {missing_splits}. Available: {available_splits}")
    examples = [
        example
        for split in selected_splits
        for example in usable_rows(dataset_dict[split])
    ]
    if not examples:
        raise ValueError("No usable training examples were found.")
    validation_examples = [row for row in examples if is_validation_group(row[0], args.validation_fraction)]
    train_examples = [row for row in examples if not is_validation_group(row[0], args.validation_fraction)]
    if not train_examples or not validation_examples:
        raise ValueError("The deterministic split produced an empty training or validation set.")
    print(f"Loaded {len(examples):,} examples from {DATASET_ID}: {', '.join(selected_splits)}")
    print(f"Training: {len(train_examples):,}; validation: {len(validation_examples):,} (state-group-disjoint)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    freeze_encoder_layers(model, args.trainable_layers)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=list)
    output = Path(args.output)
    save_validation_split(validation_examples, output / "validation-split.jsonl")
    best_validation_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            for state, instruction, options, target in batch:
                state_inputs = tokenizer(state, return_tensors="pt", truncation=True,
                                         max_length=args.max_length).to(device)
                choice_inputs = tokenizer([f"{instruction}: {option}" for option in options],
                                          return_tensors="pt", padding=True, truncation=True,
                                          max_length=args.max_length).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    state_embedding = F.normalize(model(**state_inputs).last_hidden_state[:, 0], dim=-1)
                    choice_embeddings = F.normalize(model(**choice_inputs).last_hidden_state[:, 0], dim=-1)
                    loss = F.cross_entropy(
                        (state_embedding @ choice_embeddings.T) * 20,
                        torch.tensor([target], device=device),
                    ) / len(batch)
                scaler.scale(loss).backward()
                total_loss += loss.detach().item()
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(loss=f"{total_loss / (progress.n + 1):.4f}")

        metadata = {
            "base_model": MODEL_ID,
            "dataset": DATASET_ID,
            "splits": selected_splits,
            "epoch": epoch,
            "training_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "validation_fraction": args.validation_fraction,
            "max_length": args.max_length,
            "trainable_layers": args.trainable_layers,
            "mean_training_loss": total_loss / len(loader),
        }
        validation_accuracy, validation_loss = evaluate(
            model, tokenizer, validation_examples, device, args.max_length
        )
        metadata.update({
            "validation_accuracy": validation_accuracy,
            "validation_cross_entropy_loss": validation_loss,
        })
        save_model(model, tokenizer, output / f"checkpoint-epoch-{epoch}", metadata)
        print(f"epoch={epoch} validation_accuracy={validation_accuracy:.2%} validation_loss={validation_loss:.4f}")
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            save_model(model, tokenizer, output, metadata)
            print(f"New best model saved to {output}")
        print(f"Saved checkpoint to {output / f'checkpoint-epoch-{epoch}'}")

    print(f"Best validation accuracy: {best_validation_accuracy:.2%}")


if __name__ == "__main__":
    main()
