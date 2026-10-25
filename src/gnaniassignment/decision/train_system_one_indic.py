"""Train the SystemOne-Indic dual-encoder from the full IndicJevBench dataset.

The default configuration is deliberately sized for consumer GPUs (including 4 GB
cards): it tunes the final two encoder blocks and accumulates gradients one
example at a time. ``--trainable-layers 0`` requests full fine-tuning and needs
substantially more VRAM.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

MODEL = "microsoft/Multilingual-MiniLM-L12-H384"


def rows():
    data = load_dataset("cmul8-hf/IndicJevBench", "default")
    for split in data.values():
        for row in split:
            question = row["question"]
            options = question.get("options") or (["no", "yes"] if question["type"] == "noul" else [])
            if options and 0 <= row["expected"] < len(options):
                yield row["state"], question["instructions"], options, row["expected"]


def freeze_encoder_layers(model: torch.nn.Module, trainable_layers: int) -> None:
    """Freeze all but the last N transformer blocks (0 means train all)."""
    if trainable_layers == 0:
        return
    layers = getattr(getattr(model, "encoder", None), "layer", None)
    if layers is None:
        raise ValueError("Could not find encoder layers; use --trainable-layers 0 for this model.")
    if not 1 <= trainable_layers <= len(layers):
        raise ValueError(f"--trainable-layers must be between 0 and {len(layers)}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in layers[-trainable_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8, help="Gradient-accumulation batch size.")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--trainable-layers", type=int, default=2, help="Final blocks to tune; 0 tunes all 12.")
    parser.add_argument("--output", default="artifacts/models/systemone-indic-v1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    encoder = AutoModel.from_pretrained(MODEL).to(device)
    freeze_encoder_layers(encoder, args.trainable_layers)
    encoder.train()

    trainable = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = DataLoader(list(rows()), batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: batch)

    for epoch in range(args.epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            # Do not retain graphs for every item in the batch: that was the OOM source.
            for state, instruction, options, target in batch:
                state_inputs = tokenizer(state, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
                choice_inputs = tokenizer(
                    [f"{instruction}: {option}" for option in options],
                    return_tensors="pt", padding=True, truncation=True, max_length=args.max_length,
                ).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    state_embedding = F.normalize(encoder(**state_inputs).last_hidden_state[:, 0], dim=-1)
                    choice_embeddings = F.normalize(encoder(**choice_inputs).last_hidden_state[:, 0], dim=-1)
                    item_loss = F.cross_entropy((state_embedding @ choice_embeddings.T) * 20, torch.tensor([target], device=device))
                    item_loss = item_loss / len(batch)
                scaler.scale(item_loss).backward()
                batch_loss += item_loss.detach().item()
            scaler.step(optimizer)
            scaler.update()
        print(f"epoch={epoch + 1} loss={batch_loss:.4f}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(output)
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    main()
