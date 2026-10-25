"""Measure a trained MiniLM dual encoder on labelled, unseen JSONL examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def load_examples(path: Path) -> list[dict]:
    examples: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"state", "instruction", "options", "expected"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{line_number} is missing {sorted(missing)}")
        if not isinstance(row["options"], list) or not row["options"]:
            raise ValueError(f"{path}:{line_number} needs a non-empty options list")
        if not isinstance(row["expected"], int) or not 0 <= row["expected"] < len(row["options"]):
            raise ValueError(f"{path}:{line_number} has an invalid expected option index")
        examples.append(row)
    if not examples:
        raise ValueError(f"{path} contains no examples")
    return examples


def max_length_from_metadata(model_path: Path) -> int:
    metadata_path = model_path / "training_metadata.json"
    if metadata_path.exists():
        value = json.loads(metadata_path.read_text(encoding="utf-8")).get("max_length")
        if isinstance(value, int) and value > 0:
            return value
    return 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path,
                        help="Unseen JSONL data with state, instruction, options, and expected fields.")
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    examples = load_examples(args.data)
    max_length = args.max_length or max_length_from_metadata(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    use_amp = device.type == "cuda"

    correct = 0
    total_loss = 0.0
    predictions: list[dict] = []
    with torch.inference_mode():
        for row in tqdm(examples, desc="Evaluating"):
            state_inputs = tokenizer(row["state"], return_tensors="pt", truncation=True,
                                     max_length=max_length).to(device)
            option_inputs = tokenizer(
                [f"{row['instruction']}: {option}" for option in row["options"]],
                return_tensors="pt", padding=True, truncation=True, max_length=max_length,
            ).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                state_embedding = F.normalize(model(**state_inputs).last_hidden_state[:, 0], dim=-1)
                option_embeddings = F.normalize(model(**option_inputs).last_hidden_state[:, 0], dim=-1)
                scores = (state_embedding @ option_embeddings.T).squeeze(0) * 20
                probabilities = scores.softmax(dim=0)
                loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([row["expected"]], device=device))
            predicted = int(scores.argmax().item())
            correct += predicted == row["expected"]
            total_loss += loss.item()
            predictions.append({
                "id": row.get("id"),
                "expected_index": row["expected"],
                "expected_option": row["options"][row["expected"]],
                "predicted_index": predicted,
                "predicted_option": row["options"][predicted],
                "confidence": round(float(probabilities[predicted].item()), 6),
                "correct": predicted == row["expected"],
            })

    report = {
        "model": str(args.model),
        "data": str(args.data),
        "examples": len(examples),
        "accuracy": correct / len(examples),
        "mean_cross_entropy_loss": total_loss / len(examples),
        "predictions": predictions,
    }
    by_source: dict[str, dict[str, float | int]] = {}
    for row, prediction in zip(examples, predictions):
        source = str(row.get("source", "unspecified"))
        summary = by_source.setdefault(source, {"examples": 0, "correct": 0})
        summary["examples"] += 1
        summary["correct"] += int(prediction["correct"])
    for summary in by_source.values():
        summary["accuracy"] = summary["correct"] / summary["examples"]
    report["by_source"] = by_source
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Accuracy: {report['accuracy']:.2%} ({correct}/{len(examples)})")
    print(f"Mean cross-entropy loss: {report['mean_cross_entropy_loss']:.4f}")
    for source, summary in by_source.items():
        print(f"{source}: {summary['accuracy']:.2%} ({summary['correct']}/{summary['examples']})")
    print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()
