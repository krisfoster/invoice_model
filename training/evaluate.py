"""Evaluation script for invoice extraction model."""

import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from training.config import ModelConfig, DataConfig
from training.dataset import InvoiceDataset
from training.model import InvoiceExtractionModel, ModelTrainer
from training.metrics import InvoiceMetrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate invoice extraction model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data (JSON file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    # Set device (auto-detect best available: CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device: {device} ({device_name})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"Using device: {device} (Apple Metal Performance Shaders)")
    else:
        device = "cpu"
        print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = InvoiceExtractionModel.load(Path(args.model_path))
    model.to(device)
    model.eval()

    # Get label mappings
    config = model.model.config
    label2id = config.label2id
    id2label = config.id2label

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load test dataset
    print(f"Loading test data from {args.test_data}")
    test_dataset = InvoiceDataset(
        Path(args.test_data),
        tokenizer,
        label2id,
        args.max_length,
    )

    print(f"Test examples: {len(test_dataset)}")

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Initialize metrics calculator
    metrics_calculator = InvoiceMetrics(id2label)

    # Run evaluation
    print("\nEvaluating model...")

    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"]
            logits = outputs["logits"]

            predictions = torch.argmax(logits, dim=-1)

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()

    # Concatenate all predictions and labels
    import numpy as np

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    metrics = metrics_calculator.compute_metrics(all_predictions, all_labels)
    metrics["loss"] = total_loss / len(test_loader)

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Exact Match: {metrics['exact_match']:.4f}")

    # Print per-entity metrics
    print("\nPer-entity metrics:")
    entity_types = set()
    for key in metrics.keys():
        if key.endswith("_f1"):
            entity_type = key.replace("_f1", "")
            entity_types.add(entity_type)

    for entity_type in sorted(entity_types):
        if entity_type in ["loss", "precision", "recall", "f1", "exact", "match"]:
            continue

        precision = metrics.get(f"{entity_type}_precision", 0)
        recall = metrics.get(f"{entity_type}_recall", 0)
        f1 = metrics.get(f"{entity_type}_f1", 0)

        print(f"  {entity_type:20s} - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

    # Get detailed classification report
    print("\nDetailed classification report:")
    report = metrics_calculator.get_classification_report(all_predictions, all_labels)
    print(report)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "model_path": str(args.model_path),
            "test_data": str(args.test_data),
            "num_examples": len(test_dataset),
            "metrics": metrics,
            "classification_report": report,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
