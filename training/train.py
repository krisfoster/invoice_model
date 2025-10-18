"""Training script for invoice extraction model."""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from training.config import ModelConfig, DataConfig, TrainingConfig
from training.dataset import InvoiceDataset
from training.model import InvoiceExtractionModel, ModelTrainer
from training.metrics import InvoiceMetrics


def train_epoch(
    trainer: ModelTrainer,
    train_loader: DataLoader,
    epoch: int,
    scheduler=None,
) -> float:
    """
    Train for one epoch.

    Args:
        trainer: Model trainer
        train_loader: Training data loader
        epoch: Current epoch number
        scheduler: Learning rate scheduler

    Returns:
        Average training loss
    """
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        loss = trainer.train_step(batch)
        total_loss += loss

        if scheduler:
            scheduler.step()

        progress_bar.set_postfix({"loss": f"{loss:.4f}"})

    return total_loss / len(train_loader)


def evaluate(
    trainer: ModelTrainer,
    eval_loader: DataLoader,
    metrics_calculator: InvoiceMetrics,
) -> dict:
    """
    Evaluate model.

    Args:
        trainer: Model trainer
        eval_loader: Evaluation data loader
        metrics_calculator: Metrics calculator

    Returns:
        Dictionary of metrics
    """
    total_loss = 0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(eval_loader, desc="Evaluating")

    for batch in progress_bar:
        loss, predictions, labels = trainer.eval_step(batch)
        total_loss += loss

        all_predictions.append(predictions.numpy())
        all_labels.append(labels.numpy())

    # Concatenate all predictions and labels
    import numpy as np

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    metrics = metrics_calculator.compute_metrics(all_predictions, all_labels)
    metrics["loss"] = total_loss / len(eval_loader)

    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train invoice extraction model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default optimized for M3 Max 64GB)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate (default optimized for larger batch size)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )

    args = parser.parse_args()

    # Initialize configs
    model_config = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

    data_config = DataConfig(
        processed_data_dir=Path(args.data_dir),
    )

    training_config = TrainingConfig(
        output_dir=Path(args.output_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
    )

    # Create output directories
    training_config.output_dir.mkdir(parents=True, exist_ok=True)
    training_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        print("Warning: No GPU acceleration available. Training will be slower.")

    # Load tokenizer
    print(f"Loading tokenizer: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    # Load datasets
    print("Loading datasets...")
    train_path = data_config.processed_data_dir / data_config.train_file
    val_path = data_config.processed_data_dir / data_config.val_file

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            f"Please prepare your data first."
        )

    train_dataset = InvoiceDataset(
        train_path,
        tokenizer,
        model_config.label2id,
        model_config.max_length,
    )

    val_dataset = None
    if val_path.exists():
        val_dataset = InvoiceDataset(
            val_path,
            tokenizer,
            model_config.label2id,
            model_config.max_length,
        )

    print(f"Train examples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation examples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_config.batch_size,
            shuffle=False,
        )

    # Initialize model
    print("Initializing model...")
    model = InvoiceExtractionModel(
        model_name=model_config.model_name,
        num_labels=model_config.num_labels,
        label2id=model_config.label2id,
        id2label=model_config.id2label,
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )

    total_steps = len(train_loader) * model_config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=model_config.warmup_steps,
        num_training_steps=total_steps,
    )

    # Initialize trainer
    trainer = ModelTrainer(model, optimizer, device)

    # Initialize metrics
    metrics_calculator = InvoiceMetrics(model_config.id2label)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_f1 = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = trainer.load_checkpoint(Path(args.resume))
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_metric"]

    # Training loop
    print("\nStarting training...")
    training_history = []

    for epoch in range(start_epoch, model_config.num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{model_config.num_epochs}")
        print(f"{'=' * 50}")

        # Train
        train_loss = train_epoch(trainer, train_loader, epoch + 1, scheduler)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        if val_loader:
            print("\nEvaluating on validation set...")
            metrics = evaluate(trainer, val_loader, metrics_calculator)

            print(f"Validation loss: {metrics['loss']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1: {metrics['f1']:.4f}")
            print(f"Exact match: {metrics['exact_match']:.4f}")

            # Save checkpoint if best model
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                print(f"\nNew best F1: {best_f1:.4f}")

                # Save best model
                best_model_path = training_config.output_dir / "best_model"
                model.save(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                print(f"Best model and tokenizer saved to {best_model_path}")

            # Save training history
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_metrics": metrics,
                }
            )

        # Save checkpoint
        checkpoint_path = (
            training_config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        )
        trainer.save_checkpoint(checkpoint_path, epoch, 0, best_f1)

    # Save final model
    final_model_path = training_config.output_dir / "final_model"
    model.save(final_model_path)

    # Save tokenizer
    tokenizer.save_pretrained(final_model_path)
    print(f"Tokenizer saved to {final_model_path}")

    # Save training history
    history_path = training_config.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Models saved to: {training_config.output_dir}")


if __name__ == "__main__":
    main()
