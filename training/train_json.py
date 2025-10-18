"""Training script for JSON generation model."""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, get_linear_schedule_with_warmup

from training.model_json import InvoiceJSONExtractionModel, JSONModelTrainer
from training.dataset_json import InvoiceJSONDataset, collate_fn
from training.metrics_json import JSONGenerationMetrics
from training.bio_to_json_converter import convert_all_splits


def train_epoch(
    trainer: JSONModelTrainer,
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
    trainer: JSONModelTrainer,
    eval_loader: DataLoader,
    metrics_calculator: JSONGenerationMetrics,
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
    all_references = []

    progress_bar = tqdm(eval_loader, desc="Evaluating")

    for batch in progress_bar:
        loss, generated_ids, label_ids = trainer.eval_step(batch)
        total_loss += loss

        # Decode predictions and references
        predictions = trainer.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        # Replace -100 with pad_token_id for decoding
        label_ids[label_ids == -100] = trainer.tokenizer.pad_token_id
        references = trainer.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        all_predictions.extend(predictions)
        all_references.extend(references)

    # Calculate metrics
    metrics = metrics_calculator.compute_metrics(all_predictions, all_references)
    metrics["loss"] = total_loss / len(eval_loader)

    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train JSON generation model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing BIO-labeled data",
    )
    parser.add_argument(
        "--json-data-dir",
        type=str,
        default="data/json",
        help="Directory for converted JSON data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/json_model",
        help="Directory to save model",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/json_model",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/flan-t5-base",
        help="Pretrained model name (t5-small, t5-base, flan-t5-base, etc.)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
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
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
        help="Maximum input sequence length",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=512,
        help="Maximum target sequence length",
    )
    parser.add_argument(
        "--convert-data",
        action="store_true",
        help="Convert BIO data to JSON format before training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps for scheduler",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    json_data_dir = Path(args.json_data_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    json_data_dir.mkdir(parents=True, exist_ok=True)

    # Convert data if requested
    if args.convert_data:
        print("Converting BIO-labeled data to JSON format...")
        convert_all_splits(
            Path(args.data_dir),
            json_data_dir,
            splits=["train", "val", "test"],
        )

    # Set device
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
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    # Load datasets
    print("Loading datasets...")
    train_path = json_data_dir / "train_json.json"
    val_path = json_data_dir / "val_json.json"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            f"Please run with --convert-data flag or prepare your data first."
        )

    train_dataset = InvoiceJSONDataset(
        train_path,
        tokenizer,
        args.max_input_length,
        args.max_target_length,
    )

    val_dataset = None
    if val_path.exists():
        val_dataset = InvoiceJSONDataset(
            val_path,
            tokenizer,
            args.max_input_length,
            args.max_target_length,
        )

    print(f"Train examples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation examples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # Initialize model
    print("Initializing model...")
    model = InvoiceJSONExtractionModel(
        model_name=args.model_name,
        from_pretrained=True,
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Initialize trainer
    trainer = JSONModelTrainer(model, optimizer, device)

    # Initialize metrics
    metrics_calculator = JSONGenerationMetrics()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = trainer.load_checkpoint(Path(args.resume))
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]

    # Training loop
    print("\nStarting training...")
    training_history = []

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'=' * 50}")

        # Train
        train_loss = train_epoch(trainer, train_loader, epoch + 1, scheduler)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        if val_loader:
            print("\nEvaluating on validation set...")
            metrics = evaluate(trainer, val_loader, metrics_calculator)

            print(f"Validation loss:     {metrics['loss']:.4f}")
            print(f"Valid JSON rate:     {metrics['valid_json_rate']:.2%}")
            print(f"Exact match:         {metrics['exact_match']:.2%}")
            print(f"Field accuracy:      {metrics['field_accuracy']:.2%}")
            print(f"Field F1:            {metrics['field_f1']:.2%}")

            # Save checkpoint if best model (using field_f1 as main metric)
            current_metric = metrics["field_f1"]
            if current_metric > best_metric:
                best_metric = current_metric
                print(f"\nNew best Field F1: {best_metric:.4f}")

                # Save best model
                best_model_path = output_dir / "best_model"
                model.save(best_model_path)
                print(f"Best model saved to {best_model_path}")

            # Save training history
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_metrics": metrics,
                }
            )

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        trainer.save_checkpoint(checkpoint_path, epoch, 0, best_metric)

    # Save final model
    final_model_path = output_dir / "final_model"
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best Field F1 score: {best_metric:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
