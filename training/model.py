"""DistilBERT model for invoice field extraction."""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
)
from pathlib import Path
from typing import Dict, Optional


class InvoiceExtractionModel(nn.Module):
    """DistilBERT-based model for invoice field extraction."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 15,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize model.

        Args:
            model_name: Pretrained model name
            num_labels: Number of entity labels
            label2id: Label to id mapping
            id2label: Id to label mapping
        """
        super().__init__()

        # Load pretrained config
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id or {},
            id2label=id2label or {},
        )

        # Load pretrained model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, config=config
        )

        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for each token [batch_size, seq_len]

        Returns:
            Dictionary with loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }

    def save(self, save_path: Path) -> None:
        """
        Save model to disk.

        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model and config
        self.model.save_pretrained(save_path)

        print(f"Model saved to {save_path}")

    @classmethod
    def load(cls, load_path: Path) -> "InvoiceExtractionModel":
        """
        Load model from disk.

        Args:
            load_path: Directory containing saved model

        Returns:
            Loaded model
        """
        load_path = Path(load_path)

        # Load config to get num_labels and mappings
        config = AutoConfig.from_pretrained(load_path)

        # Create model instance
        model = cls(
            model_name=str(load_path),
            num_labels=config.num_labels,
            label2id=config.label2id,
            id2label=config.id2label,
        )

        print(f"Model loaded from {load_path}")

        return model


class ModelTrainer:
    """Trainer for invoice extraction model."""

    def __init__(
        self,
        model: InvoiceExtractionModel,
        optimizer: torch.optim.Optimizer,
        device: str = "auto",
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            device: Device to train on ("auto", "cuda", "mps", "cpu")
        """
        self.model = model
        self.optimizer = optimizer
        self.device = self._get_device(device)

        self.model.to(self.device)

        print(f"Training on device: {self.device}")

    @staticmethod
    def _get_device(device: str) -> str:
        """
        Get the best available device.

        Args:
            device: Requested device ("auto", "cuda", "mps", "cpu")

        Returns:
            Device string
        """
        if device == "auto":
            # Try devices in order: CUDA > MPS > CPU
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        elif device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            return "cpu"

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.

        Args:
            batch: Batch of data

        Returns:
            Loss value
        """
        self.model.train()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """
        Single evaluation step.

        Args:
            batch: Batch of data

        Returns:
            Tuple of (loss, predictions, labels)
        """
        self.model.eval()

        with torch.no_grad():
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"]
            logits = outputs["logits"]

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

        return loss.item(), predictions.cpu(), labels.cpu()

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        step: int,
        best_metric: float,
    ) -> None:
        """
        Save training checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch
            step: Current step
            best_metric: Best metric value so far
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": best_metric,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")

        return checkpoint
