"""Seq2Seq model for direct JSON generation with constrained decoding."""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoConfig,
)
from pathlib import Path
from typing import Dict, Optional


class InvoiceJSONExtractionModel(nn.Module):
    """T5-based model for direct JSON generation."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        from_pretrained: bool = True,
    ):
        """
        Initialize model.

        Args:
            model_name: Pretrained model name or path
            from_pretrained: Whether to load pretrained weights
        """
        super().__init__()

        self.model_name = model_name

        if from_pretrained:
            # Load pretrained T5 model
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        else:
            # Load from local checkpoint
            config = AutoConfig.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration(config)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)

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
            labels: Target token ids [batch_size, target_seq_len]

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

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 512,
        num_beams: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate output sequence (unconstrained).

        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments

        Returns:
            Generated token ids
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )

    def save(self, save_path: Path) -> None:
        """
        Save model to disk.

        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        print(f"Model and tokenizer saved to {save_path}")

    @classmethod
    def load(cls, load_path: Path) -> "InvoiceJSONExtractionModel":
        """
        Load model from disk.

        Args:
            load_path: Directory containing saved model

        Returns:
            Loaded model
        """
        load_path = Path(load_path)

        # Create model instance with pretrained=False to avoid re-downloading
        model = cls(model_name=str(load_path), from_pretrained=True)

        print(f"Model loaded from {load_path}")

        return model


class JSONModelTrainer:
    """Trainer for JSON generation model."""

    def __init__(
        self,
        model: InvoiceJSONExtractionModel,
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
        self.tokenizer = model.tokenizer

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
        Single evaluation step with generation.

        Args:
            batch: Batch of data

        Returns:
            Tuple of (loss, generated_ids, label_ids)
        """
        self.model.eval()

        with torch.no_grad():
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Compute loss
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"]

            # Generate outputs for evaluation
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
            )

        return loss.item(), generated_ids.cpu(), labels.cpu()

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
