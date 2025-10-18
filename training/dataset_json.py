"""Dataset for JSON generation training."""

import json
import torch
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from dataclasses import dataclass


@dataclass
class JSONExample:
    """Single JSON training example."""

    id: str
    input_text: str
    target_json: str
    metadata: Dict = None


class InvoiceJSONDataset(Dataset):
    """PyTorch dataset for JSON generation."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: T5Tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 512,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON file with examples
            tokenizer: T5 tokenizer
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # Load examples
        self.examples = self._load_examples()

    def _load_examples(self) -> List[JSONExample]:
        """Load examples from JSON file."""
        examples = []

        with open(self.data_path, "r") as f:
            data = json.load(f)

        for item in data:
            examples.append(
                JSONExample(
                    id=item["id"],
                    input_text=item["input"],
                    target_json=item["target"],
                    metadata=item.get("metadata", {}),
                )
            )

        return examples

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.

        Args:
            idx: Example index

        Returns:
            Dictionary with tokenized inputs and labels
        """
        example = self.examples[idx]

        # Tokenize input
        input_encoding = self.tokenizer(
            example.input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target (JSON output)
        target_encoding = self.tokenizer(
            example.target_json,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # T5 uses -100 for padding in labels (ignored in loss)
        labels = target_encoding["input_ids"].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


class InvoiceJSONDatasetBuilder:
    """Build JSON dataset from various sources."""

    @staticmethod
    def from_bio_dataset(
        bio_data_path: Path,
        output_path: Path,
        tokenizer: T5Tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 512,
    ) -> "InvoiceJSONDataset":
        """
        Build JSON dataset from BIO-labeled data.

        Args:
            bio_data_path: Path to BIO-labeled JSON file
            output_path: Path to save converted JSON data
            tokenizer: T5 tokenizer
            max_input_length: Maximum input length
            max_target_length: Maximum target length

        Returns:
            InvoiceJSONDataset instance
        """
        from training.bio_to_json_converter import BIOToJSONConverter

        # Convert BIO to JSON format
        converter = BIOToJSONConverter()
        converter.convert_dataset(bio_data_path, output_path)

        # Load and return dataset
        return InvoiceJSONDataset(
            output_path,
            tokenizer,
            max_input_length,
            max_target_length,
        )

    @staticmethod
    def create_synthetic_example(
        invoice_text: str,
        invoice_data: Dict,
        tokenizer: T5Tokenizer,
    ) -> Dict[str, torch.Tensor]:
        """
        Create a single synthetic training example.

        Args:
            invoice_text: Raw invoice text
            invoice_data: Structured invoice data (dict matching schema)
            tokenizer: T5 tokenizer

        Returns:
            Tokenized example dictionary
        """
        # Format input
        input_text = f"extract invoice fields: {invoice_text}"

        # Format target
        target_json = json.dumps(invoice_data, ensure_ascii=False)

        # Tokenize
        input_encoding = tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = tokenizer(
            target_json,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze().clone()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Args:
        batch: List of examples

    Returns:
        Batched dictionary
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }
