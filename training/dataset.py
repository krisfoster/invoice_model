"""Dataset preparation and loading for invoice extraction."""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass


@dataclass
class InvoiceExample:
    """Single invoice training example."""

    id: str
    text: str
    tokens: List[str]
    labels: List[str]
    metadata: Optional[Dict] = None


class InvoiceDataset(Dataset):
    """PyTorch dataset for invoice extraction."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int = 512,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON file with examples
            tokenizer: Hugging Face tokenizer
            label2id: Mapping from label names to ids
            max_length: Maximum sequence length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

        # Load examples
        self.examples = self._load_examples()

    def _load_examples(self) -> List[InvoiceExample]:
        """Load examples from JSON file."""
        examples = []

        with open(self.data_path, "r") as f:
            data = json.load(f)

        for item in data:
            examples.append(
                InvoiceExample(
                    id=item["id"],
                    text=item["text"],
                    tokens=item["tokens"],
                    labels=item["labels"],
                    metadata=item.get("metadata", {}),
                )
            )

        return examples

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single example.

        Args:
            idx: Example index

        Returns:
            Dictionary with tokenized inputs and labels
        """
        example = self.examples[idx]

        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            example.tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Align labels with tokens
        labels = self._align_labels(
            example.labels, encoding.word_ids(batch_index=0)
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _align_labels(
        self, labels: List[str], word_ids: List[Optional[int]]
    ) -> List[int]:
        """
        Align labels with tokenized words.

        For subword tokens, we use the label of the first token and
        ignore the rest (-100 is the ignore index in PyTorch).

        Args:
            labels: Original word-level labels
            word_ids: Word ids from tokenizer

        Returns:
            Aligned label ids
        """
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special token (CLS, SEP, PAD)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word
                label = labels[word_idx] if word_idx < len(labels) else "O"
                aligned_labels.append(self.label2id.get(label, 0))
            else:
                # Subsequent token of a word (subword)
                # Use -100 to ignore in loss computation
                aligned_labels.append(-100)

            previous_word_idx = word_idx

        return aligned_labels


class DatasetPreprocessor:
    """Preprocess raw Kaggle invoice data into training format."""

    def __init__(self, label2id: Dict[str, int]):
        """
        Initialize preprocessor.

        Args:
            label2id: Mapping from label names to ids
        """
        self.label2id = label2id

    def prepare_from_kaggle(
        self,
        kaggle_data_path: Path,
        output_path: Path,
        split: str = "train",
    ) -> None:
        """
        Convert Kaggle invoice dataset to our format.

        Expected Kaggle format:
        - CSV or JSON with columns: text, annotations
        - annotations should be a dict with entity types and their positions

        Args:
            kaggle_data_path: Path to Kaggle dataset
            output_path: Path to save processed data
            split: Dataset split name
        """
        import pandas as pd

        # Load Kaggle data
        if kaggle_data_path.suffix == ".csv":
            df = pd.read_csv(kaggle_data_path)
        elif kaggle_data_path.suffix == ".json":
            df = pd.read_json(kaggle_data_path)
        else:
            raise ValueError(f"Unsupported file format: {kaggle_data_path.suffix}")

        examples = []

        for idx, row in df.iterrows():
            # Extract text
            text = row.get("text", "")

            # Parse annotations
            annotations = row.get("annotations", {})
            if isinstance(annotations, str):
                annotations = json.loads(annotations)

            # Convert to tokens and BIO labels
            tokens, labels = self._text_to_bio(text, annotations)

            examples.append(
                {
                    "id": f"{split}_{idx}",
                    "text": text,
                    "tokens": tokens,
                    "labels": labels,
                    "metadata": {
                        "split": split,
                        "source": "kaggle",
                    },
                }
            )

        # Save processed examples
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)

        print(f"Saved {len(examples)} examples to {output_path}")

    def _text_to_bio(
        self, text: str, annotations: Dict
    ) -> Tuple[List[str], List[str]]:
        """
        Convert text and annotations to BIO format.

        Args:
            text: Invoice text
            annotations: Dict mapping entity types to spans

        Returns:
            Tuple of (tokens, labels)
        """
        # Simple whitespace tokenization
        tokens = text.split()

        # Initialize all labels as "O" (outside)
        labels = ["O"] * len(tokens)

        # Map character positions to token indices
        token_positions = []
        current_pos = 0

        for token in tokens:
            start = text.find(token, current_pos)
            end = start + len(token)
            token_positions.append((start, end))
            current_pos = end

        # Apply annotations
        for entity_type, spans in annotations.items():
            if not isinstance(spans, list):
                spans = [spans]

            for span in spans:
                start_char = span.get("start", -1)
                end_char = span.get("end", -1)

                if start_char == -1 or end_char == -1:
                    continue

                # Find tokens that overlap with this span
                is_first = True
                for token_idx, (token_start, token_end) in enumerate(token_positions):
                    # Check if token overlaps with span
                    if token_start < end_char and token_end > start_char:
                        prefix = "B-" if is_first else "I-"
                        labels[token_idx] = f"{prefix}{entity_type.upper()}"
                        is_first = False

        return tokens, labels

    def create_synthetic_example(
        self, invoice_text: str, annotations: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Create a single synthetic training example.

        Useful for augmenting dataset or creating test examples.

        Args:
            invoice_text: Raw invoice text
            annotations: Dictionary mapping field types to spans
                Example: {
                    "total": [{"text": "1234.56", "start": 100, "end": 107}],
                    "date": [{"text": "2024-01-15", "start": 50, "end": 60}]
                }

        Returns:
            Processed example dictionary
        """
        tokens, labels = self._text_to_bio(invoice_text, annotations)

        return {
            "id": "synthetic",
            "text": invoice_text,
            "tokens": tokens,
            "labels": labels,
            "metadata": {"source": "synthetic"},
        }
