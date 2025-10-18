"""Data preprocessing script for Company Documents invoice dataset."""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from training.config import ModelConfig
from training.pdf_processor import PDFProcessor


class InvoiceAnnotator:
    """Semi-automated invoice annotation using pattern matching."""

    def __init__(self):
        """Initialize annotator with common patterns."""
        # Regex patterns for common invoice fields
        self.patterns = {
            "invoice_number": [
                r"(?:order|ord)\s*id[\s:]*([A-Z0-9-]+)",
            ],
            "customer_name": [
                r"(?:customer|cust)\s*id[\s:]*([A-Z0-9-]+)",
            ],
            "date": [
                # Full 4-digit year formats (check first for priority)
                r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
                # Written dates like "July 4, 2016" or "July 4 2016"
                r"(?:date|dated)[\s:]*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                # 2-digit year formats (only if 4-digit not matched)
                r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            ],
            "total": [
                # Match currency amounts with various formats
                # Examples: 440.0, $440.00, 1,234.56, 1234.5
                r"(?:total|amount due|balance)[\s:$]*(\$?\d{1,}(?:[,.]\d+)*(?:\.\d+)?)",
                r"(?:^|\s)(?:total|sum)[\s:]*(?:\$|USD|EUR)?\s*(\d{1,}(?:[,.]\d+)*(?:\.\d+)?)",
                r"(?:^|\s)(?:totalprice)[\s:]*(?:\$|USD|EUR)?\s*(\d{1,}(?:[,.]\d+)*(?:\.\d+)?)",
            ],
        }

    def find_matches(self, text: str, field_type: str) -> List[Dict]:
        """
        Find all matches for a field type in text.

        Args:
            text: Text to search
            field_type: Field type (invoice_number, date, total, etc.)

        Returns:
            List of match dictionaries with text, start, end
        """
        matches = []

        if field_type not in self.patterns:
            return matches

        for pattern in self.patterns[field_type]:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                # Get the matched group (usually group 1)
                matched_text = match.group(1) if match.groups() else match.group(0)
                start = match.start(1) if match.groups() else match.start(0)
                end = match.end(1) if match.groups() else match.end(0)

                matches.append({
                    "text": matched_text.strip(),
                    "start": start,
                    "end": end,
                })

        return matches

    def auto_annotate(self, text: str) -> Dict[str, List[Dict]]:
        """
        Automatically annotate invoice text.

        Args:
            text: Invoice text

        Returns:
            Dictionary of annotations by field type
        """
        annotations = {}

        for field_type in self.patterns.keys():
            matches = self.find_matches(text, field_type)
            if matches:
                # Take first match for single-value fields
                if field_type in ["invoice_number", "date", "total"]:
                    annotations[field_type] = [matches[0]]
                else:
                    annotations[field_type] = matches

        return annotations


def process_invoice_pdfs(
    pdf_dir: Path,
    output_dir: Path,
    train_split: float = 0.8,
    val_split: float = 0.1,
    max_files: int = None,
) -> Tuple[int, int, int]:
    """
    Process invoice PDFs and create training data.

    Args:
        pdf_dir: Directory containing invoice PDFs
        output_dir: Directory to save processed data
        train_split: Training split ratio
        val_split: Validation split ratio
        max_files: Maximum number of files to process (None for all)

    Returns:
        Tuple of (train_count, val_count, test_count)
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all PDF files
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return 0, 0, 0

    if max_files:
        pdf_files = pdf_files[:max_files]

    print(f"Found {len(pdf_files)} PDF files")

    # Initialize processors
    pdf_processor = PDFProcessor()
    annotator = InvoiceAnnotator()
    model_config = ModelConfig()

    # Process PDFs
    examples = []

    print("Processing PDFs and extracting text...")
    for pdf_file in tqdm(pdf_files):
        try:
            # Extract text from PDF
            text = pdf_processor.extract_text(pdf_file)

            if not text.strip():
                print(f"Warning: No text extracted from {pdf_file.name}")
                continue

            # Auto-annotate
            annotations = annotator.auto_annotate(text)

            # Convert to tokens and BIO labels
            tokens, labels = text_to_bio(text, annotations, model_config)

            examples.append({
                "id": pdf_file.stem,
                "text": text,
                "tokens": tokens,
                "labels": labels,
                "metadata": {
                    "source": "company_documents",
                    "filename": pdf_file.name,
                    "annotations": annotations,
                },
            })

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            continue

    if not examples:
        print("No examples created. Check your PDF files.")
        return 0, 0, 0

    print(f"\nCreated {len(examples)} examples")

    # Split data
    total = len(examples)
    train_end = int(total * train_split)
    val_end = int(total * (train_split + val_split))

    train_examples = examples[:train_end]
    val_examples = examples[train_end:val_end]
    test_examples = examples[val_end:]

    # Save splits
    splits = {
        "train": train_examples,
        "val": val_examples,
        "test": test_examples,
    }

    for split_name, split_examples in splits.items():
        if split_examples:
            output_path = output_dir / f"{split_name}.json"
            with open(output_path, "w") as f:
                json.dump(split_examples, f, indent=2)
            print(f"Saved {len(split_examples)} {split_name} examples to {output_path}")

    return len(train_examples), len(val_examples), len(test_examples)


def text_to_bio(text: str, annotations: Dict, model_config: ModelConfig) -> Tuple[List[str], List[str]]:
    """
    Convert text and annotations to BIO format.

    Args:
        text: Invoice text
        annotations: Dict mapping entity types to spans
        model_config: Model configuration with label mappings

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
    field_mapping = {
        "invoice_number": "INVOICE_NUMBER",
        "date": "DATE",
        "total": "TOTAL",
        "customer_name": "CUSTOMER_NAME",
        "customer_address": "CUSTOMER_ADDRESS",
        "item": "ITEM",
        "item_price": "ITEM_PRICE",
    }

    for field_type, spans in annotations.items():
        if not isinstance(spans, list):
            spans = [spans]

        # Map to model label
        entity_type = field_mapping.get(field_type, field_type.upper())

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
                    label = f"{prefix}{entity_type}"

                    # Only apply if label is in our label set
                    if label in model_config.labels:
                        labels[token_idx] = label

                    is_first = False

    return tokens, labels


def create_annotation_file(
    pdf_dir: Path,
    output_file: Path,
    max_files: int = 10,
):
    """
    Create a JSON file with auto-annotations for manual review.

    Args:
        pdf_dir: Directory containing invoice PDFs
        output_file: Output JSON file path
        max_files: Maximum number of files to process
    """
    pdf_dir = Path(pdf_dir)
    output_file = Path(output_file)

    pdf_processor = PDFProcessor()
    annotator = InvoiceAnnotator()

    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))[:max_files]

    print(f"Creating annotation file for {len(pdf_files)} PDFs...")

    annotations_data = []

    for pdf_file in tqdm(pdf_files):
        try:
            text = pdf_processor.extract_text(pdf_file)
            annotations = annotator.auto_annotate(text)

            annotations_data.append({
                "filename": pdf_file.name,
                "text": text,
                "annotations": annotations,
            })

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(annotations_data, f, indent=2)

    print(f"\nAnnotation file saved to: {output_file}")
    print("Review and correct annotations, then use this file for preprocessing.")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(
        description="Preprocess Company Documents invoice dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/invoices",
        help="Directory containing invoice PDFs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process",
    )
    parser.add_argument(
        "--create-annotation-file",
        action="store_true",
        help="Create annotation file for manual review",
    )
    parser.add_argument(
        "--annotation-output",
        type=str,
        default="data/annotations.json",
        help="Output file for annotations",
    )

    args = parser.parse_args()

    if args.create_annotation_file:
        # Create annotation file for manual review
        create_annotation_file(
            Path(args.input_dir),
            Path(args.annotation_output),
            max_files=args.max_files or 10,
        )
    else:
        # Process PDFs directly
        train_count, val_count, test_count = process_invoice_pdfs(
            pdf_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            train_split=args.train_split,
            val_split=args.val_split,
            max_files=args.max_files,
        )

        print("\n" + "=" * 70)
        print("Preprocessing complete!")
        print("=" * 70)
        print(f"Training examples: {train_count}")
        print(f"Validation examples: {val_count}")
        print(f"Test examples: {test_count}")
        print("\nNext step: Train the model")
        print("  uv run train-invoice --data-dir data/processed")
        print("=" * 70)


if __name__ == "__main__":
    main()
