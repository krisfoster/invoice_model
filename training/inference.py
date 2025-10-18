"""Inference script for invoice field extraction."""

import argparse
import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer

from training.config import ModelConfig
from training.model import InvoiceExtractionModel
from training.metrics import InvoiceMetrics
from training.pdf_processor import PDFProcessor


class InvoiceExtractor:
    """Extract fields from invoices using trained model."""

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
    ):
        """
        Initialize extractor.

        Args:
            model_path: Path to trained model
            device: Device to run inference on ("auto", "cuda", "mps", "cpu")
        """
        self.device = self._get_device(device)

        print(f"Loading model from {model_path}")
        self.model = InvoiceExtractionModel.load(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Get label mappings from model config
        config = self.model.model.config
        self.id2label = config.id2label
        self.label2id = config.label2id

        # Initialize metrics calculator for entity extraction
        self.metrics = InvoiceMetrics(self.id2label)

        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()

        print(f"Model loaded successfully")

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

    def extract_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract fields from plain text.

        Args:
            text: Invoice text

        Returns:
            Dictionary mapping field types to extracted values
        """
        # Tokenize
        tokens = text.split()
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs["logits"], dim=-1)

        # Convert predictions to list
        predictions = predictions[0].cpu().tolist()

        # Get word ids for alignment
        word_ids = encoding.word_ids(batch_index=0)

        # Map subword predictions back to word-level predictions
        word_predictions = []
        for word_idx in range(len(tokens)):
            # Get all subword predictions for this word
            subword_preds = [
                predictions[i]
                for i, wid in enumerate(word_ids)
                if wid == word_idx
            ]
            # Use the first subword's prediction for the whole word
            if subword_preds:
                word_predictions.append(subword_preds[0])
            else:
                # This shouldn't happen, but fallback to O label
                word_predictions.append(self.label2id["O"])

        # Extract entities
        entities = self.metrics.extract_entities(tokens, word_predictions)

        return entities

    def extract_from_pdf(self, pdf_path: Path) -> Dict[str, List[str]]:
        """
        Extract fields from PDF invoice.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary mapping field types to extracted values
        """
        # Extract text from PDF
        text = self.pdf_processor.extract_text(pdf_path)
        print(f">> {text}")
        # Extract fields
        entities = self.extract_from_text(text)
        print(f">> {entities}")
        return entities

    def extract_batch(
        self, texts: List[str], batch_size: int = 8
    ) -> List[Dict[str, List[str]]]:
        """
        Extract fields from multiple texts.

        Args:
            texts: List of invoice texts
            batch_size: Batch size for processing

        Returns:
            List of dictionaries with extracted fields
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            for text in batch_texts:
                entities = self.extract_from_text(text)
                results.append(entities)

        return results

    def format_output(self, entities: Dict[str, List[str]]) -> Dict[str, any]:
        """
        Format extracted entities into structured output.

        Args:
            entities: Extracted entities

        Returns:
            Formatted output dictionary
        """
        output = {
            "invoice_number": entities.get("INVOICE_NUMBER", [None])[0],
            "date": entities.get("DATE", [None])[0],
            "customer": {
                "name": entities.get("CUSTOMER_NAME", [None])[0],
                "address": " ".join(entities.get("CUSTOMER_ADDRESS", [])),
            },
            "items": [],
            "total": entities.get("TOTAL", [None])[0],
        }

        # Combine items and prices
        items = entities.get("ITEM", [])
        prices = entities.get("ITEM_PRICE", [])

        for i, item in enumerate(items):
            price = prices[i] if i < len(prices) else None
            output["items"].append({"name": item, "price": price})

        return output


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Extract fields from invoices")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to invoice (PDF or text file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save extracted fields (JSON)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["simple", "detailed"],
        default="simple",
        help="Output format",
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = InvoiceExtractor(Path(args.model_path))

    # Extract fields
    input_path = Path(args.input)

    print(f"Processing: {input_path}")

    # Time the inference (excluding model loading)
    start_time = time.perf_counter()

    if input_path.suffix.lower() == ".pdf":
        entities = extractor.extract_from_pdf(input_path)
    else:
        with open(input_path, "r") as f:
            text = f.read()
        entities = extractor.extract_from_text(text)

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    print(f"Inference time: {inference_time:.4f} seconds ({inference_time*1000:.2f} ms)")

    # Format output
    if args.format == "simple":
        output = extractor.format_output(entities)
    else:
        output = entities

    # Print or save results
    output_json = json.dumps(output, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output_json)
        print(f"Results saved to: {output_path}")
    else:
        print("\nExtracted fields:")
        print(output_json)


if __name__ == "__main__":
    main()
