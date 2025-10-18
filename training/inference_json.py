"""Constrained JSON inference with outlines integration."""

import argparse
import json
import time
import torch
from pathlib import Path
from typing import Dict, Optional

from training.model_json import InvoiceJSONExtractionModel
from training.json_schema import InvoiceOutput, get_json_schema, validate_invoice_output
from training.pdf_processor import PDFProcessor


class ConstrainedJSONExtractor:
    """Extract invoice fields with constrained JSON generation."""

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        use_constraints: bool = True,
    ):
        """
        Initialize extractor.

        Args:
            model_path: Path to trained model
            device: Device to run inference on ("auto", "cuda", "mps", "cpu")
            use_constraints: Whether to use constrained generation
        """
        self.device = self._get_device(device)
        self.use_constraints = use_constraints

        print(f"Loading model from {model_path}")
        self.model = InvoiceJSONExtractionModel.load(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer

        print(f"Using device: {self.device}")
        print(f"Constrained generation: {self.use_constraints}")

        # Initialize constrained generator if enabled
        if self.use_constraints:
            self._setup_constrained_generator()

        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()

        print("Model loaded successfully")

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

    def _setup_constrained_generator(self):
        """Setup outlines for constrained generation."""
        try:
            import outlines

            # Wrap HuggingFace model for outlines
            self.outlines_model = outlines.models.Transformers(
                self.model.model,
                self.tokenizer,
            )

            # Get JSON schema
            schema = get_json_schema()

            # Create JSON-constrained generator
            self.json_generator = outlines.generate.json(
                self.outlines_model,
                schema,
            )

            print("Constrained generation setup complete")

        except ImportError:
            print("Warning: outlines not installed. Falling back to unconstrained generation.")
            print("Install with: pip install outlines")
            self.use_constraints = False
        except Exception as e:
            print(f"Warning: Failed to setup constrained generation: {e}")
            print("Falling back to unconstrained generation.")
            self.use_constraints = False

    def extract_from_text(self, text: str) -> Dict:
        """
        Extract fields from invoice text.

        Args:
            text: Invoice text

        Returns:
            Extracted invoice data as dictionary
        """
        # Prepare input with instruction prompt
        input_text = f"extract invoice fields: {text}"

        if self.use_constraints:
            # Use constrained generation (guaranteed valid JSON!)
            result = self._extract_constrained(input_text)
        else:
            # Use standard generation
            result = self._extract_unconstrained(input_text)

        return result

    def _extract_constrained(self, input_text: str) -> Dict:
        """
        Extract with constrained generation (guaranteed valid JSON).

        Args:
            input_text: Input text with prompt

        Returns:
            Extracted data (always valid according to schema)
        """
        try:
            # Generate with JSON schema constraints
            # This ensures output matches InvoiceOutput schema!
            result = self.json_generator(input_text)

            # Result is already a valid dict matching the schema
            return result

        except Exception as e:
            print(f"Error during constrained generation: {e}")
            # Fallback to unconstrained
            return self._extract_unconstrained(input_text)

    def _extract_unconstrained(self, input_text: str) -> Dict:
        """
        Extract with standard generation (may produce invalid JSON).

        Args:
            input_text: Input text with prompt

        Returns:
            Extracted data (best effort)
        """
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_beams=4,  # Use beam search for better quality
                early_stopping=True,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse JSON
        try:
            result = json.loads(generated_text)

            # Validate against schema
            is_valid, error = validate_invoice_output(result)
            if not is_valid:
                print(f"Warning: Generated JSON doesn't match schema: {error}")

            return result

        except json.JSONDecodeError as e:
            print(f"Warning: Generated invalid JSON: {e}")
            print(f"Generated text: {generated_text}")

            # Return empty structure
            return {
                "invoice_number": None,
                "date": None,
                "customer": {"name": None, "address": None},
                "items": [],
                "total": None,
            }

    def extract_from_pdf(self, pdf_path: Path) -> Dict:
        """
        Extract fields from PDF invoice.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted invoice data
        """
        # Extract text from PDF
        text = self.pdf_processor.extract_text(pdf_path)
        print(f"Extracted text from PDF ({len(text)} characters)")

        # Extract fields
        result = self.extract_from_text(text)

        return result

    def extract_batch(
        self,
        texts: list[str],
        batch_size: int = 4,
    ) -> list[Dict]:
        """
        Extract fields from multiple texts.

        Args:
            texts: List of invoice texts
            batch_size: Batch size for processing

        Returns:
            List of extracted invoice data
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            for text in batch_texts:
                result = self.extract_from_text(text)
                results.append(result)

        return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Extract invoice fields with constrained JSON generation"
    )
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
        "--no-constraints",
        action="store_true",
        help="Disable constrained generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for inference",
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = ConstrainedJSONExtractor(
        Path(args.model_path),
        device=args.device,
        use_constraints=not args.no_constraints,
    )

    # Extract fields
    input_path = Path(args.input)
    print(f"\nProcessing: {input_path}")

    # Time the inference
    start_time = time.perf_counter()

    if input_path.suffix.lower() == ".pdf":
        result = extractor.extract_from_pdf(input_path)
    else:
        with open(input_path, "r") as f:
            text = f.read()
        result = extractor.extract_from_text(text)

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    print(f"\nInference time: {inference_time:.4f} seconds ({inference_time*1000:.2f} ms)")

    # Validate result
    is_valid, error = validate_invoice_output(result)
    if is_valid:
        print("Generated output is valid according to schema")
    else:
        print(f"Warning: Generated output validation error: {error}")

    # Print or save results
    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output_json)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nExtracted fields:")
        print(output_json)


if __name__ == "__main__":
    main()
