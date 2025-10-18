#!/usr/bin/env python3
"""
Simple inference script for invoice extraction using trained JSON model.
Loads the model and runs inference on PDFs in the data directory.

Usage:
    uv run python simple_inference.py
"""

import json
from pathlib import Path
from training.inference_json import ConstrainedJSONExtractor


def main():
    # Configuration
    model_path = "models/json_model/final_model"  # Path to the trained model
    data_dir = "data/raw/invoices"  # Directory containing PDF invoices
    num_pdfs = 3  # Number of PDFs to process (set to None to process all)

    # Initialize the extractor
    # use_constraints=False because T5 doesn't fully support constrained generation
    print(f"Loading model from: {model_path}")
    extractor = ConstrainedJSONExtractor(Path(model_path), use_constraints=False)

    # Get PDFs in the data directory
    all_pdfs = sorted(Path(data_dir).glob("*.pdf"))
    pdf_files = all_pdfs[:num_pdfs] if num_pdfs else all_pdfs

    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return

    print(f"\nFound {len(pdf_files)} PDFs to process\n")

    # Process each PDF
    for pdf_path in pdf_files:
        print("=" * 80)
        print(f"Processing: {pdf_path.name}")
        print("=" * 80)

        try:
            # Run inference
            result, inference_time = extractor.extract_from_pdf(pdf_path)

            # Pretty print the result
            print(f"\nInference time: {inference_time:.4f} seconds ({inference_time*1000:.2f} ms)")
            print("\nExtracted Invoice Data:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print()

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
