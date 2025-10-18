#!/usr/bin/env python3
"""
Minimal T5 inference example using only standard transformers library.
No custom classes - just raw transformers AutoModel classes and PyMuPDF.

Uses AutoModelForSeq2SeqLM and AutoTokenizer which automatically detect
the model architecture from config.json. This is the recommended way to
load models as it works with any seq2seq model (T5, BART, mT5, etc.).

This demonstrates how to use the trained model with minimal dependencies
and without any of the custom training code.

=============================================================================
USAGE INSTRUCTIONS
=============================================================================

1. Quick Start:
   Run the script with default settings to process 3 sample invoices:

   $ uv run python examples/minimal_t5_inference.py

2. Customize Configuration:
   Edit the variables in the main() function:

   - model_path: Path to your trained model
     Default: "models/json_model/final_model"

   - data_dir: Directory containing PDF invoices
     Default: "data/raw/invoices"

   - num_pdfs: Number of PDFs to process (None = process all)
     Default: 3

3. Using in Your Own Code:
   Copy the relevant functions into your project:

   from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
   import torch

   # Load the model
   model = AutoModelForSeq2SeqLM.from_pretrained("path/to/model")
   tokenizer = AutoTokenizer.from_pretrained("path/to/model")
   model.to("cuda")  # or "mps" or "cpu"
   model.eval()

   # Prepare input
   input_text = f"extract invoice fields: {your_invoice_text}"
   inputs = tokenizer(input_text, return_tensors="pt", max_length=512,
                      truncation=True).to("cuda")

   # Generate output
   with torch.no_grad():
       outputs = model.generate(
           input_ids=inputs["input_ids"],
           attention_mask=inputs["attention_mask"],
           max_new_tokens=512,
       )

   # Decode and parse JSON
   import json
   result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
   result = json.loads(result_text)

4. Dependencies:
   This script requires:
   - transformers (for AutoModel classes)
   - torch (for inference)
   - PyMuPDF/fitz (for PDF text extraction)

   All are included in this project's dependencies.

5. Output Format:
   The model outputs JSON with the following structure:
   {
     "invoice_number": "10248",
     "date": "2016-07-04",
     "customer": {
       "name": "VINET",
       "address": null
     },
     "items": [],
     "total": "440.0"
   }

=============================================================================
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using PyMuPDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def load_model(model_path: str, device: str = "auto"):
    """
    Load model and tokenizer using AutoModel classes.

    AutoModelForSeq2SeqLM and AutoTokenizer automatically detect
    the model type (T5, BART, etc.) from the config.

    Args:
        model_path: Path to saved model directory
        device: Device to use ("auto", "cuda", "mps", "cpu")

    Returns:
        Tuple of (model, tokenizer, device)
    """
    # Determine best device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading model from: {model_path}")

    # Load model and tokenizer using Auto classes
    # This automatically detects the model architecture from config.json
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")

    return model, tokenizer, device


def extract_invoice_fields(text: str, model, tokenizer, device: str) -> dict:
    """
    Extract invoice fields from text using the T5 model.

    Args:
        text: Invoice text
        model: T5 model
        tokenizer: T5 tokenizer
        device: Device the model is on

    Returns:
        Dictionary with extracted invoice fields
    """
    # Prepare input with the same format used during training
    input_text = f"extract invoice fields: {text}"

    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            num_beams=1,  # Greedy decoding
            do_sample=False,  # Deterministic
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse JSON output
    try:
        result = json.loads(generated_text)
        return result
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        fixed_text = fix_json(generated_text)
        try:
            result = json.loads(fixed_text)
            print("Note: Fixed malformed JSON")
            return result
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON: {e}")
            print(f"Generated text: {generated_text}")
            # Return empty structure
            return {
                "invoice_number": None,
                "date": None,
                "customer": {"name": None, "address": None},
                "items": [],
                "total": None,
            }


def fix_json(text: str) -> str:
    """
    Fix common JSON formatting issues from T5 output.

    Args:
        text: Potentially malformed JSON text

    Returns:
        Fixed JSON text
    """
    import re

    text = text.strip()

    # Add missing braces
    if not text.startswith('{'):
        text = '{' + text
    if not text.endswith('}'):
        text = text + '}'

    # Fix malformed customer field
    text = re.sub(r'"customer":\s*"name":', r'"customer": {"name":', text)
    text = re.sub(r'("address":\s*(?:"[^"]*"|null))\s*,\s*("items":)', r'\1}, \2', text)

    return text


def main():
    # Configuration
    model_path = "models/json_model/final_model"
    data_dir = "data/raw/invoices"
    num_pdfs = 3

    # Load model using standard transformers classes
    model, tokenizer, device = load_model(model_path)

    # Get PDFs
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))[:num_pdfs]

    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return

    print(f"\nProcessing {len(pdf_files)} PDFs\n")

    # Process each PDF
    for pdf_path in pdf_files:
        print("=" * 80)
        print(f"Processing: {pdf_path.name}")
        print("=" * 80)

        try:
            # Extract text from PDF
            text = extract_text_from_pdf(str(pdf_path))
            print(f"Extracted {len(text)} characters from PDF")

            # Extract invoice fields
            result = extract_invoice_fields(text, model, tokenizer, device)

            # Print result
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
