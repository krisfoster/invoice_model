#!/usr/bin/env python3
"""Minimal self-contained inference script for invoice field extraction from PDFs."""

import json
import sys
from pathlib import Path

import fitz  # pymupdf
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def extract_invoice_fields(pdf_path: str, model_path: str = "../models/best_model") -> dict:
    """Extract invoice fields from PDF using trained DistilBERT model."""

    # Load model and tokenizer
    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Move to device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Extract text from PDF
    pdf = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in pdf)
    pdf.close()

    # Tokenize and run inference
    tokens = text.split()
    encoding = tokenizer(tokens, is_split_into_words=True, max_length=512, padding="max_length", return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids=encoding["input_ids"].to(device), attention_mask=encoding["attention_mask"].to(device))
        predictions = torch.argmax(outputs.logits, dim=-1)[0]

    # Map predictions to labels
    labels = model.config.id2label
    entities = {}
    current_entity = None
    current_tokens = []

    for pred, token in zip(predictions, tokens):
        label = labels[pred.item()]

        if label == "O":
            if current_entity:
                key = current_entity.replace("B-", "").replace("I-", "")
                if key not in entities:
                    entities[key] = []
                entities[key].append(" ".join(current_tokens))
                current_entity = None
                current_tokens = []
        elif label.startswith("B-"):
            if current_entity:
                key = current_entity.replace("B-", "").replace("I-", "")
                if key not in entities:
                    entities[key] = []
                entities[key].append(" ".join(current_tokens))
            current_entity = label
            current_tokens = [token]
        elif label.startswith("I-"):
            current_tokens.append(token)

    if current_entity:
        key = current_entity.replace("B-", "").replace("I-", "")
        if key not in entities:
            entities[key] = []
        entities[key].append(" ".join(current_tokens))

    return entities


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <pdf_path> [model_path]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "../models/best_model"

    result = extract_invoice_fields(pdf_path, model_path)
    print(json.dumps(result, indent=2))
