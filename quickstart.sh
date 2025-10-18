#!/bin/bash
# Quick start script for invoice extraction training

set -e

echo "======================================================================"
echo "Invoice Field Extraction - Quick Start"
echo "======================================================================"
echo ""

# Step 1: Download dataset
echo "Step 1: Downloading Company Documents Dataset from Kaggle..."
echo "----------------------------------------------------------------------"
uv run download-dataset --extract-invoices --analyze

echo ""
echo "Step 2: Preprocessing invoice PDFs..."
echo "----------------------------------------------------------------------"
uv run preprocess-dataset --input-dir data/raw/invoices --output-dir data/processed

echo ""
echo "Step 3: Training model (this may take a while)..."
echo "----------------------------------------------------------------------"
uv run train-invoice \
  --data-dir data/processed \
  --output-dir models \
  --num-epochs 5 \
  --batch-size 8

echo ""
echo "======================================================================"
echo "Training complete!"
echo "======================================================================"
echo ""
echo "To extract fields from an invoice:"
echo "  uv run extract-invoice --model-path models/best_model --input path/to/invoice.pdf"
echo ""
echo "To evaluate the model:"
echo "  uv run evaluate-invoice --model-path models/best_model --test-data data/processed/test.json"
echo ""
