# Complete Usage Guide

This guide walks you through the entire workflow from setup to inference.

## Prerequisites

- Python 3.14
- [uv](https://github.com/astral-sh/uv) package manager
- Kaggle account (for dataset download)

## Step-by-Step Workflow

### 1. Initial Setup

Install dependencies:

```bash
uv sync
```

### 2. Set Up Kaggle Credentials

Before downloading the dataset, set up your Kaggle API credentials:

1. Log in to [Kaggle](https://www.kaggle.com)
2. Go to Account Settings: https://www.kaggle.com/settings/account
3. Scroll to the "API" section
4. Click "Create New Token" - this downloads `kaggle.json`
5. Install the credentials:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download Dataset

Download the Company Documents Dataset:

```bash
uv run download-dataset --extract-invoices --analyze
```

**What this does:**
- Downloads the dataset from Kaggle (requires credentials)
- Extracts invoice PDFs to `data/raw/invoices/`
- Shows statistics about the downloaded files

**Expected output:**
```
Downloading dataset: ayoubcherguelaine/company-documents-dataset
Destination: data/raw

Dataset downloaded successfully!

Dataset structure:
  invoices/ (XX files)
  ...

Extracted XX invoice files
Invoices extracted to: data/raw/invoices

Dataset Statistics
======================================================================
Total files: XX
PDF files: XX
...
```

### 4. Preprocess Invoice PDFs

Convert PDFs to training format with auto-annotation:

```bash
uv run preprocess-dataset --input-dir data/raw/invoices --output-dir data/processed
```

**What this does:**
- Extracts text from each PDF invoice
- Auto-detects fields using regex patterns:
  - Invoice numbers (e.g., "INV-001", "Invoice #123")
  - Dates (various formats)
  - Total amounts
  - Customer names
- Converts to BIO format for token classification
- Splits data into train/val/test sets (80/10/10 by default)

**Options:**
```bash
# Process only first 20 invoices (for testing)
uv run preprocess-dataset --input-dir data/raw/invoices --max-files 20

# Custom split ratios
uv run preprocess-dataset \
  --input-dir data/raw/invoices \
  --train-split 0.7 \
  --val-split 0.15

# Create annotation file for manual review
uv run preprocess-dataset \
  --input-dir data/raw/invoices \
  --create-annotation-file \
  --annotation-output data/annotations.json \
  --max-files 10
```

**Expected output:**
```
Found XX PDF files
Processing PDFs and extracting text...
100%|███████████████████| XX/XX [00:XX<00:00, X.XXit/s]

Created XX examples

Saved XX train examples to data/processed/train.json
Saved XX val examples to data/processed/val.json
Saved XX test examples to data/processed/test.json

Preprocessing complete!
======================================================================
Training examples: XX
Validation examples: XX
Test examples: XX
```

### 5. Train the Model

Train DistilBERT on your processed data:

```bash
uv run train-invoice \
  --data-dir data/processed \
  --output-dir models \
  --num-epochs 10 \
  --batch-size 8
```

**What this does:**
- Loads DistilBERT base model
- Fine-tunes on invoice data
- Validates after each epoch
- Saves best model based on F1 score
- Creates checkpoints every 500 steps

**Training parameters:**
- `--num-epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 8, reduce if OOM)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--max-length`: Maximum sequence length (default: 512)
- `--model-name`: Pretrained model (default: distilbert-base-uncased)

**Expected output:**
```
Using device: cuda  # or cpu

Loading tokenizer: distilbert-base-uncased
Loading datasets...
Train examples: XX
Validation examples: XX

Initializing model...
Training on device: cuda

Starting training...

==================================================
Epoch 1/10
==================================================
Epoch 1: 100%|███████| XX/XX [XX:XX<00:00, X.XXit/s, loss=X.XXXX]
Train loss: X.XXXX

Evaluating on validation set...
Validation loss: X.XXXX
Precision: 0.XXXX
Recall: 0.XXXX
F1: 0.XXXX
Exact match: 0.XXXX

New best F1: 0.XXXX
Model saved to models/best_model
...

Training complete!
Best F1 score: 0.XXXX
Models saved to: models
```

**Resume training from checkpoint:**
```bash
uv run train-invoice \
  --resume checkpoints/checkpoint_epoch_5.pt \
  --num-epochs 15
```

### 6. Extract Fields from Invoices

Use your trained model to extract fields:

```bash
# From text file
uv run extract-invoice \
  --model-path models/best_model \
  --input examples/sample_invoice.txt

# From PDF
uv run extract-invoice \
  --model-path models/best_model \
  --input path/to/invoice.pdf \
  --output results.json
```

**Expected output:**
```json
{
  "invoice_number": "INV-2024-0001",
  "date": "January 15, 2024",
  "customer": {
    "name": "John Smith",
    "address": "123 Main Street Springfield IL 62701"
  },
  "items": [
    {
      "name": "Laptop Computer",
      "price": "$1,299.99"
    },
    {
      "name": "Wireless Mouse",
      "price": "$29.99"
    }
  ],
  "total": "$1,345.97"
}
```

**Options:**
```bash
# Get detailed entity-level output
uv run extract-invoice \
  --model-path models/best_model \
  --input invoice.pdf \
  --format detailed
```

### 7. Evaluate Model Performance

Evaluate on test set:

```bash
uv run evaluate-invoice \
  --model-path models/best_model \
  --test-data data/processed/test.json \
  --output evaluation_results.json
```

**Expected output:**
```
Using device: cuda
Loading model from models/best_model
Loading test data from data/processed/test.json
Test examples: XX

Evaluating model...

======================================================================
Evaluation Results
======================================================================
Test loss: X.XXXX
Precision: 0.XXXX
Recall: 0.XXXX
F1 Score: 0.XXXX
Exact Match: 0.XXXX

Per-entity metrics:
  TOTAL                - P: 0.XXXX, R: 0.XXXX, F1: 0.XXXX
  ITEM                 - P: 0.XXXX, R: 0.XXXX, F1: 0.XXXX
  ITEM_PRICE           - P: 0.XXXX, R: 0.XXXX, F1: 0.XXXX
  CUSTOMER_NAME        - P: 0.XXXX, R: 0.XXXX, F1: 0.XXXX
  CUSTOMER_ADDRESS     - P: 0.XXXX, R: 0.XXXX, F1: 0.XXXX
  DATE                 - P: 0.XXXX, R: 0.XXXX, F1: 0.XXXX
  INVOICE_NUMBER       - P: 0.XXXX, R: 0.XXXX, F1: 0.XXXX

Detailed classification report:
...
```

## Quick Start (All Steps)

Run everything in one go:

```bash
./quickstart.sh
```

This executes all steps automatically.

## Common Workflows

### Testing with Small Dataset

```bash
# Download and extract
uv run download-dataset --extract-invoices

# Process only 10 invoices
uv run preprocess-dataset --input-dir data/raw/invoices --max-files 10

# Quick training (3 epochs)
uv run train-invoice --data-dir data/processed --num-epochs 3 --batch-size 4

# Test extraction
uv run extract-invoice --model-path models/best_model --input data/raw/invoices/invoice_001.pdf
```

### Improving Annotations

1. Create annotation file:
```bash
uv run preprocess-dataset \
  --input-dir data/raw/invoices \
  --create-annotation-file \
  --annotation-output data/annotations.json \
  --max-files 20
```

2. Review and edit `data/annotations.json` manually

3. The auto-annotation patterns can be improved in [training/preprocess_data.py](training/preprocess_data.py) in the `InvoiceAnnotator` class

### Batch Inference

```python
from pathlib import Path
from training.inference import InvoiceExtractor

# Load model
extractor = InvoiceExtractor(Path("models/best_model"))

# Process multiple invoices
invoice_files = list(Path("data/raw/invoices").glob("*.pdf"))

for invoice_file in invoice_files:
    entities = extractor.extract_from_pdf(invoice_file)
    output = extractor.format_output(entities)

    # Save results
    output_file = Path("results") / f"{invoice_file.stem}.json"
    output_file.parent.mkdir(exist_ok=True)

    import json
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
```

## Troubleshooting

### Kaggle API Not Working

**Error**: "Kaggle API credentials not found"

**Solution**:
1. Verify `kaggle.json` exists: `ls ~/.kaggle/kaggle.json`
2. Check permissions: `ls -la ~/.kaggle/kaggle.json` (should be 600)
3. Ensure file contains valid JSON with username and key

### Out of Memory During Training

**Error**: "CUDA out of memory"

**Solution**:
```bash
# Reduce batch size
uv run train-invoice --batch-size 4

# Or use CPU
uv run train-invoice --batch-size 8  # Will auto-detect no GPU
```

### No Text Extracted from PDFs

**Error**: "Warning: No text extracted from invoice.pdf"

**Possible causes**:
- PDF is scanned image (no text layer)
- PDF is corrupted

**Solutions**:
- Use OCR preprocessing for scanned PDFs
- Verify PDF opens correctly in PDF viewer
- Check PDF with: `pdfinfo invoice.pdf`

### Low Accuracy

**Solutions**:
1. Increase training epochs: `--num-epochs 20`
2. Add more training data
3. Review and correct auto-annotations
4. Improve regex patterns in `InvoiceAnnotator`
5. Try different model: `--model-name bert-base-uncased`

## File Structure After Setup

```
training/
├── data/
│   ├── raw/
│   │   ├── CompanyDocuments/      # Downloaded dataset
│   │   └── invoices/              # Extracted invoice PDFs
│   ├── processed/
│   │   ├── train.json             # Training data
│   │   ├── val.json               # Validation data
│   │   └── test.json              # Test data
│   └── annotations.json           # Optional: Manual annotations
├── models/
│   ├── best_model/                # Best model during training
│   └── final_model/               # Final model after all epochs
├── checkpoints/
│   └── checkpoint_epoch_*.pt      # Training checkpoints
└── results/                       # Inference results (manual)
```

## Next Steps

- Fine-tune regex patterns in `InvoiceAnnotator` for your invoice formats
- Add more entity types in [training/config.py](training/config.py)
- Experiment with different models (BERT, RoBERTa, etc.)
- Create custom evaluation metrics for your use case
- Build a web interface for inference
