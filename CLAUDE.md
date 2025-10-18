# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

### 1. Always Use `uv` for All Project Actions

**MANDATORY**: This project uses `uv` for ALL operations, not `pip` or direct `python` commands.

- **Installing dependencies**: `uv sync` (NOT `pip install`)
- **Running Python scripts**: `uv run python script.py` (NOT `python script.py`)
- **Running modules**: `uv run python -m module.name` (NOT `python -m module.name`)
- **Installing packages**: `uv add package-name` (NOT `pip install package-name`)
- **Removing packages**: `uv remove package-name` (NOT `pip uninstall package-name`)

**Examples:**
```bash
# Correct
uv sync
uv run python verify_setup.py
uv run python -m training.train_json
uv run train-invoice --data-dir data/processed

# Incorrect - DO NOT USE
pip install transformers
python verify_setup.py
python -m training.train_json
```

**Why**: Using `uv` ensures proper virtual environment isolation, dependency management, and consistency across development environments. Direct `python` or `pip` commands may use the wrong environment or dependencies.

## Project Overview

Invoice field extraction system with **two distinct approaches**:

1. **Method 1 (Token Classification)**: Fast DistilBERT-based NER using BIO tagging for fixed schemas
2. **Method 2 (Constrained JSON)**: T5/Flan-T5 with grammar-based constrained decoding using `outlines` library for guaranteed valid JSON output

Both methods extract invoice fields (total, items, customer details, dates, invoice numbers) from PDFs and text.

## Essential Commands

### Package Management
```bash
# Install all dependencies (uses uv, not pip)
uv sync

# Verify installation and environment
uv run python verify_setup.py
```

### Method 1: Token Classification (DistilBERT)

```bash
# Download dataset
uv run download-dataset --extract-invoices --analyze

# Preprocess PDFs into BIO-tagged format
uv run preprocess-dataset --input-dir data/raw/invoices --output-dir data/processed

# Train model
uv run train-invoice \
  --data-dir data/processed \
  --output-dir models \
  --num-epochs 10 \
  --batch-size 32  # Adjust based on available RAM/GPU

# Run inference
uv run extract-invoice \
  --model-path models/best_model \
  --input path/to/invoice.pdf

# Evaluate model
uv run evaluate-invoice \
  --model-path models/best_model \
  --test-data data/processed/test.json

# Quick start (all steps automated)
./quickstart.sh
```

### Method 2: Constrained JSON Generation (T5/Flan-T5)

```bash
# Convert BIO data to JSON format and train
uv run python -m training.train_json \
    --data-dir data/processed \
    --convert-data \
    --batch-size 8 \
    --num-epochs 10

# Run inference with constraints (guaranteed valid JSON)
uv run python -m training.inference_json \
    --model-path models/json_model/best_model \
    --input invoice.pdf

# Run without constraints (for comparison)
uv run python -m training.inference_json \
    --model-path models/json_model/best_model \
    --input invoice.pdf \
    --no-constraints

# Test demo
uv run python examples/test_constrained_json.py
```

### Testing

```bash
# Create sample data for testing without downloading dataset
uv run python examples/create_sample_data.py

# Process limited files for testing
uv run preprocess-dataset --input-dir data/raw/invoices --max-files 20
```

## Architecture

### Code Organization

**Method 1 (Token Classification):**
- `training/model.py`: DistilBERT wrapper with `InvoiceExtractionModel` and `ModelTrainer`
- `training/dataset.py`: PyTorch Dataset for BIO-tagged token classification
- `training/train.py`: Training loop with checkpointing
- `training/inference.py`: Inference pipeline with `InvoiceExtractor` class
- `training/metrics.py`: Token-level evaluation (precision, recall, F1)

**Method 2 (JSON Generation):**
- `training/model_json.py`: T5 wrapper with `InvoiceJSONExtractionModel` and `JSONModelTrainer`
- `training/json_schema.py`: Pydantic schema definitions (`InvoiceOutput`, `CustomerInfo`, `InvoiceItem`)
- `training/dataset_json.py`: Seq2seq dataset for JSON generation
- `training/train_json.py`: Training loop for T5 models
- `training/inference_json.py`: **Constrained generation using `outlines` library** - guarantees valid JSON
- `training/metrics_json.py`: JSON-specific metrics (exact match, field accuracy, JSON validity)
- `training/bio_to_json_converter.py`: Converts BIO format to JSON
- `training/convert_data_to_json.py`: Batch data conversion utility

**Shared Components:**
- `training/config.py`: `ModelConfig`, `JSONModelConfig`, `DataConfig`, `TrainingConfig` dataclasses
- `training/pdf_processor.py`: PDF text extraction using PyMuPDF
- `training/preprocess_data.py`: Auto-annotation with regex pattern matching
- `training/download_dataset.py`: Kaggle dataset downloader

### Data Flow

**Method 1:**
```
PDF → text extraction → regex auto-annotation → BIO tagging →
DistilBERT tokenization → token classification → entity extraction → JSON output
```

**Method 2:**
```
PDF → text extraction → BIO format → JSON conversion →
T5 tokenization → seq2seq generation → constrained decoding → validated JSON output
```

### Model Architectures

**Method 1 (DistilBERT):**
- Token classification with 15 labels (7 entity types × 2 for BIO + O)
- Max sequence: 512 tokens
- ~50ms/invoice
- ~1GB GPU memory
- Optimizer: AdamW with linear warmup

**Method 2 (T5/Flan-T5):**
- Sequence-to-sequence generation (220-250M params)
- Max input: 512 tokens, max output: 512 tokens
- ~70ms/invoice with constraints
- ~4GB GPU memory
- **Constrained decoding**: Uses `outlines` library with Pydantic schema to enforce JSON grammar
- **100% valid JSON guarantee** when constraints enabled
- Optimizer: AdamW with warmup

### Device Priority

Automatic device selection: **CUDA (NVIDIA) > MPS (Apple Silicon) > CPU**

- Both `ModelTrainer` and `JSONModelTrainer` implement `_get_device()` method
- MPS (Metal Performance Shaders) provides 3-10x speedup on M1/M2/M3/M4 Macs
- Default batch sizes in `config.py` optimized for M3 Max with 64GB RAM
- See `MAC_SETUP.md` for hardware-specific tuning

### Label Schema (BIO Tagging)

Method 1 uses BIO format defined in `training/config.py`:
- `O`: Outside any entity
- `B-TOTAL`, `I-TOTAL`: Total price
- `B-ITEM`, `I-ITEM`: Item names
- `B-ITEM_PRICE`, `I-ITEM_PRICE`: Item prices
- `B-CUSTOMER_NAME`, `I-CUSTOMER_NAME`: Customer name
- `B-CUSTOMER_ADDRESS`, `I-CUSTOMER_ADDRESS`: Customer address
- `B-DATE`, `I-DATE`: Invoice date
- `B-INVOICE_NUMBER`, `I-INVOICE_NUMBER`: Invoice number

Method 2 uses Pydantic schema in `training/json_schema.py` with nested structure.

## Important Implementation Details

### Constrained Decoding (Method 2)

The key innovation in Method 2 is **grammar-based constrained generation**:

1. **Pydantic Schema Definition**: `json_schema.py` defines structured output using Pydantic models
2. **Outlines Integration**: `inference_json.py` uses `outlines.from_transformers()` to wrap the T5 model
3. **Grammar Enforcement**: During generation, the `outlines` library constrains the model's token selection to only produce valid JSON conforming to the schema
4. **Zero Invalid JSON**: Unlike regular seq2seq, constrained decoding guarantees syntactically valid JSON

**Limitation**: As noted in `inference_json.py:86-90`, constrained generation has limited support for seq2seq models (T5). The implementation includes fallback to unconstrained generation with a warning.

### Auto-Annotation

`preprocess_data.py` uses regex patterns to automatically detect:
- Invoice numbers: `INV-\d+`, `Invoice #\d+`
- Dates: Multiple formats (MM/DD/YYYY, Month DD YYYY, etc.)
- Totals: Currency amounts near keywords ("Total", "Amount Due")
- Customer names: Text following "Bill To", "Customer"

These auto-annotations create initial BIO tags that can be manually reviewed before training.

### Checkpoint Management

Both trainers (`model.py` and `model_json.py`) implement:
- `save_checkpoint()`: Saves epoch, step, model state, optimizer state, best metric
- `load_checkpoint()`: Resumes training from checkpoint
- Checkpoints saved to `checkpoints/` directory
- Best model saved to `models/best_model` or `models/json_model/best_model`

### Running All Code with uv

**Critical**: All Python execution uses `uv run` prefix, not direct `python` calls:
- Correct: `uv run python -m training.train_json`
- Incorrect: `python -m training.train_json`

This ensures proper virtual environment and dependency isolation.

## Common Development Tasks

### Adding New Entity Types

1. Update `labels` list in `training/config.py` (Method 1) or `json_schema.py` (Method 2)
2. Update auto-annotation patterns in `training/preprocess_data.py`
3. Regenerate training data
4. Retrain model

### Debugging Performance Issues

- OOM errors: Reduce `batch_size` in training commands
- Slow training: Check device with `verify_setup.py`, ensure GPU/MPS detected
- Low accuracy: Increase `num_epochs`, verify annotation quality, check class imbalance in metrics

### Switching Between Methods

Method 1 → Method 2:
```bash
uv run python -m training.convert_data_to_json \
    --input-dir data/processed \
    --output-dir data/json
```

Both methods can coexist - they use different model directories and data formats.

## Dataset Information

**Default**: Company Documents Dataset from Kaggle (`ayoubcherguelaine/company-documents-dataset`)
- Real invoice PDFs from various companies
- Located in `data/raw/invoices/` after extraction
- Requires Kaggle API credentials in `~/.kaggle/kaggle.json`
- See `DATASET_INFO.md` for details

## Documentation Map

- `README.md`: Main documentation, quick start for both methods
- `USAGE_GUIDE.md`: Complete guide for Method 1
- `QUICK_START_JSON.md`: Quick start for Method 2
- `JSON_GENERATION_README.md`: Detailed Method 2 documentation
- `METHOD_COMPARISON.md`: Side-by-side comparison of both approaches
- `MAC_SETUP.md`: Mac-specific optimization guide
- `M3_MAX_PERFORMANCE.md`: Performance tuning for M3 Max
- `DOCUMENTATION_INDEX.md`: Navigation guide to all docs
