# Invoice Field Extraction with Deep Learning

A comprehensive invoice field extraction system with **two approaches**:
1. **Method 1 (Token Classification)**: Fast DistilBERT-based NER for fixed schemas
2. **Method 2 (Constrained JSON)**: T5/Flan-T5 with guaranteed valid JSON output

Extracts key fields from invoice PDFs and text including total price, items, customer details, dates, and invoice numbers.

**Default Dataset**: [Company Documents Dataset](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset) from Kaggle - automatically downloaded and preprocessed.

**Apple Silicon Optimized**: Native MPS (Metal Performance Shaders) support for 3-10x faster training. **Defaults optimized for M3 Max with 64GB RAM** - trains in 1-3 minutes per epoch! See [M3_MAX_PERFORMANCE.md](M3_MAX_PERFORMANCE.md) for performance guide or [MAC_SETUP.md](MAC_SETUP.md) for other Macs.

## Features

### Core Capabilities
- **PDF Processing**: Extract text and layout information from PDF invoices
- **Two Extraction Methods**:
  - **Method 1**: DistilBERT token classification (fast, fixed schema)
  - **Method 2**: T5/Flan-T5 with constrained JSON generation (**guaranteed valid JSON**)
- **Comprehensive Field Extraction**:
  - Total price
  - Invoice items and their prices
  - Customer name and address
  - Invoice date
  - Invoice number
- **Evaluation Metrics**: Precision, recall, F1-score, exact match, and JSON validation
- **Training Pipeline**: Complete training workflow with checkpointing and validation
- **CLI Tools**: Easy-to-use command-line interfaces for training, inference, and evaluation
- **Kaggle Integration**: Automatic dataset download from Kaggle with one command
- **Auto-annotation**: Regex-based pattern matching for semi-automated labeling
- **GPU Acceleration**: Automatic support for CUDA (NVIDIA) and MPS (Apple Silicon)

### Method 2 Special Features
- **Constrained Decoding**: Uses `outlines` library for grammar-based JSON generation
- **Schema Validation**: Pydantic models ensure output compliance
- **100% Valid JSON**: Structurally impossible to generate invalid JSON
- **Flexible Schema**: Easy to modify output structure without code changes

## Documentation Guide

**Lost? See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete navigation guide to all documentation.

**Quick links:**
- Method 1 (Token Classification): [USAGE_GUIDE.md](USAGE_GUIDE.md)
- Method 2 (Constrained JSON): [QUICK_START_JSON.md](QUICK_START_JSON.md)
- Compare Methods: [METHOD_COMPARISON.md](METHOD_COMPARISON.md)

## Project Structure

```
training/
├── training/                      # Main package
│   ├── config.py                  # Configuration (ModelConfig, JSONModelConfig)
│   ├── dataset.py                 # Dataset for Method 1 (token classification)
│   ├── model.py                   # DistilBERT model (Method 1)
│   ├── train.py                   # Training script (Method 1)
│   ├── inference.py               # Inference script (Method 1)
│   ├── evaluate.py                # Evaluation script
│   ├── metrics.py                 # Token-level metrics (Method 1)
│   ├── pdf_processor.py           # PDF text extraction
│   ├── preprocess_data.py         # Data preprocessing
│   │
│   ├── json_schema.py             # Pydantic JSON schema (Method 2)
│   ├── model_json.py              # T5/Flan-T5 model (Method 2)
│   ├── dataset_json.py            # Dataset for JSON generation (Method 2)
│   ├── train_json.py              # Training script (Method 2)
│   ├── inference_json.py          # Constrained inference (Method 2)
│   ├── metrics_json.py            # JSON-specific metrics (Method 2)
│   ├── bio_to_json_converter.py   # Convert BIO to JSON format
│   └── convert_data_to_json.py    # Data conversion script
│
├── examples/                      # Sample data and scripts
│   ├── sample_invoice.txt
│   ├── create_sample_data.py
│   └── test_constrained_json.py   # Demo for Method 2
│
├── pyproject.toml                 # Project dependencies
├── README.md                      # This file
├── USAGE_GUIDE.md                 # Complete usage guide (Method 1)
├── JSON_GENERATION_README.md      # Method 2 documentation
├── QUICK_START_JSON.md            # Quick start for Method 2
├── METHOD_COMPARISON.md           # Compare both methods
└── IMPLEMENTATION_SUMMARY.md      # Method 2 implementation overview
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and requires Python 3.14.

1. Install dependencies:
```bash
uv sync
```

2. Verify installation:
```bash
uv run python verify_setup.py
```

This checks:
- Python version
- All dependencies installed
- Kaggle API credentials
- CUDA availability
- Directory structure

All checks should pass before proceeding.

## Quick Start

### Choose Your Method

**Method 1 (Token Classification)** - Fast, fixed schema:
- Use when speed is critical
- Schema is fixed and won't change
- Running on limited hardware
- See [USAGE_GUIDE.md](USAGE_GUIDE.md) for complete guide

**Method 2 (Constrained JSON)** - Guaranteed valid JSON:
- Need 100% valid JSON output
- Schema may evolve over time
- Want flexible, nested structures
- See [QUICK_START_JSON.md](QUICK_START_JSON.md) for quick start
- See [JSON_GENERATION_README.md](JSON_GENERATION_README.md) for full docs

**Compare Methods**: See [METHOD_COMPARISON.md](METHOD_COMPARISON.md)

---

### Method 1 Quick Start

#### Option 1: Automated Setup (Recommended)

Run the quick start script to download data, preprocess, and train:

```bash
./quickstart.sh
```

This will:
1. Download the Company Documents Dataset from Kaggle
2. Extract and preprocess invoice PDFs
3. Train a DistilBERT model (Method 1)

#### Option 2: Step-by-Step Setup

#### 1. Set Up Kaggle API Credentials

To download the dataset, you need Kaggle API credentials:

1. Go to https://www.kaggle.com/settings/account
2. Scroll to 'API' section and click 'Create New Token'
3. This downloads `kaggle.json`
4. Move it to the correct location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 2. Download and Extract Dataset

```bash
uv run download-dataset --extract-invoices --analyze
```

This downloads the Company Documents Dataset and extracts invoices to `data/raw/invoices/`.

#### 3. Preprocess Invoice PDFs

```bash
uv run preprocess-dataset --input-dir data/raw/invoices --output-dir data/processed
```

This will:
- Extract text from PDF invoices
- Auto-annotate fields using pattern matching
- Create train/val/test splits in BIO format

#### Option 3: Test with Sample Data

Create sample training data to test the system without downloading:

```bash
uv run python examples/create_sample_data.py
```

This creates minimal sample data in `data/processed/`.

---

### Method 2 Quick Start

See [QUICK_START_JSON.md](QUICK_START_JSON.md) for complete Method 2 guide.

**TL;DR**:

```bash
# 1. Install dependencies (includes outlines and pydantic)
uv sync

# 2. Convert data and train
uv run python -m training.train_json \
    --data-dir data/processed \
    --convert-data \
    --batch-size 8 \
    --num-epochs 10

# 3. Run inference with constraints
uv run python -m training.inference_json \
    --model-path models/json_model/best_model \
    --input invoice.pdf

# 4. Test the demo
uv run python examples/test_constrained_json.py
```

**Key Difference**: Method 2 guarantees 100% valid JSON using constrained decoding!

---

## Training

### Method 1: Train Token Classification Model

Train a DistilBERT model on your data:

```bash
uv run train-invoice \
  --data-dir data/processed \
  --output-dir models \
  --num-epochs 10 \
  --batch-size 8
```

### Method 2: Train JSON Generation Model

Train a T5/Flan-T5 model for JSON generation:

```bash
uv run python -m training.train_json \
    --data-dir data/processed \
    --convert-data \
    --output-dir models/json_model \
    --model-name google/flan-t5-base \
    --batch-size 8 \
    --num-epochs 10
```

**Key parameters**:
- `--model-name`: Choose from `t5-small`, `t5-base`, `google/flan-t5-base` (recommended), `google/flan-t5-large`
- `--convert-data`: Auto-convert BIO data to JSON format
- `--batch-size`: Smaller than Method 1 due to larger model

See [JSON_GENERATION_README.md](JSON_GENERATION_README.md) for details.

---

## Inference

### Method 1: Extract with Token Classification

Extract fields from a new invoice:

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

### Method 2: Extract with Constrained JSON

Extract with guaranteed valid JSON:

```bash
# With constraints (guaranteed valid JSON!)
uv run python -m training.inference_json \
    --model-path models/json_model/best_model \
    --input invoice.pdf \
    --output results.json

# Without constraints (for comparison)
uv run python -m training.inference_json \
    --model-path models/json_model/best_model \
    --input invoice.pdf \
    --no-constraints
```

**Output format** (both methods):
```json
{
  "invoice_number": "INV-2024-0001",
  "date": "January 15, 2024",
  "customer": {
    "name": "John Smith",
    "address": "123 Main Street Springfield IL 62701"
  },
  "items": [
    {"name": "Laptop Computer", "price": "$1,299.99"},
    {"name": "Wireless Mouse", "price": "$29.99"},
    {"name": "USB-C Cable", "price": "$15.99"}
  ],
  "total": "$1,345.97"
}
```

**Method 2 difference**: JSON is **always valid** (100% guarantee with constraints enabled).

---

## Evaluation

Evaluate model performance on test data:

```bash
uv run evaluate-invoice \
  --model-path models/best_model \
  --test-data data/processed/test.json \
  --output evaluation_results.json
```

This will show:
- Overall precision, recall, and F1-score
- Per-entity metrics for each field type
- Exact match accuracy
- Detailed classification report

## Working with the Company Documents Dataset

### Dataset Information

This project uses the [Company Documents Dataset](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset) from Kaggle, which contains:
- Real invoice PDFs from various companies
- Located in `CompanyDocuments/invoices/` folder
- Multiple invoice formats and layouts

### Download Commands

```bash
# Download and extract invoices
uv run download-dataset --extract-invoices

# Download with analysis
uv run download-dataset --extract-invoices --analyze

# Force redownload
uv run download-dataset --force --extract-invoices
```

### Preprocessing Options

```bash
# Process all invoices
uv run preprocess-dataset --input-dir data/raw/invoices

# Process limited number for testing
uv run preprocess-dataset --input-dir data/raw/invoices --max-files 20

# Create annotation file for manual review
uv run preprocess-dataset \
  --input-dir data/raw/invoices \
  --create-annotation-file \
  --annotation-output data/annotations.json \
  --max-files 10
```

### Auto-annotation

The preprocessing script uses regex patterns to automatically detect:
- **Invoice Numbers**: Patterns like "Invoice #123", "INV-001"
- **Dates**: Various formats (MM/DD/YYYY, Month DD, YYYY, etc.)
- **Totals**: Currency amounts near keywords like "Total", "Amount Due"
- **Customer Names**: Names following "Bill To", "Customer", etc.

You can review and improve annotations in the generated JSON files before training.

## Label Schema

The system uses BIO (Begin-Inside-Outside) tagging:

| Field Type | Labels | Description |
|------------|--------|-------------|
| Total Price | B-TOTAL, I-TOTAL | Invoice total amount |
| Item Name | B-ITEM, I-ITEM | Product/service name |
| Item Price | B-ITEM_PRICE, I-ITEM_PRICE | Individual item cost |
| Customer Name | B-CUSTOMER_NAME, I-CUSTOMER_NAME | Customer's full name |
| Customer Address | B-CUSTOMER_ADDRESS, I-CUSTOMER_ADDRESS | Customer's address |
| Date | B-DATE, I-DATE | Invoice date |
| Invoice Number | B-INVOICE_NUMBER, I-INVOICE_NUMBER | Unique invoice ID |
| Outside | O | Not part of any entity |

## Model Architectures

### Method 1: Token Classification (DistilBERT)

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Token Classification (NER)
- **Labels**: 15 labels (7 entity types × 2 + O)
- **Max Sequence Length**: 512 tokens
- **Training**: AdamW optimizer with linear warmup
- **Speed**: ~50ms per invoice
- **Memory**: ~1GB GPU

### Method 2: JSON Generation (T5/Flan-T5)

- **Base Model**: T5-base or Flan-T5-base (220-250M params)
- **Task**: Sequence-to-Sequence Generation
- **Output**: JSON string (validated by schema)
- **Max Input Length**: 512 tokens
- **Max Target Length**: 512 tokens
- **Training**: AdamW optimizer with warmup
- **Constrained Decoding**: `outlines` library for grammar enforcement
- **Speed**: ~70ms per invoice (with constraints)
- **Memory**: ~4GB GPU
- **Valid JSON Rate**: **100%** (guaranteed)

## Evaluation Metrics

The system provides comprehensive evaluation:

1. **Token-level Metrics**:
   - Precision: Percentage of predicted entities that are correct
   - Recall: Percentage of true entities that were found
   - F1-score: Harmonic mean of precision and recall

2. **Per-entity Metrics**: Separate metrics for each field type

3. **Exact Match**: Percentage of invoices where all fields match exactly

4. **Classification Report**: Detailed per-class performance

## Advanced Usage

### Resume Training from Checkpoint

```bash
uv run train-invoice \
  --resume checkpoints/checkpoint_epoch_5.pt \
  --num-epochs 15
```

### Batch Inference

```python
from pathlib import Path
from training.inference import InvoiceExtractor

extractor = InvoiceExtractor(Path("models/best_model"))

texts = [
    "Invoice INV-001...",
    "Invoice INV-002...",
]

results = extractor.extract_batch(texts, batch_size=8)
```

### Custom Configuration

Modify [training/config.py](training/config.py) to customize:
- Model hyperparameters
- Label definitions
- Training settings
- File paths

### GPU Acceleration

The system automatically detects and uses the best available hardware acceleration:

**Priority**: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU

#### Check Available Acceleration

```bash
# Run verification script
uv run python verify_setup.py

# Or check manually
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

#### Performance by Device

| Device | Training Speed | Relative Performance |
|--------|---------------|---------------------|
| **Apple Silicon (MPS)** | 2-5 min/epoch | **3-10x faster than CPU** |
| NVIDIA RTX 3090 (CUDA) | 1-2 min/epoch | 10-15x faster than CPU |
| CPU (M1/M2/Intel) | 15-20 min/epoch | Baseline |

#### Mac Users (Apple Silicon)

**Automatic MPS acceleration** on M1/M2/M3/M4 Macs:

```bash
# Just run training - MPS is automatic!
uv run train-invoice --data-dir data/processed
```

**Output will show**:
```
Using device: mps (Apple Metal Performance Shaders)
```

**For detailed Mac optimization**, see [MAC_SETUP.md](MAC_SETUP.md):
- Optimal batch sizes for your chip
- Performance tuning tips
- Troubleshooting guide
- Memory management

#### NVIDIA GPU Users

CUDA is automatically detected:

```bash
# Train with CUDA
uv run train-invoice --data-dir data/processed
```

**Output will show**:
```
Using device: cuda (NVIDIA GeForce RTX 3090)
```

## Development

### Project Commands

Defined in [pyproject.toml](pyproject.toml):
- `download-dataset`: Download Company Documents Dataset from Kaggle
- `preprocess-dataset`: Preprocess invoice PDFs into training format
- `train-invoice`: Train the model
- `extract-invoice`: Extract fields from invoices
- `evaluate-invoice`: Evaluate model performance

### Adding New Entity Types

1. Update labels in [training/config.py](training/config.py#L18):
```python
labels = [
    "O",
    "B-NEW_ENTITY", "I-NEW_ENTITY",
    # ... existing labels
]
```

2. Update your annotation data with the new labels

3. Retrain the model

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
uv run train-invoice --batch-size 4
```

### Low Accuracy

- Increase training epochs: `--num-epochs 20`
- Ensure data quality and annotation consistency
- Add more training examples
- Try data augmentation

### PDF Extraction Issues

- Ensure PDFs contain extractable text (not scanned images)
- For scanned PDFs, consider adding OCR preprocessing

## Performance Tips

1. **Use GPU**: Training is 10-50x faster on GPU
2. **Optimize Batch Size**: Larger batches (if memory allows) train faster
3. **Mixed Precision**: Use `--fp16` flag for faster training (requires GPU)
4. **Data Quality**: Clean, consistent annotations improve results significantly

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use DistilBERT in your work, cite:
```
@article{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}
```
