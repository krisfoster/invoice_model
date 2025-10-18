# Company Documents Dataset Integration

This document describes how the [Company Documents Dataset](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset) is integrated into the invoice extraction system.

## Dataset Overview

**Dataset Name**: Company Documents Dataset
**Kaggle URL**: https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset
**Owner**: Ayoub Cherguelaine
**Content**: Real company documents including invoices, receipts, and other business documents

### Invoice Subset

The invoices are located in the `CompanyDocuments/invoices/` folder and include:
- Multiple invoice formats from different companies
- Real-world invoice layouts and structures
- Various document qualities and styles

## Automatic Download

The system includes automatic download functionality:

```bash
uv run download-dataset --extract-invoices --analyze
```

### What Happens During Download

1. **Authentication**: Uses Kaggle API credentials from `~/.kaggle/kaggle.json`
2. **Download**: Fetches entire Company Documents Dataset
3. **Extraction**: Unzips to `data/raw/CompanyDocuments/`
4. **Invoice Isolation**: Copies invoices to `data/raw/invoices/`
5. **Analysis**: Shows statistics about downloaded files

### Download Options

```bash
# Basic download
uv run download-dataset

# Extract invoices to separate folder
uv run download-dataset --extract-invoices

# Show dataset statistics
uv run download-dataset --analyze

# Force redownload
uv run download-dataset --force

# All options combined
uv run download-dataset --extract-invoices --analyze --force
```

## Preprocessing Pipeline

### Auto-annotation Strategy

The preprocessing script uses regex-based pattern matching to automatically annotate invoice fields:

#### 1. Invoice Numbers
Patterns:
- `"Invoice #123"`, `"INV-001"`, `"Invoice: ABC-123"`
- Captures alphanumeric identifiers after keywords
- Location: Usually at top of invoice

#### 2. Dates
Patterns:
- `"January 15, 2024"` - Full month name
- `"01/15/2024"` - Slash format
- `"2024-01-15"` - ISO format
- Location: Near invoice number or at top

#### 3. Total Amounts
Patterns:
- `"Total: $1,234.56"` - With keyword
- `"Amount Due: 1234.56"` - Various keywords
- Captures amounts with currency symbols
- Location: Usually at bottom of invoice

#### 4. Customer Names
Patterns:
- Names following "Bill To:", "Customer:", "Client:"
- Capitalized multi-word names
- Location: Usually in billing section

### Preprocessing Workflow

```bash
uv run preprocess-dataset --input-dir data/raw/invoices
```

Steps:
1. **PDF Text Extraction**: Uses PyMuPDF to extract text from PDFs
2. **Auto-annotation**: Applies regex patterns to find entities
3. **Tokenization**: Splits text into tokens (words)
4. **BIO Tagging**: Converts annotations to Begin-Inside-Outside format
5. **Data Splitting**: Creates train/val/test splits (80/10/10)

### Output Format

Each example in the processed JSON files contains:

```json
{
  "id": "invoice_001",
  "text": "INVOICE INV-2024-001 Date: January 15, 2024 ...",
  "tokens": ["INVOICE", "INV-2024-001", "Date:", ...],
  "labels": ["O", "B-INVOICE_NUMBER", "O", ...],
  "metadata": {
    "source": "company_documents",
    "filename": "invoice_001.pdf",
    "annotations": {
      "invoice_number": [{"text": "INV-2024-001", "start": 8, "end": 20}],
      "date": [{"text": "January 15, 2024", "start": 27, "end": 44}]
    }
  }
}
```

## Improving Auto-annotation

The auto-annotation is designed to be a starting point. You can improve it:

### 1. Review Auto-annotations

Create an annotation file for manual review:

```bash
uv run preprocess-dataset \
  --input-dir data/raw/invoices \
  --create-annotation-file \
  --annotation-output data/annotations.json \
  --max-files 10
```

This creates a JSON file with auto-annotations that you can review and correct.

### 2. Customize Regex Patterns

Edit [training/preprocess_data.py](training/preprocess_data.py):

```python
class InvoiceAnnotator:
    def __init__(self):
        self.patterns = {
            "invoice_number": [
                r"(?:invoice|inv)[\s#:]*([A-Z0-9-]+)",
                # Add your custom patterns here
                r"your_custom_pattern_here",
            ],
            # ... other patterns
        }
```

### 3. Add New Entity Types

1. Add pattern to `InvoiceAnnotator.patterns`
2. Update `field_mapping` in `text_to_bio()` function
3. Add labels to [training/config.py](training/config.py)

## Dataset Statistics

After downloading, you can analyze the dataset:

```bash
uv run download-dataset --extract-invoices --analyze
```

This shows:
- Total number of files
- Number of PDF files
- Number of text files
- File sizes
- Sample filenames

## Using Custom Datasets

While this system is configured for the Company Documents Dataset, you can use any invoice dataset:

### Option 1: Replace Invoice PDFs

Simply place your invoice PDFs in `data/raw/invoices/` and run preprocessing:

```bash
# Copy your invoices
cp your_invoices/*.pdf data/raw/invoices/

# Preprocess
uv run preprocess-dataset --input-dir data/raw/invoices
```

### Option 2: Download Different Kaggle Dataset

Modify [training/config.py](training/config.py):

```python
@dataclass
class DataConfig:
    kaggle_dataset: str = "your-username/your-dataset-name"
    # ... rest of config
```

Then run:

```bash
uv run download-dataset --extract-invoices
```

### Option 3: Manual Dataset Preparation

If you have pre-annotated data in a different format:

1. Convert to the expected JSON format (see Output Format above)
2. Place train.json, val.json, test.json in `data/processed/`
3. Run training directly:

```bash
uv run train-invoice --data-dir data/processed
```

## Data Quality Tips

### For Best Results

1. **Invoice Variety**: Include diverse invoice formats
2. **Clear Text**: Ensure PDFs have text layer (not scanned images)
3. **Annotation Quality**: Review and correct auto-annotations
4. **Balanced Data**: Try to have similar numbers of each entity type
5. **Sufficient Data**: Aim for 100+ invoices for good performance

### Handling Scanned Invoices

If invoices are scanned images without text layer:

1. Use OCR preprocessing (Tesseract, Cloud Vision API, etc.)
2. Convert scanned PDFs to text before processing
3. Consider using a vision model instead of text-based extraction

## Dataset License and Usage

- **Source**: Kaggle - Company Documents Dataset
- **License**: Check Kaggle dataset page for specific license
- **Usage**: Follow Kaggle's terms of service
- **Citation**: Credit the dataset owner when publishing results

## Troubleshooting

### "Dataset not found" Error

- Verify dataset name: `ayoubcherguelaine/company-documents-dataset`
- Check Kaggle credentials are set up correctly
- Ensure you have internet connection
- Try downloading directly from Kaggle website

### "No invoices found" Error

- Verify dataset structure has `CompanyDocuments/invoices/` folder
- Check files were extracted correctly
- Use `--analyze` flag to see what was downloaded

### "No text extracted" Warnings

- Some PDFs may be scanned images
- Check PDF can be opened normally
- Consider OCR preprocessing for scanned documents

## Performance Expectations

With the Company Documents Dataset:

- **Training Time**: 10-30 minutes per epoch (depending on GPU)
- **Expected F1 Score**: 0.70-0.90 (varies by entity type)
- **Best Performing Fields**: Invoice number, date, total
- **Challenging Fields**: Item names, addresses (more variability)

## Next Steps

1. Download dataset: `uv run download-dataset --extract-invoices --analyze`
2. Review sample invoices in `data/raw/invoices/`
3. Run preprocessing with small sample first: `--max-files 10`
4. Check auto-annotations quality
5. Adjust regex patterns if needed
6. Process full dataset
7. Train model

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).
