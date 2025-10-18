# Invoice Field Extraction - Inference Examples

## Quick Start

Run inference on a PDF file:

```bash
cd examples
python infer.py path/to/invoice.pdf
```

Or specify a custom model path:

```bash
python infer.py path/to/invoice.pdf ../models/best_model
```

## Output

The script outputs extracted invoice fields as JSON:

```json
{
  "INVOICE_NUMBER": ["INV-2024-0001"],
  "DATE": ["January 15, 2024"],
  "CUSTOMER_NAME": ["John Smith"],
  "CUSTOMER_ADDRESS": ["123 Main Street Springfield IL"],
  "ITEM": ["Laptop Computer", "Wireless Mouse"],
  "ITEM_PRICE": ["$1,299.99", "$29.99"],
  "TOTAL": ["$1,345.97"]
}
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PyMuPDF

Install from parent directory:
```bash
cd ..
uv sync
```

## Script Details

`infer.py` is a self-contained script that:

1. Loads the fine-tuned DistilBERT model from disk
2. Extracts text from PDF using PyMuPDF
3. Runs token classification inference
4. Extracts entities using BIO tagging
5. Returns results as JSON

Total code: ~80 lines including comments
