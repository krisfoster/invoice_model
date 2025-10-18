# JSON Generation with Constrained Decoding

This document describes **Method 2: Constrained Decoding with Grammar** for training models to produce JSON output directly.

## Overview

This implementation uses **sequence-to-sequence models** (T5/Flan-T5) with **constrained generation** to extract invoice fields and output them as **guaranteed valid JSON**.

### Key Features

- ✅ **Guaranteed Valid JSON**: Uses grammar-based constraints during generation
- ✅ **Schema Compliance**: Output always matches the defined Pydantic schema
- ✅ **No Post-Processing**: Direct JSON output, no parsing needed
- ✅ **Flexible Architecture**: Easy to modify schema for different use cases

## Architecture

### Model: T5/Flan-T5 (Seq2Seq)

```
Input:  "extract invoice fields: INVOICE #12345 Date: 2024-01-15..."
                         ↓
              T5ForConditionalGeneration
                         ↓
              Constrained Decoding (outlines)
                         ↓
Output: {"invoice_number": "12345", "date": "2024-01-15", ...}
```

### How Constrained Decoding Works

During generation, at each token:
1. Model predicts probability distribution over vocabulary
2. **Filter out tokens that violate JSON grammar**
3. Select next token only from valid tokens
4. Result: **Always valid JSON matching the schema**

## Installation

### Dependencies

Already added to `pyproject.toml`:

```toml
dependencies = [
    "outlines>=0.1.0",    # Constrained generation
    "pydantic>=2.0.0",    # Schema validation
    ...
]
```

Install all dependencies:

```bash
pip install -e .
# or with uv:
uv pip install -e .
```

## Quick Start

### 1. Convert Existing Data

If you have BIO-labeled data, convert it to JSON format:

```bash
python -m training.convert_data_to_json \
    --input data/processed \
    --output data/json \
    --splits train val test
```

This creates:
- `data/json/train_json.json`
- `data/json/val_json.json`
- `data/json/test_json.json`

### 2. Train the Model

```bash
python -m training.train_json \
    --data-dir data/processed \
    --json-data-dir data/json \
    --output-dir models/json_model \
    --model-name google/flan-t5-base \
    --batch-size 8 \
    --num-epochs 10 \
    --convert-data  # Optional: convert on-the-fly
```

**Model Options:**
- `t5-small` (60M params) - Fast, lower quality
- `t5-base` (220M params) - Good balance
- `google/flan-t5-base` (250M params) - **Recommended**, instruction-tuned
- `google/flan-t5-large` (780M params) - Best quality, slower

### 3. Run Inference

#### With Constraints (Guaranteed Valid JSON)

```bash
python -m training.inference_json \
    --model-path models/json_model/best_model \
    --input path/to/invoice.pdf \
    --output results.json
```

#### Without Constraints (For Comparison)

```bash
python -m training.inference_json \
    --model-path models/json_model/best_model \
    --input path/to/invoice.pdf \
    --output results.json \
    --no-constraints
```

### 4. Test with Demo Script

```bash
python examples/test_constrained_json.py
```

This runs three tests:
1. **Well-formed invoice** with constraints
2. **Well-formed invoice** without constraints
3. **Malformed input** to test robustness

## JSON Schema

The output schema is defined in [`training/json_schema.py`](training/json_schema.py):

```python
{
  "invoice_number": str | None,
  "date": str | None,
  "customer": {
    "name": str | None,
    "address": str | None
  },
  "items": [
    {
      "name": str,
      "price": str | None
    }
  ],
  "total": str | None
}
```

### Modifying the Schema

To add/remove fields:

1. Edit [`training/json_schema.py`](training/json_schema.py)
2. Update the Pydantic models
3. Regenerate training data
4. Retrain the model

Example - Adding a "tax" field:

```python
class InvoiceOutput(BaseModel):
    invoice_number: Optional[str]
    date: Optional[str]
    customer: CustomerInfo
    items: List[InvoiceItem]
    total: Optional[str]
    tax: Optional[str]  # New field
```

## File Structure

```
training/
├── json_schema.py              # Pydantic schema definition
├── model_json.py               # T5-based seq2seq model
├── dataset_json.py             # Dataset for JSON generation
├── bio_to_json_converter.py    # Convert BIO labels to JSON
├── train_json.py               # Training script
├── inference_json.py           # Inference with constrained generation
├── metrics_json.py             # JSON-specific metrics
├── convert_data_to_json.py     # Standalone conversion script
└── config.py                   # Added JSONModelConfig

examples/
└── test_constrained_json.py    # Demo script
```

## How It Works: Constrained Decoding

### Normal Generation

```python
# At each step:
logits = model(...)           # [vocab_size]
next_token = argmax(logits)   # Can be ANY token
# Result: May produce invalid JSON ❌
```

### Constrained Generation

```python
# At each step:
logits = model(...)                          # [vocab_size]
valid_tokens = grammar.get_valid_tokens()    # Based on JSON state
logits[~valid_tokens] = -inf                 # Mask invalid tokens
next_token = argmax(logits)                  # Only valid tokens
# Result: Always valid JSON ✅
```

### State Machine Example

```
Current State: START
Valid Tokens: ['{']

Current State: AFTER_OPEN_BRACE
Valid Tokens: ['"invoice_number"', '"date"', '"customer"', ...]

Current State: AFTER_KEY_"invoice_number"
Valid Tokens: [':']

Current State: AFTER_COLON
Valid Tokens: ['"']  # Start string value

Current State: IN_STRING_VALUE
Valid Tokens: [alphanumeric + special chars + '"']

Current State: AFTER_STRING_VALUE
Valid Tokens: [',', '}']  # Next field or close object
```

## Training Details

### Dataset Format

Input-output pairs with instruction prompts:

```json
{
  "id": "train_001",
  "input": "extract invoice fields: INVOICE #12345 Date: 2024-01-15 Customer: John Doe...",
  "target": "{\"invoice_number\": \"12345\", \"date\": \"2024-01-15\", \"customer\": {\"name\": \"John Doe\", \"address\": null}, \"items\": [], \"total\": null}",
  "metadata": {...}
}
```

### Metrics

- **Valid JSON Rate**: % of outputs that are parseable JSON
- **Exact Match**: % of outputs that exactly match reference
- **Field Accuracy**: % of fields correctly extracted
- **Field F1**: Harmonic mean of field-level precision/recall

With constrained generation:
- Valid JSON Rate: **100%** (guaranteed)
- Field F1: Depends on model quality

### Hyperparameters

Recommended settings (from `JSONModelConfig`):

```python
model_name = "google/flan-t5-base"
batch_size = 8
learning_rate = 5e-5
num_epochs = 10
max_input_length = 512
max_target_length = 512
num_beams = 4
```

## Performance Considerations

### Inference Speed

| Method | Relative Speed | Valid JSON |
|--------|---------------|------------|
| Unconstrained | 1.0x (baseline) | ~80-95% |
| **Constrained** | **0.7-0.9x** | **100%** |

Constrained generation adds 10-30% overhead but **guarantees valid JSON**.

### Memory Requirements

| Model | Parameters | GPU Memory | Quality |
|-------|-----------|------------|---------|
| t5-small | 60M | ~2GB | Fair |
| t5-base | 220M | ~4GB | Good |
| flan-t5-base | 250M | ~4GB | **Best** |
| flan-t5-large | 780M | ~8GB | Excellent |

### Optimization Tips

1. **Use KV-cache**: Enable `use_cache=True` in generation
2. **Batch processing**: Process multiple invoices together
3. **Quantization**: Use `load_in_8bit=True` for lower memory
4. **Smaller model**: Start with `t5-small` for prototyping

## Comparison: Method 1 vs Method 2

| Aspect | Method 1 (Token Classification) | **Method 2 (Constrained JSON)** |
|--------|--------------------------------|----------------------------------|
| Model | DistilBERT (67M) | T5-base (220M) |
| Output | BIO labels → JSON | **Direct JSON** |
| Valid JSON | Requires post-processing | **100% guaranteed** |
| Flexibility | Fixed schema | **Easy to modify** |
| Training Data | BIO labels required | Can use BIO or direct JSON |
| Inference Speed | Fast (~50ms) | Moderate (~70ms) |
| Memory | Low (~1GB) | Medium (~4GB) |
| **Best For** | Fast, fixed schema | **Schema flexibility, guaranteed validity** |

## Troubleshooting

### "outlines not installed"

```bash
pip install outlines
# or
uv pip install outlines
```

### "Model generates invalid JSON" (with constraints disabled)

This is expected when `use_constraints=False`. Enable constraints:

```python
extractor = ConstrainedJSONExtractor(
    model_path=model_path,
    use_constraints=True  # Enable constraints
)
```

### Slow inference

1. Reduce `num_beams` (e.g., from 4 to 1)
2. Use smaller model (`t5-small` instead of `flan-t5-base`)
3. Enable KV-cache in generation

### Out of memory during training

1. Reduce `batch_size` (try 4 or 2)
2. Reduce `max_input_length` and `max_target_length` (try 256)
3. Use `t5-small` instead of `flan-t5-base`
4. Use gradient accumulation

## Advanced Usage

### Custom Prompt Engineering

Modify prompts in [`bio_to_json_converter.py`](training/bio_to_json_converter.py):

```python
# Default:
input_text = f"extract invoice fields: {text}"

# Custom:
input_text = f"Extract all fields from this invoice and format as JSON: {text}"
```

### Multiple Output Formats

Define multiple schemas:

```python
# training/json_schema.py
class DetailedInvoiceOutput(BaseModel):
    # More detailed schema
    ...

class SimpleInvoiceOutput(BaseModel):
    # Simpler schema
    ...
```

Use different schemas for different use cases.

### Batch Inference

```python
extractor = ConstrainedJSONExtractor(model_path, use_constraints=True)

texts = ["invoice 1 text...", "invoice 2 text...", ...]
results = extractor.extract_batch(texts, batch_size=8)
```

## References

- **Outlines**: https://github.com/outlines-dev/outlines
- **Pydantic**: https://docs.pydantic.dev/
- **T5 Paper**: https://arxiv.org/abs/1910.10683
- **Constrained Decoding**: https://arxiv.org/abs/2307.09702

## Next Steps

1. **Train your model**: Follow Quick Start above
2. **Run demo**: `python examples/test_constrained_json.py`
3. **Evaluate**: Compare with/without constraints
4. **Deploy**: Use in production with guaranteed valid JSON

## Support

For issues or questions:
1. Check this README
2. Review example scripts in `examples/`
3. Open an issue on the repository
