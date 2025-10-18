# Quick Reference - M3 Max 64GB

## TL;DR - Just Run This

```bash
# Download, preprocess, and train with optimized defaults
./quickstart.sh
```

**That's it!** Everything is optimized for your M3 Max.

## One-Line Commands

```bash
# Setup Kaggle credentials (one-time)
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Download dataset
uv run download-dataset --extract-invoices --analyze

# Preprocess invoices
uv run preprocess-dataset --input-dir data/raw/invoices

# Train (default: optimized for M3 Max)
uv run train-invoice --data-dir data/processed

# Train faster (high performance)
uv run train-invoice --data-dir data/processed --batch-size 48

# Train fastest (maximum speed)
uv run train-invoice --data-dir data/processed --batch-size 64

# Extract from invoice
uv run extract-invoice --model-path models/best_model --input invoice.pdf

# Evaluate model
uv run evaluate-invoice --model-path models/best_model --test-data data/processed/test.json
```

## Your Default Settings

| Setting | Value | Why |
|---------|-------|-----|
| Batch size | **32** | Optimal for M3 Max 64GB |
| Learning rate | **3e-5** | Tuned for larger batches |
| Device | **mps** | Apple Silicon GPU |
| Expected speed | **1-3 min/epoch** | 2.5x faster than generic |

## Performance Modes

| Mode | Command | Speed | Memory |
|------|---------|-------|--------|
| **Default** ✅ | `uv run train-invoice` | Fast | 15-20GB |
| High Perf | `--batch-size 48` | Very Fast (+30%) | 20-25GB |
| Maximum | `--batch-size 64` | Fastest (+50%) | 25-30GB |

## Common Tasks

```bash
# Check setup
uv run python verify_setup.py

# Monitor GPU
sudo powermetrics --samplers gpu_power -i 1000

# Quick test (1 epoch)
uv run train-invoice --num-epochs 1

# Prevent sleep during training
caffeinate -i uv run train-invoice --data-dir data/processed

# Process limited files (testing)
uv run preprocess-dataset --input-dir data/raw/invoices --max-files 20
```

## Expected Times (Your M3 Max)

| Task | Time |
|------|------|
| Preprocess 100 invoices | 2-3 min |
| Train 1 epoch (100 invoices) | 1-2 min |
| Train 10 epochs | 15-20 min |
| Evaluate | 30 sec |
| **Total workflow** | **~25 min** |

## Documentation

- **[M3_MAX_PERFORMANCE.md](M3_MAX_PERFORMANCE.md)** - Complete performance guide
- **[MAC_SETUP.md](MAC_SETUP.md)** - Mac-specific setup
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed usage
- **[README.md](README.md)** - Main documentation

## Troubleshooting

```bash
# MPS not detected?
uv run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Out of memory?
uv run train-invoice --batch-size 24  # Reduce from 32

# Too slow?
uv run train-invoice --batch-size 48  # Increase to 48 or 64

# Check macOS version
sw_vers  # Need macOS 12.3+
```

## Batch Size Guide

| Your RAM | Recommended | Aggressive |
|----------|------------|-----------|
| **64GB** ✅ | **32** | **48-64** |
| 32GB | 24 | 32 |
| 16GB | 16 | 20 |
| 8GB | 8 | 12 |

## Hardware Check

```bash
# Check chip
sysctl -n machdep.cpu.brand_string

# Check memory
sysctl hw.memsize | awk '{print $2/1073741824 " GB"}'

# Check MPS
uv run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

## Workflow

```
1. Download    →  uv run download-dataset --extract-invoices
2. Preprocess  →  uv run preprocess-dataset --input-dir data/raw/invoices
3. Train       →  uv run train-invoice --data-dir data/processed
4. Evaluate    →  uv run evaluate-invoice --model-path models/best_model --test-data data/processed/test.json
5. Extract     →  uv run extract-invoice --model-path models/best_model --input new_invoice.pdf
```

## Files & Directories

```
training/
├── data/
│   ├── raw/invoices/          # Downloaded PDFs
│   └── processed/             # Training data (train/val/test.json)
├── models/
│   ├── best_model/           # Best checkpoint (use this!)
│   └── final_model/          # Final epoch
├── checkpoints/              # Training checkpoints
└── configs/
    └── m3_max_64gb.json     # Your config profile
```

## Quick Links

- Kaggle dataset: https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset
- Kaggle API setup: https://www.kaggle.com/settings/account
- PyTorch MPS docs: https://pytorch.org/docs/stable/notes/mps.html

---

**You're all set!** Your M3 Max will train models blazingly fast. Enjoy! 🚀
