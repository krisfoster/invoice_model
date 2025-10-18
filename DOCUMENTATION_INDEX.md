# Documentation Index

Quick guide to find the documentation you need.

## Start Here

### New Users

**Read this first**: [README.md](README.md)
- Overview of both methods
- Quick feature comparison
- Installation instructions

### Choose Your Method

Not sure which method to use? See [METHOD_COMPARISON.md](METHOD_COMPARISON.md)

**Quick decision tree:**
- Need 100% valid JSON? -> Method 2
- Need maximum speed? -> Method 1
- Schema may change? -> Method 2
- Limited hardware? -> Method 1
- Fixed schema forever? -> Method 1

---

## Method 1: Token Classification (DistilBERT)

**Best for**: Speed, fixed schemas, limited hardware

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Complete step-by-step guide | First time setup |
| [README.md](README.md) | Quick reference | Quick commands |
| [MAC_SETUP.md](MAC_SETUP.md) | Mac-specific optimization | Using Apple Silicon |
| [M3_MAX_PERFORMANCE.md](M3_MAX_PERFORMANCE.md) | M3 Max optimizations | Have M3 Max Mac |

### Quick Commands (Method 1)

```bash
# Download data
uv run download-dataset --extract-invoices --analyze

# Preprocess
uv run preprocess-dataset --input-dir data/raw/invoices

# Train
uv run train-invoice --data-dir data/processed --num-epochs 10

# Infer
uv run extract-invoice --model-path models/best_model --input invoice.pdf
```

---

## Method 2: Constrained JSON Generation (T5/Flan-T5)

**Best for**: Guaranteed valid JSON, flexible schemas, complex structures

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [QUICK_START_JSON.md](QUICK_START_JSON.md) | Get started quickly | First time with Method 2 |
| [JSON_GENERATION_README.md](JSON_GENERATION_README.md) | Complete technical docs | Deep dive into Method 2 |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Implementation overview | Understand architecture |
| [METHOD_COMPARISON.md](METHOD_COMPARISON.md) | Compare both methods | Choosing between methods |

### Quick Commands (Method 2)

```bash
# Install dependencies
uv sync

# Convert data and train
uv run python -m training.train_json \
    --data-dir data/processed \
    --convert-data \
    --batch-size 8

# Infer with constraints
uv run python -m training.inference_json \
    --model-path models/json_model/best_model \
    --input invoice.pdf

# Test demo
uv run python examples/test_constrained_json.py
```

---

## Reference Documents

### Configuration & Setup

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Main readme with overview |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Method 1 complete guide |
| [QUICK_START_JSON.md](QUICK_START_JSON.md) | Method 2 quick start |
| [MAC_SETUP.md](MAC_SETUP.md) | Mac optimization guide |

### Technical Deep Dives

| Document | Purpose |
|----------|---------|
| [JSON_GENERATION_README.md](JSON_GENERATION_README.md) | Method 2 full documentation |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Method 2 implementation details |
| [METHOD_COMPARISON.md](METHOD_COMPARISON.md) | Side-by-side comparison |

### Dataset Information

| Document | Purpose |
|----------|---------|
| [DATASET_INFO.md](DATASET_INFO.md) | Dataset details and statistics |

### Performance Optimization

| Document | Purpose |
|----------|---------|
| [M3_MAX_PERFORMANCE.md](M3_MAX_PERFORMANCE.md) | M3 Max specific optimizations |
| [MPS_SUMMARY.md](MPS_SUMMARY.md) | Apple Silicon MPS guide |
| [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) | General optimization tips |

### Quick References

| Document | Purpose |
|----------|---------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Command quick reference |

---

## Common Tasks

### Getting Started

1. **First time setup**: [README.md](README.md) -> [USAGE_GUIDE.md](USAGE_GUIDE.md)
2. **Choose method**: [METHOD_COMPARISON.md](METHOD_COMPARISON.md)
3. **Try Method 2**: [QUICK_START_JSON.md](QUICK_START_JSON.md)

### Training

- **Method 1 training**: [USAGE_GUIDE.md](USAGE_GUIDE.md) - Step 5
- **Method 2 training**: [QUICK_START_JSON.md](QUICK_START_JSON.md) - Step 2
- **Optimize for Mac**: [MAC_SETUP.md](MAC_SETUP.md)

### Inference

- **Method 1 inference**: [USAGE_GUIDE.md](USAGE_GUIDE.md) - Step 6
- **Method 2 inference**: [QUICK_START_JSON.md](QUICK_START_JSON.md) - Step 3
- **Constrained decoding details**: [JSON_GENERATION_README.md](JSON_GENERATION_README.md)

### Troubleshooting

- **Method 1 issues**: [USAGE_GUIDE.md](USAGE_GUIDE.md) - Troubleshooting section
- **Method 2 issues**: [JSON_GENERATION_README.md](JSON_GENERATION_README.md) - Troubleshooting section
- **Mac performance**: [MAC_SETUP.md](MAC_SETUP.md)

### Understanding the Code

- **Method 1 architecture**: [README.md](README.md) - Model Architecture section
- **Method 2 architecture**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **How constrained decoding works**: [JSON_GENERATION_README.md](JSON_GENERATION_README.md) - "How It Works" section

---

## By User Type

### Researchers / ML Engineers

Read in this order:
1. [README.md](README.md) - Overview
2. [METHOD_COMPARISON.md](METHOD_COMPARISON.md) - Technical comparison
3. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Architecture details
4. [JSON_GENERATION_README.md](JSON_GENERATION_README.md) - Constrained decoding deep dive

### Software Engineers / Application Developers

Read in this order:
1. [README.md](README.md) - Overview
2. [QUICK_START_JSON.md](QUICK_START_JSON.md) - Get started fast
3. [METHOD_COMPARISON.md](METHOD_COMPARISON.md) - Choose best method
4. [USAGE_GUIDE.md](USAGE_GUIDE.md) or [JSON_GENERATION_README.md](JSON_GENERATION_README.md) - Full guide for chosen method

### Data Scientists

Read in this order:
1. [README.md](README.md) - Overview
2. [DATASET_INFO.md](DATASET_INFO.md) - Dataset details
3. [USAGE_GUIDE.md](USAGE_GUIDE.md) - Data preprocessing
4. [METHOD_COMPARISON.md](METHOD_COMPARISON.md) - Choose method

### DevOps / Deployment Engineers

Read in this order:
1. [README.md](README.md) - Overview
2. [METHOD_COMPARISON.md](METHOD_COMPARISON.md) - Performance characteristics
3. [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - Optimization tips
4. [JSON_GENERATION_README.md](JSON_GENERATION_README.md) - Production deployment

---

## File Structure Map

```
training/
├── README.md                          ← Start here
├── DOCUMENTATION_INDEX.md             ← This file (navigation guide)
│
├── Method 1 (Token Classification):
│   ├── USAGE_GUIDE.md                 ← Complete Method 1 guide
│   └── QUICK_REFERENCE.md             ← Quick commands
│
├── Method 2 (Constrained JSON):
│   ├── QUICK_START_JSON.md            ← Quick start guide
│   ├── JSON_GENERATION_README.md      ← Complete documentation
│   └── IMPLEMENTATION_SUMMARY.md      ← Implementation details
│
├── Comparison & Decision:
│   └── METHOD_COMPARISON.md           ← Side-by-side comparison
│
├── Dataset:
│   └── DATASET_INFO.md                ← Dataset information
│
├── Hardware Optimization:
│   ├── MAC_SETUP.md                   ← Mac optimization
│   ├── M3_MAX_PERFORMANCE.md          ← M3 Max specific
│   ├── MPS_SUMMARY.md                 ← Apple Silicon guide
│   └── OPTIMIZATION_SUMMARY.md        ← General optimization
│
└── Code:
    └── training/                      ← Python package
```

---

## Quick Links

### Essential Docs
- [README.md](README.md) - Main overview
- [Quick Start Method 1](USAGE_GUIDE.md)
- [Quick Start Method 2](QUICK_START_JSON.md)
- [Compare Methods](METHOD_COMPARISON.md)

### Deep Dives
- [Method 2 Full Docs](JSON_GENERATION_README.md)
- [Implementation Details](IMPLEMENTATION_SUMMARY.md)
- [Dataset Info](DATASET_INFO.md)

### Optimization
- [Mac Setup](MAC_SETUP.md)
- [M3 Max Performance](M3_MAX_PERFORMANCE.md)
- [General Optimization](OPTIMIZATION_SUMMARY.md)

---

## Still Can't Find What You Need?

1. Check the [README.md](README.md) first
2. Use GitHub search in this repository
3. Check inline code comments in `training/` directory
4. Review example scripts in `examples/` directory

---

**Last Updated**: Compatible with current project state (both Method 1 and Method 2 implemented)
