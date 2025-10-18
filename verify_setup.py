#!/usr/bin/env python3
"""Verify installation and setup of invoice extraction system."""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 14:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires >= 3.14)")
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\nChecking dependencies...")

    required_packages = [
        "torch",
        "transformers",
        "pymupdf",
        "datasets",
        "sklearn",
        "numpy",
        "pandas",
        "PIL",
        "tqdm",
        "seqeval",
        "kaggle",
        "openpyxl",
    ]

    all_installed = True

    for package in required_packages:
        try:
            if package == "PIL":
                __import__("PIL")
            elif package == "sklearn":
                __import__("sklearn")
            elif package == "pymupdf":
                __import__("fitz")
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - not installed")
            all_installed = False

    return all_installed


def check_kaggle_credentials():
    """Check Kaggle API credentials."""
    print("\nChecking Kaggle API credentials...")

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if kaggle_json.exists():
        import os
        import stat

        # Check permissions
        perms = oct(os.stat(kaggle_json).st_mode)[-3:]
        if perms == "600":
            print("  ✓ Kaggle credentials found with correct permissions")
            return True
        else:
            print(f"  ⚠ Kaggle credentials found but permissions are {perms} (should be 600)")
            print(f"    Run: chmod 600 {kaggle_json}")
            return False
    else:
        print("  ✗ Kaggle credentials not found")
        print("    Set up at: https://www.kaggle.com/settings/account")
        print(f"    Place kaggle.json at: {kaggle_json}")
        return False


def check_gpu_acceleration():
    """Check GPU acceleration (CUDA or MPS)."""
    print("\nChecking GPU acceleration...")

    try:
        import torch

        has_acceleration = False

        # Check CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available - {device_count} device(s)")
            print(f"    Device: {device_name}")
            has_acceleration = True

        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print(f"  ✓ MPS (Metal Performance Shaders) available")
            print(f"    Apple Silicon GPU acceleration enabled")
            print(f"    Expected speedup: 3-10x faster than CPU")
            has_acceleration = True

        if not has_acceleration:
            print("  ⚠ No GPU acceleration available - will use CPU")
            print("    Training will be significantly slower on CPU")
            print("    Consider using a Mac with Apple Silicon or system with NVIDIA GPU")

        return has_acceleration
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False


def check_directory_structure():
    """Check directory structure."""
    print("\nChecking directory structure...")

    required_dirs = [
        "training",
        "examples",
    ]

    all_exist = True

    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - not found")
            all_exist = False

    return all_exist


def check_data_directories():
    """Check data directories."""
    print("\nChecking data directories...")

    data_dirs = [
        ("data", False),
        ("data/raw", False),
        ("data/raw/invoices", False),
        ("data/processed", False),
    ]

    for dir_path, required in data_dirs:
        path = Path(dir_path)
        if path.exists():
            num_files = len(list(path.iterdir())) if path.is_dir() else 0
            print(f"  ✓ {dir_path}/ ({num_files} items)")
        else:
            if required:
                print(f"  ✗ {dir_path}/ - not found (required)")
            else:
                print(f"  ○ {dir_path}/ - not found (will be created)")

    return True


def check_training_files():
    """Check training module files."""
    print("\nChecking training module files...")

    required_files = [
        "training/__init__.py",
        "training/config.py",
        "training/dataset.py",
        "training/model.py",
        "training/metrics.py",
        "training/pdf_processor.py",
        "training/train.py",
        "training/inference.py",
        "training/evaluate.py",
        "training/download_dataset.py",
        "training/preprocess_data.py",
    ]

    all_exist = True

    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {file_path} - not found")
            all_exist = False

    return all_exist


def print_summary(results):
    """Print summary of checks."""
    print("\n" + "=" * 70)
    print("SETUP VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = all(results.values())

    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {check_name}")

    print("=" * 70)

    if all_passed:
        print("\n✓ All checks passed! You're ready to start.")
        print("\nNext steps:")
        print("  1. Download dataset: uv run download-dataset --extract-invoices")
        print("  2. Preprocess data: uv run preprocess-dataset")
        print("  3. Train model: uv run train-invoice")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nTo install dependencies:")
        print("  uv sync")

    return all_passed


def main():
    """Main verification function."""
    print("=" * 70)
    print("INVOICE EXTRACTION SYSTEM - SETUP VERIFICATION")
    print("=" * 70)

    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Kaggle Credentials": check_kaggle_credentials(),
        "GPU Acceleration": check_gpu_acceleration(),
        "Directory Structure": check_directory_structure(),
        "Data Directories": check_data_directories(),
        "Training Files": check_training_files(),
    }

    success = print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
