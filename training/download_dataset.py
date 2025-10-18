"""Download and extract the Company Documents Dataset from Kaggle."""

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional


def setup_kaggle_credentials():
    """
    Check for Kaggle API credentials and provide setup instructions.

    Returns:
        bool: True if credentials are set up, False otherwise
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        # Ensure proper permissions
        os.chmod(kaggle_json, 0o600)
        return True

    print("=" * 70)
    print("Kaggle API credentials not found!")
    print("=" * 70)
    print("\nTo download datasets from Kaggle, you need to set up API credentials:")
    print("\n1. Go to https://www.kaggle.com/settings/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New Token'")
    print("4. This will download kaggle.json")
    print("5. Move kaggle.json to:", kaggle_dir)
    print("\nCommands to set up:")
    print(f"  mkdir -p {kaggle_dir}")
    print(f"  mv ~/Downloads/kaggle.json {kaggle_json}")
    print(f"  chmod 600 {kaggle_json}")
    print("=" * 70)

    return False


def download_dataset(
    dataset_name: str = "ayoubcherguelaine/company-documents-dataset",
    output_dir: Path = Path("data/raw"),
    force: bool = False,
) -> bool:
    """
    Download dataset from Kaggle.

    Args:
        dataset_name: Kaggle dataset identifier
        output_dir: Directory to save dataset
        force: Force redownload if already exists

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import kaggle
    except ImportError:
        print("Error: kaggle package not installed. Run: uv sync")
        return False

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    dataset_dir = output_dir / "CompanyDocuments"
    if dataset_dir.exists() and not force:
        print(f"Dataset already exists at: {dataset_dir}")
        print("Use --force to redownload")
        return True

    print(f"Downloading dataset: {dataset_name}")
    print(f"Destination: {output_dir}")

    try:
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True,
            quiet=False,
        )

        print("\nDataset downloaded successfully!")

        # Show what was downloaded
        if dataset_dir.exists():
            print(f"\nDataset structure:")
            for item in sorted(dataset_dir.iterdir()):
                if item.is_dir():
                    num_files = len(list(item.glob("*")))
                    print(f"  {item.name}/ ({num_files} files)")
                else:
                    print(f"  {item.name}")

        return True

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def extract_invoices(
    raw_dir: Path = Path("data/raw/CompanyDocuments"),
    output_dir: Path = Path("data/raw/invoices"),
) -> int:
    """
    Extract invoice PDFs to a separate directory.

    Args:
        raw_dir: Directory containing CompanyDocuments
        output_dir: Directory to copy invoices to

    Returns:
        int: Number of invoices extracted
    """
    invoices_dir = raw_dir / "invoices"

    if not invoices_dir.exists():
        print(f"Error: Invoices directory not found at {invoices_dir}")
        return 0

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy invoice files
    invoice_files = list(invoices_dir.glob("*"))

    print(f"\nExtracting {len(invoice_files)} invoice files...")

    for invoice_file in invoice_files:
        if invoice_file.is_file():
            dest_file = output_dir / invoice_file.name
            shutil.copy2(invoice_file, dest_file)

    print(f"Invoices extracted to: {output_dir}")

    return len(invoice_files)


def analyze_dataset(data_dir: Path = Path("data/raw/invoices")) -> dict:
    """
    Analyze the downloaded invoice dataset.

    Args:
        data_dir: Directory containing invoices

    Returns:
        dict: Dataset statistics
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return {}

    # Count files by type
    pdf_files = list(data_dir.glob("*.pdf"))
    txt_files = list(data_dir.glob("*.txt"))
    other_files = [f for f in data_dir.iterdir()
                   if f.is_file() and f.suffix not in [".pdf", ".txt"]]

    stats = {
        "total_files": len(list(data_dir.iterdir())),
        "pdf_files": len(pdf_files),
        "txt_files": len(txt_files),
        "other_files": len(other_files),
    }

    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    print(f"Total files: {stats['total_files']}")
    print(f"PDF files: {stats['pdf_files']}")
    print(f"Text files: {stats['txt_files']}")
    print(f"Other files: {stats['other_files']}")

    # Show sample files
    if pdf_files:
        print(f"\nSample PDF files:")
        for pdf_file in pdf_files[:5]:
            size_mb = pdf_file.stat().st_size / (1024 * 1024)
            print(f"  {pdf_file.name} ({size_mb:.2f} MB)")
        if len(pdf_files) > 5:
            print(f"  ... and {len(pdf_files) - 5} more")

    return stats


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(
        description="Download Company Documents Dataset from Kaggle"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ayoubcherguelaine/company-documents-dataset",
        help="Kaggle dataset identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload if already exists",
    )
    parser.add_argument(
        "--extract-invoices",
        action="store_true",
        help="Extract invoices to separate directory",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze dataset after download",
    )

    args = parser.parse_args()

    # Check Kaggle credentials
    if not setup_kaggle_credentials():
        return

    # Download dataset
    success = download_dataset(
        dataset_name=args.dataset,
        output_dir=Path(args.output_dir),
        force=args.force,
    )

    if not success:
        return

    # Extract invoices if requested
    if args.extract_invoices:
        num_invoices = extract_invoices(
            raw_dir=Path(args.output_dir) / "CompanyDocuments",
            output_dir=Path(args.output_dir) / "invoices",
        )
        print(f"\nExtracted {num_invoices} invoice files")

    # Analyze dataset if requested
    if args.analyze:
        analyze_dataset(Path(args.output_dir) / "invoices")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("=" * 70)
    print("1. Annotate your invoice data with field labels")
    print("2. Run preprocessing: uv run preprocess-dataset")
    print("3. Train the model: uv run train-invoice")
    print("=" * 70)


if __name__ == "__main__":
    main()
