"""Create sample training data for testing."""

import json
from pathlib import Path


def create_sample_data():
    """Create sample annotated invoice data."""

    # Sample invoice text
    invoice_text = """INVOICE INV-2024-0001 Date: January 15, 2024 Bill To: John Smith 123 Main Street Springfield IL 62701 Items: Laptop Computer $1,299.99 Wireless Mouse $29.99 USB-C Cable $15.99 TOTAL: $1,345.97"""

    # Tokenize
    tokens = invoice_text.split()

    # Create labels (BIO format)
    labels = ["O"] * len(tokens)

    # Manually label entities (you would do this programmatically with your dataset)
    # This is a simplified example
    token_to_label = {
        "INV-2024-0001": "B-INVOICE_NUMBER",
        "January": "B-DATE",
        "15,": "I-DATE",
        "2024": "I-DATE",
        "John": "B-CUSTOMER_NAME",
        "Smith": "I-CUSTOMER_NAME",
        "123": "B-CUSTOMER_ADDRESS",
        "Main": "I-CUSTOMER_ADDRESS",
        "Street": "I-CUSTOMER_ADDRESS",
        "Springfield": "I-CUSTOMER_ADDRESS",
        "IL": "I-CUSTOMER_ADDRESS",
        "62701": "I-CUSTOMER_ADDRESS",
        "Laptop": "B-ITEM",
        "Computer": "I-ITEM",
        "$1,299.99": "B-ITEM_PRICE",
        "Wireless": "B-ITEM",
        "Mouse": "I-ITEM",
        "$29.99": "B-ITEM_PRICE",
        "USB-C": "B-ITEM",
        "Cable": "I-ITEM",
        "$15.99": "B-ITEM_PRICE",
        "$1,345.97": "B-TOTAL",
    }

    # Apply labels
    for i, token in enumerate(tokens):
        if token in token_to_label:
            labels[i] = token_to_label[token]

    # Create examples
    examples = [
        {
            "id": "sample_1",
            "text": invoice_text,
            "tokens": tokens,
            "labels": labels,
            "metadata": {"source": "sample"},
        }
    ]

    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save train, val, and test sets (all same for this example)
    for split in ["train", "val", "test"]:
        output_path = output_dir / f"{split}.json"
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)

        print(f"Created {output_path}")


if __name__ == "__main__":
    create_sample_data()
    print("\nSample data created! You can now run training with:")
    print("  uv run train-invoice")
