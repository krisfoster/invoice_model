"""Convert BIO-labeled data to JSON target format for seq2seq training."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from training.json_schema import InvoiceOutput, CustomerInfo, InvoiceItem


class BIOToJSONConverter:
    """Convert BIO-labeled invoice data to JSON format."""

    def __init__(self):
        """Initialize converter."""
        pass

    def convert_dataset(
        self,
        input_path: Path,
        output_path: Path,
        add_prompt: bool = True,
    ) -> None:
        """
        Convert a BIO-labeled dataset to JSON format.

        Args:
            input_path: Path to BIO-labeled JSON file
            output_path: Path to save converted data
            add_prompt: Whether to add instruction prompt to input
        """
        print(f"Converting {input_path} to JSON format...")

        with open(input_path, "r") as f:
            data = json.load(f)

        converted_examples = []

        for example in data:
            try:
                converted = self._convert_example(example, add_prompt)
                converted_examples.append(converted)
            except Exception as e:
                print(f"Warning: Failed to convert example {example.get('id', 'unknown')}: {e}")
                continue

        # Save converted data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(converted_examples, f, indent=2, ensure_ascii=False)

        print(f"Converted {len(converted_examples)} examples to {output_path}")

    def _convert_example(self, example: Dict, add_prompt: bool = True) -> Dict:
        """
        Convert a single BIO example to JSON format.

        Args:
            example: BIO-labeled example
            add_prompt: Whether to add instruction prompt

        Returns:
            Converted example with input and target
        """
        tokens = example["tokens"]
        labels = example["labels"]

        # Extract entities from BIO labels
        entities = self._extract_entities_from_bio(tokens, labels)

        # Create structured invoice
        invoice = self._create_invoice_structure(entities)

        # Format as JSON string
        json_output = json.dumps(invoice, ensure_ascii=False)

        # Create input text
        input_text = example.get("text", " ".join(tokens))
        if add_prompt:
            input_text = f"extract invoice fields: {input_text}"

        return {
            "id": example.get("id", "unknown"),
            "input": input_text,
            "target": json_output,
            "metadata": example.get("metadata", {}),
        }

    def _extract_entities_from_bio(
        self, tokens: List[str], labels: List[str]
    ) -> Dict[str, List[str]]:
        """
        Extract entities from BIO-labeled tokens.

        Args:
            tokens: List of tokens
            labels: List of BIO labels

        Returns:
            Dictionary mapping entity types to values
        """
        entities = {}
        current_entity_type = None
        current_entity_tokens = []

        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                # Save previous entity
                if current_entity_type and current_entity_tokens:
                    self._add_entity(entities, current_entity_type, current_entity_tokens)

                # Start new entity
                current_entity_type = label[2:]  # Remove 'B-' prefix
                current_entity_tokens = [token]

            elif label.startswith("I-"):
                # Continue current entity
                entity_type = label[2:]  # Remove 'I-' prefix
                if entity_type == current_entity_type:
                    current_entity_tokens.append(token)
                else:
                    # Label inconsistency, start new entity
                    if current_entity_type and current_entity_tokens:
                        self._add_entity(entities, current_entity_type, current_entity_tokens)
                    current_entity_type = entity_type
                    current_entity_tokens = [token]

            else:  # "O" label
                # Save previous entity and reset
                if current_entity_type and current_entity_tokens:
                    self._add_entity(entities, current_entity_type, current_entity_tokens)
                current_entity_type = None
                current_entity_tokens = []

        # Save last entity
        if current_entity_type and current_entity_tokens:
            self._add_entity(entities, current_entity_type, current_entity_tokens)

        return entities

    def _add_entity(
        self, entities: Dict[str, List[str]], entity_type: str, tokens: List[str]
    ) -> None:
        """
        Add entity to entities dictionary.

        Args:
            entities: Dictionary to add to
            entity_type: Type of entity
            tokens: Entity tokens
        """
        entity_value = " ".join(tokens)

        if entity_type not in entities:
            entities[entity_type] = []

        entities[entity_type].append(entity_value)

    def _create_invoice_structure(self, entities: Dict[str, List[str]]) -> Dict:
        """
        Create structured invoice from entities.

        Args:
            entities: Dictionary of extracted entities

        Returns:
            Structured invoice dictionary
        """
        # Helper to get first value or None
        def first_or_none(key):
            return entities.get(key, [None])[0]

        # Helper to get all values or empty list
        def all_values(key):
            return entities.get(key, [])

        # Build customer info
        customer = {
            "name": first_or_none("CUSTOMER_NAME"),
            "address": " ".join(all_values("CUSTOMER_ADDRESS")) or None,
        }

        # Build items list
        items = []
        item_names = all_values("ITEM")
        item_prices = all_values("ITEM_PRICE")

        # Match items with prices
        for i, item_name in enumerate(item_names):
            item_price = item_prices[i] if i < len(item_prices) else None
            items.append({"name": item_name, "price": item_price})

        # Build complete invoice
        invoice = {
            "invoice_number": first_or_none("INVOICE_NUMBER"),
            "date": first_or_none("DATE"),
            "customer": customer,
            "items": items,
            "total": first_or_none("TOTAL"),
        }

        return invoice

    def convert_single_text(
        self, text: str, tokens: List[str], labels: List[str]
    ) -> Tuple[str, str]:
        """
        Convert a single text with BIO labels to JSON format.

        Useful for testing or one-off conversions.

        Args:
            text: Original invoice text
            tokens: List of tokens
            labels: List of BIO labels

        Returns:
            Tuple of (input_text, json_output)
        """
        example = {
            "id": "single",
            "text": text,
            "tokens": tokens,
            "labels": labels,
        }

        converted = self._convert_example(example, add_prompt=True)
        return converted["input"], converted["target"]


def convert_all_splits(
    input_dir: Path,
    output_dir: Path,
    splits: List[str] = ["train", "val", "test"],
) -> None:
    """
    Convert all dataset splits from BIO to JSON format.

    Args:
        input_dir: Directory containing BIO-labeled files
        output_dir: Directory to save converted files
        splits: List of split names to convert
    """
    converter = BIOToJSONConverter()

    for split in splits:
        input_path = input_dir / f"{split}.json"
        output_path = output_dir / f"{split}_json.json"

        if input_path.exists():
            converter.convert_dataset(input_path, output_path)
        else:
            print(f"Skipping {split}: {input_path} not found")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Convert BIO data to JSON format")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Input directory with BIO-labeled data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/json",
        help="Output directory for JSON-formatted data",
    )

    args = parser.parse_args()

    convert_all_splits(
        Path(args.input_dir),
        Path(args.output_dir),
    )
