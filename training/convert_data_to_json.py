"""Standalone script to convert BIO-labeled data to JSON format."""

import argparse
from pathlib import Path

from training.bio_to_json_converter import BIOToJSONConverter


def main():
    """Convert BIO-labeled data to JSON format."""
    parser = argparse.ArgumentParser(
        description="Convert BIO-labeled data to JSON target format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory with BIO-labeled data",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file or directory for JSON-formatted data",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to convert (if input is directory)",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Don't add instruction prompt to inputs",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    converter = BIOToJSONConverter()

    # Check if input is directory or file
    if input_path.is_dir():
        print(f"Converting directory: {input_path}")
        print(f"Splits: {args.splits}")

        for split in args.splits:
            input_file = input_path / f"{split}.json"
            output_file = output_path / f"{split}_json.json"

            if input_file.exists():
                print(f"\nConverting {split} split...")
                converter.convert_dataset(
                    input_file,
                    output_file,
                    add_prompt=not args.no_prompt,
                )
            else:
                print(f"Skipping {split}: {input_file} not found")

    elif input_path.is_file():
        print(f"Converting file: {input_path}")
        converter.convert_dataset(
            input_path,
            output_path,
            add_prompt=not args.no_prompt,
        )

    else:
        print(f"Error: {input_path} not found")
        return 1

    print("\nConversion complete!")
    return 0


if __name__ == "__main__":
    exit(main())
