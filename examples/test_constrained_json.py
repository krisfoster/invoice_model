"""Example script demonstrating constrained JSON generation."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.inference_json import ConstrainedJSONExtractor
from training.json_schema import validate_invoice_output


def test_sample_invoice():
    """Test extraction with a sample invoice."""

    # Sample invoice text
    sample_invoice = """
    INVOICE

    Invoice Number: INV-2024-001
    Date: January 15, 2024

    Bill To:
    John Smith
    123 Main Street
    New York, NY 10001

    Items:
    - Web Development Services    $2,500.00
    - SEO Optimization             $1,200.00
    - Content Writing              $800.00

    Total: $4,500.00

    Payment due within 30 days.
    Thank you for your business!
    """

    print("=" * 70)
    print("CONSTRAINED JSON GENERATION DEMO")
    print("=" * 70)

    print("\nSample Invoice:")
    print("-" * 70)
    print(sample_invoice)
    print("-" * 70)

    # Check if model exists
    model_path = Path("models/json_model/best_model")
    if not model_path.exists():
        print("\nError: Trained model not found!")
        print(f"Expected location: {model_path}")
        print("\nPlease train the model first using:")
        print("  python -m training.train_json --data-dir data/processed --convert-data")
        return

    print("\nLoading model...")
    print(f"Model path: {model_path}")

    # Test with constraints enabled
    print("\n" + "=" * 70)
    print("TEST 1: WITH CONSTRAINED GENERATION (Guaranteed Valid JSON)")
    print("=" * 70)

    extractor_constrained = ConstrainedJSONExtractor(
        model_path=model_path,
        device="auto",
        use_constraints=True,
    )

    result_constrained = extractor_constrained.extract_from_text(sample_invoice)

    print("\nExtracted JSON (Constrained):")
    print(json.dumps(result_constrained, indent=2))

    # Validate
    is_valid, error = validate_invoice_output(result_constrained)
    print(f"\nSchema Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    if not is_valid:
        print(f"Error: {error}")

    # Test without constraints
    print("\n" + "=" * 70)
    print("TEST 2: WITHOUT CONSTRAINED GENERATION (May Be Invalid)")
    print("=" * 70)

    extractor_unconstrained = ConstrainedJSONExtractor(
        model_path=model_path,
        device="auto",
        use_constraints=False,
    )

    result_unconstrained = extractor_unconstrained.extract_from_text(sample_invoice)

    print("\nExtracted JSON (Unconstrained):")
    print(json.dumps(result_unconstrained, indent=2))

    # Validate
    is_valid, error = validate_invoice_output(result_unconstrained)
    print(f"\nSchema Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    if not is_valid:
        print(f"Error: {error}")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print("\nKey Differences:")
    print(f"  Constrained:   Always valid JSON matching schema")
    print(f"  Unconstrained: May produce invalid JSON or wrong schema")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


def test_malformed_invoice():
    """Test with a malformed invoice to show constraint benefits."""

    malformed_invoice = """
    Some random text that doesn't look like an invoice at all.
    Maybe some numbers: 12345
    And a date: 2024-01-15
    But no clear structure!
    """

    print("\n" + "=" * 70)
    print("TEST 3: MALFORMED INPUT (Testing Robustness)")
    print("=" * 70)

    print("\nMalformed Input:")
    print("-" * 70)
    print(malformed_invoice)
    print("-" * 70)

    model_path = Path("models/json_model/best_model")
    if not model_path.exists():
        print("\nModel not found, skipping test.")
        return

    extractor = ConstrainedJSONExtractor(
        model_path=model_path,
        device="auto",
        use_constraints=True,
    )

    result = extractor.extract_from_text(malformed_invoice)

    print("\nExtracted JSON (Constrained):")
    print(json.dumps(result, indent=2))

    # Even with malformed input, output should be valid JSON
    is_valid, error = validate_invoice_output(result)
    print(f"\nSchema Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    print("\nNote: Even with poor input, constrained generation produces valid JSON!")
    print("      (Though field values may be None or incorrect)")


def main():
    """Run all tests."""
    try:
        test_sample_invoice()
        test_malformed_invoice()

        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
