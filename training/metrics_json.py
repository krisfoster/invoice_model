"""Metrics for JSON generation evaluation."""

import json
from typing import Dict, List, Any
from collections import defaultdict


class JSONGenerationMetrics:
    """Calculate metrics for JSON generation tasks."""

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics for JSON generation.

        Args:
            predictions: List of predicted JSON strings
            references: List of reference JSON strings

        Returns:
            Dictionary of metrics
        """
        total = len(predictions)

        # Track various metrics
        valid_json_count = 0
        exact_match_count = 0
        field_accuracies = []
        field_precisions = []
        field_recalls = []
        field_f1s = []

        for pred_str, ref_str in zip(predictions, references):
            # Parse JSONs
            try:
                pred_json = json.loads(pred_str)
                valid_json_count += 1
            except json.JSONDecodeError:
                pred_json = None

            try:
                ref_json = json.loads(ref_str)
            except json.JSONDecodeError:
                ref_json = None
                continue  # Skip if reference is invalid

            # Skip if prediction couldn't be parsed
            if pred_json is None:
                field_accuracies.append(0.0)
                field_precisions.append(0.0)
                field_recalls.append(0.0)
                field_f1s.append(0.0)
                continue

            # Exact match
            if pred_json == ref_json:
                exact_match_count += 1

            # Field-level metrics
            field_acc, field_prec, field_rec, field_f1 = self._compute_field_metrics(
                pred_json, ref_json
            )
            field_accuracies.append(field_acc)
            field_precisions.append(field_prec)
            field_recalls.append(field_rec)
            field_f1s.append(field_f1)

        # Aggregate metrics
        metrics = {
            "valid_json_rate": valid_json_count / total if total > 0 else 0.0,
            "exact_match": exact_match_count / total if total > 0 else 0.0,
            "field_accuracy": sum(field_accuracies) / len(field_accuracies) if field_accuracies else 0.0,
            "field_precision": sum(field_precisions) / len(field_precisions) if field_precisions else 0.0,
            "field_recall": sum(field_recalls) / len(field_recalls) if field_recalls else 0.0,
            "field_f1": sum(field_f1s) / len(field_f1s) if field_f1s else 0.0,
        }

        return metrics

    def _compute_field_metrics(
        self, pred: Dict, ref: Dict
    ) -> tuple[float, float, float, float]:
        """
        Compute field-level metrics.

        Args:
            pred: Predicted dictionary
            ref: Reference dictionary

        Returns:
            Tuple of (accuracy, precision, recall, f1)
        """
        # Flatten nested dictionaries
        pred_flat = self._flatten_dict(pred)
        ref_flat = self._flatten_dict(ref)

        # Get all keys
        all_keys = set(pred_flat.keys()) | set(ref_flat.keys())

        if not all_keys:
            return 1.0, 1.0, 1.0, 1.0  # Both empty is perfect match

        # Count matches
        correct = 0
        pred_non_none = 0
        ref_non_none = 0

        for key in all_keys:
            pred_val = pred_flat.get(key)
            ref_val = ref_flat.get(key)

            # Count non-None values
            if pred_val is not None and pred_val != "" and pred_val != []:
                pred_non_none += 1
            if ref_val is not None and ref_val != "" and ref_val != []:
                ref_non_none += 1

            # Check if values match
            if self._values_match(pred_val, ref_val):
                correct += 1

        # Calculate metrics
        accuracy = correct / len(all_keys) if all_keys else 0.0
        precision = correct / pred_non_none if pred_non_none > 0 else 0.0
        recall = correct / ref_non_none if ref_non_none > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return accuracy, precision, recall, f1

    def _flatten_dict(
        self, d: Dict, parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Handle lists specially
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(
                            self._flatten_dict(item, f"{new_key}[{i}]", sep=sep).items()
                        )
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))

        return dict(items)

    def _values_match(self, val1: Any, val2: Any) -> bool:
        """
        Check if two values match (with fuzzy matching for strings).

        Args:
            val1: First value
            val2: Second value

        Returns:
            True if values match
        """
        # Handle None cases
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False

        # Handle empty cases
        if (val1 == "" or val1 == []) and (val2 == "" or val2 == []):
            return True

        # String comparison (case-insensitive, whitespace-normalized)
        if isinstance(val1, str) and isinstance(val2, str):
            return val1.strip().lower() == val2.strip().lower()

        # Direct comparison for other types
        return val1 == val2

    def print_detailed_metrics(
        self, predictions: List[str], references: List[str]
    ) -> None:
        """
        Print detailed metrics with per-field breakdown.

        Args:
            predictions: List of predicted JSON strings
            references: List of reference JSON strings
        """
        metrics = self.compute_metrics(predictions, references)

        print("\n" + "=" * 50)
        print("JSON Generation Metrics")
        print("=" * 50)
        print(f"Valid JSON Rate:     {metrics['valid_json_rate']:.2%}")
        print(f"Exact Match:         {metrics['exact_match']:.2%}")
        print(f"Field Accuracy:      {metrics['field_accuracy']:.2%}")
        print(f"Field Precision:     {metrics['field_precision']:.2%}")
        print(f"Field Recall:        {metrics['field_recall']:.2%}")
        print(f"Field F1:            {metrics['field_f1']:.2%}")
        print("=" * 50)

        # Per-field breakdown
        field_stats = self._compute_per_field_stats(predictions, references)

        print("\nPer-Field Statistics:")
        print("-" * 50)
        for field, stats in sorted(field_stats.items()):
            print(f"{field:30s} {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        print("=" * 50)

    def _compute_per_field_stats(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Dict]:
        """
        Compute per-field statistics.

        Args:
            predictions: List of predicted JSON strings
            references: List of reference JSON strings

        Returns:
            Dictionary of per-field statistics
        """
        field_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for pred_str, ref_str in zip(predictions, references):
            try:
                pred_json = json.loads(pred_str)
            except json.JSONDecodeError:
                pred_json = {}

            try:
                ref_json = json.loads(ref_str)
            except json.JSONDecodeError:
                continue

            pred_flat = self._flatten_dict(pred_json)
            ref_flat = self._flatten_dict(ref_json)

            all_keys = set(pred_flat.keys()) | set(ref_flat.keys())

            for key in all_keys:
                field_stats[key]["total"] += 1
                if self._values_match(pred_flat.get(key), ref_flat.get(key)):
                    field_stats[key]["correct"] += 1

        # Calculate accuracies
        for field in field_stats:
            stats = field_stats[field]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        return dict(field_stats)
