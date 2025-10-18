"""Evaluation metrics for invoice field extraction."""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


class InvoiceMetrics:
    """Calculate metrics for invoice field extraction."""

    def __init__(self, id2label: Dict[int, str]):
        """
        Initialize metrics calculator.

        Args:
            id2label: Mapping from label ids to names
        """
        self.id2label = id2label

    def compute_metrics(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: Predicted label ids [batch_size, seq_len]
            labels: True label ids [batch_size, seq_len]

        Returns:
            Dictionary of metrics
        """
        # Convert ids to labels
        true_labels = []
        pred_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq = []
            pred_seq_labels = []

            for pred, label in zip(pred_seq, label_seq):
                if label != -100:  # Ignore special tokens
                    true_seq.append(self.id2label[label])
                    pred_seq_labels.append(self.id2label[pred])

            if true_seq:  # Only add non-empty sequences
                true_labels.append(true_seq)
                pred_labels.append(pred_seq_labels)

        # Calculate metrics using seqeval
        metrics = {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
        }

        # Calculate per-entity metrics
        entity_metrics = self._compute_entity_metrics(true_labels, pred_labels)
        metrics.update(entity_metrics)

        # Calculate exact match accuracy
        exact_match = self._compute_exact_match(true_labels, pred_labels)
        metrics["exact_match"] = exact_match

        return metrics

    def _compute_entity_metrics(
        self, true_labels: List[List[str]], pred_labels: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute per-entity type metrics.

        Args:
            true_labels: True label sequences
            pred_labels: Predicted label sequences

        Returns:
            Dictionary with per-entity metrics
        """
        # Get unique entity types (remove B- and I- prefixes)
        entity_types = set()
        for seq in true_labels:
            for label in seq:
                if label != "O":
                    entity_type = label.split("-")[1]
                    entity_types.add(entity_type)

        metrics = {}

        for entity_type in entity_types:
            # Filter labels for this entity type
            filtered_true = []
            filtered_pred = []

            for true_seq, pred_seq in zip(true_labels, pred_labels):
                filtered_true_seq = []
                filtered_pred_seq = []

                for true_label, pred_label in zip(true_seq, pred_seq):
                    # Keep labels for this entity or O
                    if true_label == "O" or entity_type in true_label:
                        filtered_true_seq.append(true_label)
                    else:
                        filtered_true_seq.append("O")

                    if pred_label == "O" or entity_type in pred_label:
                        filtered_pred_seq.append(pred_label)
                    else:
                        filtered_pred_seq.append("O")

                filtered_true.append(filtered_true_seq)
                filtered_pred.append(filtered_pred_seq)

            # Calculate metrics for this entity
            try:
                metrics[f"{entity_type}_precision"] = precision_score(
                    filtered_true, filtered_pred
                )
                metrics[f"{entity_type}_recall"] = recall_score(
                    filtered_true, filtered_pred
                )
                metrics[f"{entity_type}_f1"] = f1_score(filtered_true, filtered_pred)
            except:
                # Handle cases where entity doesn't appear
                metrics[f"{entity_type}_precision"] = 0.0
                metrics[f"{entity_type}_recall"] = 0.0
                metrics[f"{entity_type}_f1"] = 0.0

        return metrics

    def _compute_exact_match(
        self, true_labels: List[List[str]], pred_labels: List[List[str]]
    ) -> float:
        """
        Compute exact match accuracy (percentage of sequences that match exactly).

        Args:
            true_labels: True label sequences
            pred_labels: Predicted label sequences

        Returns:
            Exact match accuracy
        """
        if not true_labels:
            return 0.0

        exact_matches = sum(
            1 for true, pred in zip(true_labels, pred_labels) if true == pred
        )

        return exact_matches / len(true_labels)

    def get_classification_report(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> str:
        """
        Get detailed classification report.

        Args:
            predictions: Predicted label ids
            labels: True label ids

        Returns:
            Classification report string
        """
        # Convert ids to labels
        true_labels = []
        pred_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq = []
            pred_seq_labels = []

            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    true_seq.append(self.id2label[label])
                    pred_seq_labels.append(self.id2label[pred])

            if true_seq:
                true_labels.append(true_seq)
                pred_labels.append(pred_seq_labels)

        return classification_report(true_labels, pred_labels)

    def extract_entities(
        self, tokens: List[str], predictions: List[int]
    ) -> Dict[str, List[str]]:
        """
        Extract entities from predictions.

        Args:
            tokens: List of tokens
            predictions: List of predicted label ids

        Returns:
            Dictionary mapping entity types to extracted values
        """
        entities = defaultdict(list)

        current_entity = None
        current_tokens = []

        for token, pred_id in zip(tokens, predictions):
            if pred_id == -100:
                continue

            label = self.id2label[pred_id]

            if label.startswith("B-"):
                # Start of new entity
                if current_entity and current_tokens:
                    entities[current_entity].append(" ".join(current_tokens))

                current_entity = label[2:]  # Remove B- prefix
                current_tokens = [token]

            elif label.startswith("I-"):
                # Continuation of entity
                entity_type = label[2:]  # Remove I- prefix
                if entity_type == current_entity:
                    current_tokens.append(token)
                else:
                    # New entity started without B- tag
                    if current_entity and current_tokens:
                        entities[current_entity].append(" ".join(current_tokens))
                    current_entity = entity_type
                    current_tokens = [token]

            else:
                # Outside entity (O label)
                if current_entity and current_tokens:
                    entities[current_entity].append(" ".join(current_tokens))

                current_entity = None
                current_tokens = []

        # Add last entity
        if current_entity and current_tokens:
            entities[current_entity].append(" ".join(current_tokens))

        return dict(entities)
