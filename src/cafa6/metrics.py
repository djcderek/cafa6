"""Evaluation metrics for CAFA weighted max-F1 scoring.

The CAFA evaluation metric is a weighted max-F1 score that:
1. For each threshold, computes precision and recall across all proteins
2. Uses Information Accretion (IA) weights for terms
3. Finds the threshold that maximizes F1
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class MetricResult:
    """Container for metric computation results."""

    score: float
    precision: float
    recall: float
    f1: float
    threshold: float
    num_proteins: int
    num_predictions: int

    def __str__(self) -> str:
        """Format result as string."""
        return (
            f"Score: {self.score:.4f} | "
            f"P: {self.precision:.4f} | "
            f"R: {self.recall:.4f} | "
            f"F1: {self.f1:.4f} | "
            f"Threshold: {self.threshold:.4f}"
        )


class MetricInterface(Protocol):
    """Interface for evaluation metrics."""

    def compute(
        self,
        predictions: dict[str, dict[str, float]],
        ground_truth: dict[str, list[str]],
    ) -> MetricResult:
        """
        Compute the metric.

        Args:
            predictions: Dictionary mapping protein IDs to term scores.
            ground_truth: Dictionary mapping protein IDs to true GO terms.

        Returns:
            MetricResult with computed scores.
        """
        ...


def compute_precision_recall_at_threshold(
    predictions: dict[str, dict[str, float]],
    ground_truth: dict[str, list[str]],
    threshold: float,
    term_weights: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 at a given threshold.

    If term_weights are provided, computes weighted precision/recall.

    Args:
        predictions: Dictionary mapping protein IDs to term scores.
        ground_truth: Dictionary mapping protein IDs to true GO terms.
        threshold: Score threshold for positive predictions.
        term_weights: Optional IA weights for GO terms.

    Returns:
        Tuple of (precision, recall, f1).
    """
    total_tp_weight = 0.0
    total_fp_weight = 0.0
    total_fn_weight = 0.0

    # Only evaluate proteins that are in ground truth
    for protein_id, true_terms in ground_truth.items():
        true_set = set(true_terms)

        if protein_id in predictions:
            pred_terms = {
                term for term, score in predictions[protein_id].items()
                if score >= threshold
            }
        else:
            pred_terms = set()

        # Compute TP, FP, FN
        tp_terms = pred_terms & true_set
        fp_terms = pred_terms - true_set
        fn_terms = true_set - pred_terms

        if term_weights is not None:
            # Weighted by IA
            tp_weight = sum(term_weights.get(t, 1.0) for t in tp_terms)
            fp_weight = sum(term_weights.get(t, 1.0) for t in fp_terms)
            fn_weight = sum(term_weights.get(t, 1.0) for t in fn_terms)
        else:
            # Unweighted
            tp_weight = len(tp_terms)
            fp_weight = len(fp_terms)
            fn_weight = len(fn_terms)

        total_tp_weight += tp_weight
        total_fp_weight += fp_weight
        total_fn_weight += fn_weight

    # Compute precision, recall, F1
    precision = total_tp_weight / max(total_tp_weight + total_fp_weight, 1e-9)
    recall = total_tp_weight / max(total_tp_weight + total_fn_weight, 1e-9)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return precision, recall, f1


def find_optimal_threshold(
    predictions: dict[str, dict[str, float]],
    ground_truth: dict[str, list[str]],
    term_weights: dict[str, float] | None = None,
    thresholds: list[float] | None = None,
) -> tuple[float, float, float, float]:
    """
    Find the threshold that maximizes F1 score.

    Args:
        predictions: Dictionary mapping protein IDs to term scores.
        ground_truth: Dictionary mapping protein IDs to true GO terms.
        term_weights: Optional IA weights for GO terms.
        thresholds: Thresholds to evaluate (default: 0.01 to 0.99).

    Returns:
        Tuple of (best_threshold, precision, recall, f1).
    """
    if thresholds is None:
        thresholds = [i / 100.0 for i in range(1, 100)]

    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0

    for threshold in thresholds:
        precision, recall, f1 = compute_precision_recall_at_threshold(
            predictions, ground_truth, threshold, term_weights
        )

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_threshold = threshold

    return best_threshold, best_precision, best_recall, best_f1


class WeightedMaxF1:
    """
    Weighted max-F1 metric for CAFA evaluation.

    The metric:
    1. Optionally weights terms by their Information Accretion (IA)
    2. Searches over thresholds to find the one that maximizes F1
    3. Reports precision, recall, and F1 at the optimal threshold
    """

    def __init__(
        self,
        term_ia_weights: dict[str, float] | None = None,
        thresholds: list[float] | None = None,
    ) -> None:
        """
        Initialize the metric.

        Args:
            term_ia_weights: Information Accretion weights for GO terms.
            thresholds: Thresholds to evaluate.
        """
        self.term_ia_weights = term_ia_weights
        self.thresholds = thresholds or [i / 100.0 for i in range(1, 100)]

    def compute(
        self,
        predictions: dict[str, dict[str, float]],
        ground_truth: dict[str, list[str]],
    ) -> MetricResult:
        """
        Compute the weighted max-F1 metric.

        Args:
            predictions: Dictionary mapping protein IDs to term scores.
            ground_truth: Dictionary mapping protein IDs to true GO terms.

        Returns:
            MetricResult with computed scores.
        """
        # Find optimal threshold
        threshold, precision, recall, f1 = find_optimal_threshold(
            predictions,
            ground_truth,
            self.term_ia_weights,
            self.thresholds,
        )

        # Count statistics
        common_proteins = set(predictions.keys()) & set(ground_truth.keys())
        num_predictions = sum(
            sum(1 for score in terms.values() if score >= threshold)
            for pid, terms in predictions.items()
            if pid in common_proteins
        )

        return MetricResult(
            score=f1,  # The main score is the max-F1
            precision=precision,
            recall=recall,
            f1=f1,
            threshold=threshold,
            num_proteins=len(common_proteins),
            num_predictions=num_predictions,
        )

    def compute_at_threshold(
        self,
        predictions: dict[str, dict[str, float]],
        ground_truth: dict[str, list[str]],
        threshold: float = 0.5,
    ) -> tuple[float, float, float]:
        """
        Compute precision, recall, F1 at a fixed threshold.

        Args:
            predictions: Dictionary mapping protein IDs to term scores.
            ground_truth: Dictionary mapping protein IDs to true GO terms.
            threshold: Score threshold for positive predictions.

        Returns:
            Tuple of (precision, recall, f1).
        """
        return compute_precision_recall_at_threshold(
            predictions, ground_truth, threshold, self.term_ia_weights
        )


# Backwards compatibility alias
WeightedMaxF1Placeholder = WeightedMaxF1


def evaluate_predictions(
    predictions: dict[str, dict[str, float]],
    ground_truth: dict[str, list[str]],
    term_ia_weights: dict[str, float] | None = None,
) -> MetricResult:
    """
    Convenience function to evaluate predictions.

    Args:
        predictions: Dictionary mapping protein IDs to term scores.
        ground_truth: Dictionary mapping protein IDs to true GO terms.
        term_ia_weights: Optional Information Accretion weights.

    Returns:
        MetricResult with evaluation scores.
    """
    metric = WeightedMaxF1(term_ia_weights)
    return metric.compute(predictions, ground_truth)


class LocalEvaluator:
    """
    Local evaluation harness for development and validation.

    Loads ground truth and IA weights, then evaluates predictions.
    """

    def __init__(
        self,
        ground_truth_path: str | Path | None = None,
        term_ia_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            ground_truth_path: Path to ground truth annotations (TSV).
            term_ia_path: Path to term IA weights (TSV).
        """
        self.ground_truth_path = Path(ground_truth_path) if ground_truth_path else None
        self.term_ia_path = Path(term_ia_path) if term_ia_path else None
        self.ground_truth: dict[str, list[str]] = {}
        self.term_ia_weights: dict[str, float] = {}

    def load_ground_truth(self) -> None:
        """
        Load ground truth annotations from TSV file.

        Supports two formats:
        - CAFA format: EntryID<TAB>term<TAB>aspect (with header)
        - Simple format: protein_id<TAB>GO_term (no header)
        """
        if self.ground_truth_path is None:
            return

        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {self.ground_truth_path}"
            )

        self.ground_truth = {}
        first_line = True

        with open(self.ground_truth_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")

                # Skip header row
                if first_line:
                    first_line = False
                    if parts[0].lower() in ("entryid", "protein_id", "id"):
                        continue

                if len(parts) >= 2:
                    protein_id, term = parts[0], parts[1]
                    if protein_id not in self.ground_truth:
                        self.ground_truth[protein_id] = []
                    self.ground_truth[protein_id].append(term)

    def load_term_ia(self) -> None:
        """
        Load term Information Accretion weights from TSV file.

        Format: GO_term<TAB>IA_weight (no header)
        """
        if self.term_ia_path is None:
            return

        if not self.term_ia_path.exists():
            raise FileNotFoundError(
                f"IA weights file not found: {self.term_ia_path}"
            )

        self.term_ia_weights = {}

        with open(self.term_ia_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 2:
                    term = parts[0]
                    if term.startswith("GO:"):
                        try:
                            self.term_ia_weights[term] = float(parts[1])
                        except ValueError:
                            self.term_ia_weights[term] = 0.0

    def load(self) -> None:
        """Load both ground truth and IA weights."""
        self.load_ground_truth()
        self.load_term_ia()

    def evaluate(
        self,
        predictions: dict[str, dict[str, float]],
    ) -> MetricResult:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: Dictionary mapping protein IDs to term scores.

        Returns:
            MetricResult with evaluation scores.
        """
        weights = self.term_ia_weights if self.term_ia_weights else None
        return evaluate_predictions(predictions, self.ground_truth, weights)

    def evaluate_submission(
        self,
        submission_path: str | Path,
    ) -> MetricResult:
        """
        Evaluate a submission file against ground truth.

        Args:
            submission_path: Path to submission TSV file.

        Returns:
            MetricResult with evaluation scores.
        """
        submission_path = Path(submission_path)

        # Parse submission file
        predictions: dict[str, dict[str, float]] = {}

        with open(submission_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 3:
                    protein_id, term, score_str = parts[0], parts[1], parts[2]
                    try:
                        score = float(score_str)
                    except ValueError:
                        continue

                    if protein_id not in predictions:
                        predictions[protein_id] = {}
                    predictions[protein_id][term] = score

        return self.evaluate(predictions)


def compute_per_ontology_metrics(
    predictions: dict[str, dict[str, float]],
    ground_truth: dict[str, list[str]],
    term_to_ontology: dict[str, str],
    term_ia_weights: dict[str, float] | None = None,
) -> dict[str, MetricResult]:
    """
    Compute metrics separately for each ontology (MF, BP, CC).

    Args:
        predictions: Dictionary mapping protein IDs to term scores.
        ground_truth: Dictionary mapping protein IDs to true GO terms.
        term_to_ontology: Mapping from GO terms to ontology type.
        term_ia_weights: Optional IA weights for GO terms.

    Returns:
        Dictionary mapping ontology names to MetricResults.
    """
    results = {}

    for ontology in ["MF", "BP", "CC"]:
        # Filter predictions to this ontology
        ont_predictions: dict[str, dict[str, float]] = {}
        for protein_id, term_scores in predictions.items():
            ont_scores = {
                term: score
                for term, score in term_scores.items()
                if term_to_ontology.get(term) == ontology
            }
            if ont_scores:
                ont_predictions[protein_id] = ont_scores

        # Filter ground truth to this ontology
        ont_ground_truth: dict[str, list[str]] = {}
        for protein_id, terms in ground_truth.items():
            ont_terms = [
                term for term in terms
                if term_to_ontology.get(term) == ontology
            ]
            if ont_terms:
                ont_ground_truth[protein_id] = ont_terms

        # Compute metrics for this ontology
        if ont_predictions and ont_ground_truth:
            metric = WeightedMaxF1(term_ia_weights)
            results[ontology] = metric.compute(ont_predictions, ont_ground_truth)
        else:
            results[ontology] = MetricResult(
                score=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                threshold=0.5,
                num_proteins=0,
                num_predictions=0,
            )

    return results
