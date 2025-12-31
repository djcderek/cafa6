"""Kaggle submission file generation.

Submission format:
- No header
- Columns: protein_id, GO_term, score
- Score in (0, 1] (must be > 0)
- Maximum 1500 terms per protein
- Only valid GO terms allowed
"""

from pathlib import Path
from typing import TextIO


def filter_and_sort_predictions(
    predictions: dict[str, dict[str, float]],
    valid_terms: set[str],
    max_terms_per_protein: int = 1500,
    min_score: float = 1e-6,
) -> dict[str, list[tuple[str, float]]]:
    """
    Filter and sort predictions for submission.

    Args:
        predictions: Dictionary mapping protein IDs to term scores.
        valid_terms: Set of valid GO terms.
        max_terms_per_protein: Maximum terms per protein.
        min_score: Minimum score threshold (must be > 0).

    Returns:
        Dictionary mapping protein IDs to sorted (term, score) lists.
    """
    filtered: dict[str, list[tuple[str, float]]] = {}

    for protein_id, term_scores in predictions.items():
        # Filter valid terms and clip scores
        valid_scores: list[tuple[str, float]] = []

        for term, score in term_scores.items():
            if term not in valid_terms:
                continue

            # Ensure score is in (0, 1]
            score = max(min_score, min(score, 1.0))
            valid_scores.append((term, score))

        # Sort by score descending and limit to max_terms_per_protein
        valid_scores.sort(key=lambda x: -x[1])
        valid_scores = valid_scores[:max_terms_per_protein]

        filtered[protein_id] = valid_scores

    return filtered


def write_submission(
    predictions: dict[str, dict[str, float]],
    valid_terms: set[str],
    output_path: Path | str,
    max_terms_per_protein: int = 1500,
    min_score: float = 1e-6,
) -> int:
    """
    Write a Kaggle submission file.

    Args:
        predictions: Dictionary mapping protein IDs to term scores.
        valid_terms: Set of valid GO terms.
        output_path: Path to save the submission file.
        max_terms_per_protein: Maximum terms per protein.
        min_score: Minimum score threshold.

    Returns:
        Total number of predictions written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter and sort predictions
    filtered = filter_and_sort_predictions(
        predictions,
        valid_terms,
        max_terms_per_protein,
        min_score,
    )

    # Write submission file (no header)
    total_predictions = 0

    with open(output_path, "w") as f:
        # Sort proteins for deterministic output
        for protein_id in sorted(filtered.keys()):
            for term, score in filtered[protein_id]:
                f.write(f"{protein_id}\t{term}\t{score:.6f}\n")
                total_predictions += 1

    return total_predictions


def validate_submission(
    submission_path: Path | str,
    valid_terms: set[str],
    max_terms_per_protein: int = 1500,
) -> dict[str, list[str]]:
    """
    Validate a submission file.

    Args:
        submission_path: Path to the submission file.
        valid_terms: Set of valid GO terms.
        max_terms_per_protein: Maximum terms per protein.

    Returns:
        Dictionary of errors by category.
    """
    submission_path = Path(submission_path)
    errors: dict[str, list[str]] = {
        "invalid_terms": [],
        "invalid_scores": [],
        "too_many_terms": [],
        "format_errors": [],
    }

    protein_term_counts: dict[str, int] = {}

    with open(submission_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                errors["format_errors"].append(f"Line {line_num}: Expected 3 columns, got {len(parts)}")
                continue

            protein_id, term, score_str = parts

            # Check term validity
            if term not in valid_terms:
                errors["invalid_terms"].append(f"Line {line_num}: Invalid term {term}")

            # Check score validity
            try:
                score = float(score_str)
                if score <= 0 or score > 1:
                    errors["invalid_scores"].append(
                        f"Line {line_num}: Score {score} not in (0, 1]"
                    )
            except ValueError:
                errors["invalid_scores"].append(f"Line {line_num}: Invalid score {score_str}")

            # Count terms per protein
            protein_term_counts[protein_id] = protein_term_counts.get(protein_id, 0) + 1

    # Check term counts
    for protein_id, count in protein_term_counts.items():
        if count > max_terms_per_protein:
            errors["too_many_terms"].append(
                f"Protein {protein_id}: {count} terms (max {max_terms_per_protein})"
            )

    return errors


class SubmissionWriter:
    """
    Streaming submission writer for large prediction sets.

    Use this for memory-efficient writing when predictions are generated in batches.
    """

    def __init__(
        self,
        output_path: Path | str,
        valid_terms: set[str],
        max_terms_per_protein: int = 1500,
        min_score: float = 1e-6,
    ) -> None:
        """
        Initialize the submission writer.

        Args:
            output_path: Path to save the submission file.
            valid_terms: Set of valid GO terms.
            max_terms_per_protein: Maximum terms per protein.
            min_score: Minimum score threshold.
        """
        self.output_path = Path(output_path)
        self.valid_terms = valid_terms
        self.max_terms_per_protein = max_terms_per_protein
        self.min_score = min_score
        self.total_predictions = 0

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = None

    def __enter__(self) -> "SubmissionWriter":
        """Open the file for writing."""
        self._file = open(self.output_path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the file."""
        if self._file is not None:
            self._file.close()

    def write_protein(
        self,
        protein_id: str,
        term_scores: dict[str, float],
    ) -> int:
        """
        Write predictions for a single protein.

        Args:
            protein_id: Protein ID.
            term_scores: Dictionary mapping GO terms to scores.

        Returns:
            Number of predictions written.
        """
        assert self._file is not None, "Writer not opened. Use 'with' context."

        # Filter and sort
        valid_scores: list[tuple[str, float]] = []

        for term, score in term_scores.items():
            if term not in self.valid_terms:
                continue

            score = max(self.min_score, min(score, 1.0))
            valid_scores.append((term, score))

        valid_scores.sort(key=lambda x: -x[1])
        valid_scores = valid_scores[: self.max_terms_per_protein]

        # Write
        count = 0
        for term, score in valid_scores:
            self._file.write(f"{protein_id}\t{term}\t{score:.6f}\n")
            count += 1

        self.total_predictions += count
        return count

