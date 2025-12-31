"""Inference pipeline for generating predictions."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MultiHeadMLP, load_model
from .ontology import GOTermIndex, OntologyType
from .propagation import GOPropagator


@torch.no_grad()
def run_inference(
    model: MultiHeadMLP,
    dataloader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Run inference on a dataloader.

    Args:
        model: Trained model.
        dataloader: Dataloader for inference data.
        device: Device for inference.

    Returns:
        Dictionary mapping protein IDs to ontology probabilities:
            {protein_id: {"MF": array, "BP": array, "CC": array}}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    results: dict[str, dict[str, np.ndarray]] = {}

    for batch in tqdm(dataloader, desc="Inference"):
        protein_ids = batch["protein_id"]
        embeddings = batch["embedding"].to(device)

        # Get probabilities
        proba = model.predict_proba(embeddings)

        # Store results for each protein
        for i, protein_id in enumerate(protein_ids):
            results[protein_id] = {
                "MF": proba["proba_mf"][i].cpu().numpy(),
                "BP": proba["proba_bp"][i].cpu().numpy(),
                "CC": proba["proba_cc"][i].cpu().numpy(),
            }

    return results


def probabilities_to_term_scores(
    probabilities: dict[str, np.ndarray],
    term_index: GOTermIndex,
) -> dict[str, float]:
    """
    Convert probability arrays to term-score dictionary.

    Args:
        probabilities: Dictionary with ontology probabilities.
        term_index: GO term index.

    Returns:
        Dictionary mapping GO terms to scores.
    """
    term_scores: dict[str, float] = {}

    for ontology in ["MF", "BP", "CC"]:
        ont: OntologyType = ontology  # type: ignore
        proba = probabilities[ontology]

        for idx, score in enumerate(proba):
            term = term_index.get_term(idx, ont)
            if term is not None and score > 0:
                term_scores[term] = float(score)

    return term_scores


def apply_propagation(
    predictions: dict[str, dict[str, np.ndarray]],
    term_index: GOTermIndex,
    propagator: GOPropagator,
) -> dict[str, dict[str, float]]:
    """
    Apply GO propagation to predictions.

    Args:
        predictions: Dictionary mapping protein IDs to ontology probabilities.
        term_index: GO term index.
        propagator: GO propagator.

    Returns:
        Dictionary mapping protein IDs to propagated term scores.
    """
    propagated: dict[str, dict[str, float]] = {}

    for protein_id, probabilities in predictions.items():
        # Convert to term scores
        term_scores = probabilities_to_term_scores(probabilities, term_index)

        # Apply propagation
        propagated_scores = propagator.propagate_scores(term_scores)

        propagated[protein_id] = propagated_scores

    return propagated


class InferencePipeline:
    """
    Complete inference pipeline for generating predictions.

    Handles loading checkpoint, running predictions, and applying GO propagation.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        term_index: GOTermIndex,
        propagator: GOPropagator | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the inference pipeline.

        Args:
            checkpoint_path: Path to the model checkpoint.
            term_index: GO term index.
            propagator: Optional GO propagator for score propagation.
            device: Device for inference.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(str(checkpoint_path), self.device)
        self.term_index = term_index
        self.propagator = propagator

    def predict(
        self,
        dataloader: DataLoader,
        apply_go_propagation: bool = True,
    ) -> dict[str, dict[str, float]]:
        """
        Run prediction pipeline.

        Args:
            dataloader: Dataloader for test data.
            apply_go_propagation: Whether to apply GO propagation.

        Returns:
            Dictionary mapping protein IDs to term scores.
        """
        # Run inference
        raw_predictions = run_inference(self.model, dataloader, self.device)

        # Convert to term scores
        term_predictions: dict[str, dict[str, float]] = {}
        for protein_id, probabilities in raw_predictions.items():
            term_predictions[protein_id] = probabilities_to_term_scores(
                probabilities, self.term_index
            )

        # Apply propagation if requested
        if apply_go_propagation and self.propagator is not None:
            for protein_id in term_predictions:
                term_predictions[protein_id] = self.propagator.propagate_scores(
                    term_predictions[protein_id]
                )

        return term_predictions


def run_full_inference(
    checkpoint_path: Path | str,
    test_protein_ids: list[str],
    test_embeddings: np.ndarray,
    term_index: GOTermIndex,
    propagator: GOPropagator | None = None,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> dict[str, dict[str, float]]:
    """
    Convenience function to run the full inference pipeline.

    Args:
        checkpoint_path: Path to the model checkpoint.
        test_protein_ids: List of test protein IDs.
        test_embeddings: Array of test embeddings.
        term_index: GO term index.
        propagator: Optional GO propagator.
        batch_size: Batch size for inference.
        device: Device for inference.

    Returns:
        Dictionary mapping protein IDs to term scores.
    """
    from .dataset import create_test_dataloader

    # Create dataloader
    dataloader = create_test_dataloader(
        test_protein_ids,
        test_embeddings,
        batch_size=batch_size,
    )

    # Create pipeline
    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path,
        term_index=term_index,
        propagator=propagator,
        device=device,
    )

    # Run prediction
    return pipeline.predict(dataloader, apply_go_propagation=propagator is not None)

