"""PyTorch Dataset and DataLoader for embeddings and multi-label targets."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .ontology import GOTermIndex, OntologyType


class ProteinDataset(Dataset):
    """
    Dataset for protein embeddings with multi-label GO term targets.

    Supports separate targets for MF, BP, and CC ontologies.
    """

    def __init__(
        self,
        protein_ids: list[str],
        embeddings: np.ndarray,
        annotations: dict[str, list[str]] | None = None,
        term_index: GOTermIndex | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            protein_ids: List of protein IDs.
            embeddings: Array of embeddings of shape (num_proteins, embedding_dim).
            annotations: Optional dict mapping protein_id to list of GO terms.
            term_index: Optional GOTermIndex for converting terms to indices.
        """
        self.protein_ids = protein_ids
        self.embeddings = embeddings
        self.annotations = annotations
        self.term_index = term_index

        # Build protein_id to index mapping
        self.id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

        # Pre-compute targets if annotations provided
        self.targets: dict[str, dict[OntologyType, np.ndarray]] | None = None
        if annotations is not None and term_index is not None:
            self._build_targets()

    def _build_targets(self) -> None:
        """Build sparse target arrays for each protein."""
        assert self.annotations is not None
        assert self.term_index is not None

        self.targets = {}

        for protein_id in self.protein_ids:
            protein_terms = self.annotations.get(protein_id, [])

            targets: dict[OntologyType, np.ndarray] = {}
            for ontology in ["MF", "BP", "CC"]:
                ont: OntologyType = ontology  # type: ignore
                num_terms = self.term_index.num_terms(ont)
                target = np.zeros(num_terms, dtype=np.float32)

                for term in protein_terms:
                    idx = self.term_index.get_idx(term, ont)
                    if idx is not None:
                        target[idx] = 1.0

                targets[ont] = target

            self.targets[protein_id] = targets

    def __len__(self) -> int:
        """Return the number of proteins."""
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - protein_id: str
                - embedding: Tensor of shape (embedding_dim,)
                - targets_mf: Tensor of shape (num_mf_terms,) [if annotations provided]
                - targets_bp: Tensor of shape (num_bp_terms,)
                - targets_cc: Tensor of shape (num_cc_terms,)
        """
        protein_id = self.protein_ids[idx]
        embedding = torch.from_numpy(self.embeddings[idx]).float()

        sample = {
            "protein_id": protein_id,
            "embedding": embedding,
        }

        if self.targets is not None:
            sample["targets_mf"] = torch.from_numpy(self.targets[protein_id]["MF"])
            sample["targets_bp"] = torch.from_numpy(self.targets[protein_id]["BP"])
            sample["targets_cc"] = torch.from_numpy(self.targets[protein_id]["CC"])

        return sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for batching samples.

    Args:
        batch: List of samples from the dataset.

    Returns:
        Batched dictionary.
    """
    result: dict[str, Any] = {
        "protein_id": [sample["protein_id"] for sample in batch],
        "embedding": torch.stack([sample["embedding"] for sample in batch]),
    }

    # Stack targets if present
    if "targets_mf" in batch[0]:
        result["targets_mf"] = torch.stack([sample["targets_mf"] for sample in batch])
        result["targets_bp"] = torch.stack([sample["targets_bp"] for sample in batch])
        result["targets_cc"] = torch.stack([sample["targets_cc"] for sample in batch])

    return result


def create_dataloaders(
    train_protein_ids: list[str],
    train_embeddings: np.ndarray,
    train_annotations: dict[str, list[str]],
    term_index: GOTermIndex,
    batch_size: int = 32,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_protein_ids: List of training protein IDs.
        train_embeddings: Array of training embeddings.
        train_annotations: Annotations for training proteins.
        term_index: GO term index.
        batch_size: Batch size.
        val_split: Fraction of data for validation.
        seed: Random seed for splitting.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Shuffle and split
    n = len(train_protein_ids)
    indices = np.random.permutation(n)
    val_size = int(n * val_split)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Split data
    train_ids = [train_protein_ids[i] for i in train_indices]
    train_embs = train_embeddings[train_indices]
    val_ids = [train_protein_ids[i] for i in val_indices]
    val_embs = train_embeddings[val_indices]

    # Create datasets
    train_dataset = ProteinDataset(
        protein_ids=train_ids,
        embeddings=train_embs,
        annotations=train_annotations,
        term_index=term_index,
    )

    val_dataset = ProteinDataset(
        protein_ids=val_ids,
        embeddings=val_embs,
        annotations=train_annotations,
        term_index=term_index,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def create_test_dataloader(
    test_protein_ids: list[str],
    test_embeddings: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a dataloader for test/inference data.

    Args:
        test_protein_ids: List of test protein IDs.
        test_embeddings: Array of test embeddings.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory.

    Returns:
        Test dataloader.
    """
    test_dataset = ProteinDataset(
        protein_ids=test_protein_ids,
        embeddings=test_embeddings,
        annotations=None,
        term_index=None,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_annotations(annotations_path: Path | str) -> dict[str, list[str]]:
    """
    Load protein annotations from a TSV file.

    Supports two formats:
    - CAFA-6 format: EntryID<TAB>term<TAB>aspect (with header row)
    - Legacy format: protein_id<TAB>GO_term (no header)

    Args:
        annotations_path: Path to the annotations file.

    Returns:
        Dictionary mapping protein IDs to lists of GO terms.
    """
    annotations_path = Path(annotations_path)
    annotations: dict[str, list[str]] = {}
    first_line = True

    with open(annotations_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            # Skip header row (check if first column looks like a header)
            if first_line:
                first_line = False
                # Check if this looks like a header (EntryID or protein_id, not a protein accession)
                if parts[0].lower() in ("entryid", "protein_id", "id"):
                    continue

            if len(parts) >= 2:
                protein_id, term = parts[0], parts[1]
                if protein_id not in annotations:
                    annotations[protein_id] = []
                annotations[protein_id].append(term)

    return annotations

