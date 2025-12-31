"""Multi-head MLP model for GO term prediction.

Architecture:
- Shared trunk: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout
- Three output heads (MF, BP, CC): Linear -> logits
"""

from typing import Any

import torch
import torch.nn as nn


class MultiHeadMLP(nn.Module):
    """
    MLP with shared trunk and separate heads for MF, BP, CC ontologies.

    The model takes protein embeddings and outputs logits for each ontology.
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        num_mf_terms: int = 100,
        num_bp_terms: int = 100,
        num_cc_terms: int = 100,
    ) -> None:
        """
        Initialize the multi-head MLP.

        Args:
            embedding_dim: Dimension of input embeddings (ProtT5 = 1024).
            hidden_dims: List of hidden layer dimensions for the trunk.
            dropout: Dropout probability.
            num_mf_terms: Number of MF (Molecular Function) terms.
            num_bp_terms: Number of BP (Biological Process) terms.
            num_cc_terms: Number of CC (Cellular Component) terms.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Build shared trunk
        trunk_layers: list[nn.Module] = []
        in_dim = embedding_dim

        for hidden_dim in hidden_dims:
            trunk_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.trunk = nn.Sequential(*trunk_layers)

        # Output heads for each ontology
        self.head_mf = nn.Linear(in_dim, num_mf_terms)
        self.head_bp = nn.Linear(in_dim, num_bp_terms)
        self.head_cc = nn.Linear(in_dim, num_cc_terms)

        # Store dimensions and hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.num_mf_terms = num_mf_terms
        self.num_bp_terms = num_bp_terms
        self.num_cc_terms = num_cc_terms

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim).

        Returns:
            Dictionary with logits for each ontology:
                - logits_mf: (batch_size, num_mf_terms)
                - logits_bp: (batch_size, num_bp_terms)
                - logits_cc: (batch_size, num_cc_terms)
        """
        # Shared trunk
        features = self.trunk(embeddings)

        # Ontology-specific heads
        logits_mf = self.head_mf(features)
        logits_bp = self.head_bp(features)
        logits_cc = self.head_cc(features)

        return {
            "logits_mf": logits_mf,
            "logits_bp": logits_bp,
            "logits_cc": logits_cc,
        }

    def predict_proba(
        self,
        embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict probabilities for each GO term.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim).

        Returns:
            Dictionary with probabilities for each ontology:
                - proba_mf: (batch_size, num_mf_terms)
                - proba_bp: (batch_size, num_bp_terms)
                - proba_cc: (batch_size, num_cc_terms)
        """
        logits = self.forward(embeddings)

        return {
            "proba_mf": torch.sigmoid(logits["logits_mf"]),
            "proba_bp": torch.sigmoid(logits["logits_bp"]),
            "proba_cc": torch.sigmoid(logits["logits_cc"]),
        }

    def get_config(self) -> dict[str, Any]:
        """Get model configuration for saving."""
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "num_mf_terms": self.num_mf_terms,
            "num_bp_terms": self.num_bp_terms,
            "num_cc_terms": self.num_cc_terms,
        }


def create_model(
    embedding_dim: int = 1024,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.1,
    num_mf_terms: int = 100,
    num_bp_terms: int = 100,
    num_cc_terms: int = 100,
    device: torch.device | None = None,
) -> MultiHeadMLP:
    """
    Create and initialize the model.

    Args:
        embedding_dim: Dimension of input embeddings.
        hidden_dims: Hidden layer dimensions.
        dropout: Dropout probability.
        num_mf_terms: Number of MF terms.
        num_bp_terms: Number of BP terms.
        num_cc_terms: Number of CC terms.
        device: Device to place model on.

    Returns:
        Initialized MultiHeadMLP model.
    """
    model = MultiHeadMLP(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_mf_terms=num_mf_terms,
        num_bp_terms=num_bp_terms,
        num_cc_terms=num_cc_terms,
    )

    if device is not None:
        model = model.to(device)

    return model


def load_model(
    checkpoint_path: str,
    device: torch.device | None = None,
) -> MultiHeadMLP:
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to place model on.

    Returns:
        Loaded model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model from saved config
    config = checkpoint["model_config"]
    model = MultiHeadMLP(
        embedding_dim=config["embedding_dim"],
        hidden_dims=config["hidden_dims"],
        dropout=config.get("dropout", 0.1),  # Default for backwards compatibility
        num_mf_terms=config["num_mf_terms"],
        num_bp_terms=config["num_bp_terms"],
        num_cc_terms=config["num_cc_terms"],
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model

