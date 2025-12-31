"""Configuration management with dataclasses and YAML loading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    """Configuration for data paths and processing."""

    train_fasta: Path = Path("data/cafa-6-protein-function-prediction/Train/train_sequences.fasta")
    test_fasta: Path = Path("data/cafa-6-protein-function-prediction/Test/testsuperset.fasta")
    train_terms: Path = Path("data/cafa-6-protein-function-prediction/Train/train_terms.tsv")
    go_obo: Path = Path("data/cafa-6-protein-function-prediction/Train/go-basic.obo")
    ia_weights: Path = Path("data/cafa-6-protein-function-prediction/IA.tsv")
    embeddings_dir: Path = Path("data/embeddings")
    ontology_terms_dir: Path = Path("data/ontology")

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        self.train_fasta = Path(self.train_fasta)
        self.test_fasta = Path(self.test_fasta)
        self.train_terms = Path(self.train_terms)
        self.go_obo = Path(self.go_obo)
        self.ia_weights = Path(self.ia_weights)
        self.embeddings_dir = Path(self.embeddings_dir)
        self.ontology_terms_dir = Path(self.ontology_terms_dir)


@dataclass
class ModelConfig:
    """Configuration for the MLP model architecture."""

    embedding_dim: int = 1024
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    num_mf_terms: int = 100  # Will be set dynamically based on data
    num_bp_terms: int = 100
    num_cc_terms: int = 100


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 10  # Early stopping patience
    val_split: float = 0.1
    checkpoint_dir: Path = Path("checkpoints")
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        self.checkpoint_dir = Path(self.checkpoint_dir)


@dataclass
class InferenceConfig:
    """Configuration for inference and submission."""

    checkpoint_path: Path = Path("checkpoints/best_model.pt")
    submission_path: Path = Path("submissions/submission.tsv")
    batch_size: int = 64
    max_terms_per_protein: int = 1500
    min_score: float = 1e-6  # Minimum score threshold (must be > 0)

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        self.checkpoint_path = Path(self.checkpoint_path)
        self.submission_path = Path(self.submission_path)


@dataclass
class EmbeddingConfig:
    """Configuration for ProtT5 embedding generation."""

    model_name: str = "Rostlab/prot_t5_xl_uniref50"
    batch_size: int = 8
    max_length: int = 1024  # Maximum sequence length
    cache_format: str = "parquet"  # "parquet" or "npz"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.cache_format not in ("parquet", "npz"):
            raise ValueError(f"cache_format must be 'parquet' or 'npz', got {self.cache_format}")


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


def load_config(config_path: Path | str) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object with all settings.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f) or {}

    # Build config from YAML
    data_cfg = DataConfig(**raw_config.get("data", {}))
    model_cfg = ModelConfig(**raw_config.get("model", {}))
    training_cfg = TrainingConfig(**raw_config.get("training", {}))
    inference_cfg = InferenceConfig(**raw_config.get("inference", {}))
    embedding_cfg = EmbeddingConfig(**raw_config.get("embedding", {}))

    return Config(
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        inference=inference_cfg,
        embedding=embedding_cfg,
    )


def save_config(config: Config, config_path: Path | str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Config object to save.
        config_path: Path to save the YAML file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    config_dict = {
        "data": {
            "train_fasta": str(config.data.train_fasta),
            "test_fasta": str(config.data.test_fasta),
            "train_terms": str(config.data.train_terms),
            "go_obo": str(config.data.go_obo),
            "ia_weights": str(config.data.ia_weights),
            "embeddings_dir": str(config.data.embeddings_dir),
            "ontology_terms_dir": str(config.data.ontology_terms_dir),
        },
        "model": {
            "embedding_dim": config.model.embedding_dim,
            "hidden_dims": config.model.hidden_dims,
            "dropout": config.model.dropout,
            "num_mf_terms": config.model.num_mf_terms,
            "num_bp_terms": config.model.num_bp_terms,
            "num_cc_terms": config.model.num_cc_terms,
        },
        "training": {
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "epochs": config.training.epochs,
            "patience": config.training.patience,
            "val_split": config.training.val_split,
            "checkpoint_dir": str(config.training.checkpoint_dir),
            "seed": config.training.seed,
            "num_workers": config.training.num_workers,
            "pin_memory": config.training.pin_memory,
        },
        "inference": {
            "checkpoint_path": str(config.inference.checkpoint_path),
            "submission_path": str(config.inference.submission_path),
            "batch_size": config.inference.batch_size,
            "max_terms_per_protein": config.inference.max_terms_per_protein,
            "min_score": config.inference.min_score,
        },
        "embedding": {
            "model_name": config.embedding.model_name,
            "batch_size": config.embedding.batch_size,
            "max_length": config.embedding.max_length,
            "cache_format": config.embedding.cache_format,
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

