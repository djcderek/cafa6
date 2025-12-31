#!/usr/bin/env python3
"""Train the multi-head MLP model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --dummy --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from cafa6.config import Config, load_config
from cafa6.dataset import create_dataloaders, load_annotations
from cafa6.dummy import DummyDataGenerator, generate_dummy_embeddings
from cafa6.embeddings import load_embeddings
from cafa6.model import create_model
from cafa6.ontology import (
    GOTermIndex,
    build_term_index_from_terms,
    load_term_ontology_from_train_terms,
    load_term_ontology_mapping,
    load_training_terms,
)
from cafa6.propagation import load_go_edges_from_obo
from cafa6.trainer import set_seed, train_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the multi-head MLP model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy data for testing.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Name for the wandb run.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config from args
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr

    # Set seed for reproducibility
    set_seed(config.training.seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dummy:
        print("\nRunning in dummy mode...")

        # Generate or load dummy data
        data_dir = config.data.embeddings_dir.parent
        embeddings_path = data_dir / "embeddings" / "train.npz"

        if not embeddings_path.exists():
            print("Generating dummy data...")
            generator = DummyDataGenerator(
                output_dir=data_dir,
                seed=config.training.seed,
            )
            generator.generate()
            generator.save()
        else:
            print("Using existing dummy data...")

        # Load dummy embeddings
        data = np.load(embeddings_path, allow_pickle=True)
        train_protein_ids = data["protein_ids"].tolist()
        train_embeddings = data["embeddings"]

        # Load term-ontology mapping
        term_ontology_path = data_dir / "term_ontology.tsv"
        term_to_ontology = load_term_ontology_mapping(term_ontology_path)

        # Load annotations
        annotations_path = data_dir / "train_terms.tsv"
        train_annotations = load_annotations(annotations_path)

        # Build term index from annotations
        terms_in_training = set()
        for terms in train_annotations.values():
            terms_in_training.update(terms)

        term_index = build_term_index_from_terms(terms_in_training, term_to_ontology)

    else:
        print("\nLoading training data...")

        # Load embeddings
        embeddings_path = config.data.embeddings_dir / "train.parquet"
        if not embeddings_path.exists():
            embeddings_path = config.data.embeddings_dir / "train.npz"

        print(f"Loading embeddings from {embeddings_path}")
        train_protein_ids, train_embeddings = load_embeddings(embeddings_path)
        print(f"Loaded {len(train_protein_ids)} protein embeddings")

        # Load term-ontology mapping from OBO file or train_terms.tsv
        if config.data.go_obo.exists():
            print(f"Loading GO ontology from {config.data.go_obo}")
            _, term_to_ontology = load_go_edges_from_obo(config.data.go_obo)
            print(f"Loaded {len(term_to_ontology)} GO terms from ontology")
        else:
            # Fallback: extract from train_terms.tsv aspect column
            print(f"Loading term-ontology mapping from {config.data.train_terms}")
            term_to_ontology = load_term_ontology_from_train_terms(config.data.train_terms)
            print(f"Loaded {len(term_to_ontology)} term-ontology mappings")

        # Load annotations
        print(f"Loading annotations from {config.data.train_terms}")
        train_annotations = load_annotations(config.data.train_terms)
        print(f"Loaded annotations for {len(train_annotations)} proteins")

        # Build term index from training annotations
        terms_in_training = set()
        for terms in train_annotations.values():
            terms_in_training.update(terms)
        term_index = build_term_index_from_terms(terms_in_training, term_to_ontology)

    # Print term counts
    print(f"\nTerm counts:")
    print(f"  MF: {term_index.num_terms('MF')}")
    print(f"  BP: {term_index.num_terms('BP')}")
    print(f"  CC: {term_index.num_terms('CC')}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_protein_ids=train_protein_ids,
        train_embeddings=train_embeddings,
        train_annotations=train_annotations,
        term_index=term_index,
        batch_size=config.training.batch_size,
        val_split=config.training.val_split,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory and device.type == "cuda",
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        embedding_dim=config.model.embedding_dim,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        num_mf_terms=term_index.num_terms("MF"),
        num_bp_terms=term_index.num_terms("BP"),
        num_cc_terms=term_index.num_terms("CC"),
        device=device,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Build wandb config
    wandb_config = {
        "model_hidden_dims": config.model.hidden_dims,
        "model_dropout": config.model.dropout,
        "val_split": config.training.val_split,
        "num_mf_terms": term_index.num_terms("MF"),
        "num_bp_terms": term_index.num_terms("BP"),
        "num_cc_terms": term_index.num_terms("CC"),
        "num_train_proteins": len(train_protein_ids),
    }
    if args.wandb_run_name:
        wandb_config["run_name"] = args.wandb_run_name

    # Train
    print(f"\nStarting training for {config.training.epochs} epochs...")
    if args.wandb:
        print("Weights & Biases logging enabled")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        epochs=config.training.epochs,
        patience=config.training.patience,
        checkpoint_dir=config.training.checkpoint_dir,
        seed=config.training.seed,
        device=device,
        use_wandb=args.wandb,
        wandb_config=wandb_config,
    )

    # Save term index for inference
    term_index_path = config.training.checkpoint_dir / "term_index.npz"
    np.savez(
        term_index_path,
        mf_terms=list(term_index.get_all_terms("MF")),
        bp_terms=list(term_index.get_all_terms("BP")),
        cc_terms=list(term_index.get_all_terms("CC")),
    )
    print(f"\nSaved term index to {term_index_path}")

    print("\nTraining complete!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}")


if __name__ == "__main__":
    main()

