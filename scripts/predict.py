#!/usr/bin/env python3
"""Run inference and generate Kaggle submission.

Usage:
    python scripts/predict.py --config configs/default.yaml
    python scripts/predict.py --dummy --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from cafa6.config import Config, load_config
from cafa6.dataset import create_test_dataloader
from cafa6.embeddings import load_embeddings
from cafa6.inference import InferencePipeline
from cafa6.ontology import GOTermIndex, load_valid_terms, load_valid_terms_from_ia
from cafa6.propagation import GOPropagator, load_go_edges_from_obo
from cafa6.submission import write_submission


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference and generate Kaggle submission."
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
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path from config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path from config.",
    )
    parser.add_argument(
        "--no-propagation",
        action="store_true",
        help="Skip GO propagation.",
    )
    return parser.parse_args()


def load_term_index(checkpoint_dir: Path) -> GOTermIndex:
    """
    Load term index from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory.

    Returns:
        GOTermIndex.
    """
    term_index_path = checkpoint_dir / "term_index.npz"
    data = np.load(term_index_path, allow_pickle=True)

    term_index = GOTermIndex()

    for term in data["mf_terms"]:
        term_index.add_term(str(term), "MF")
    for term in data["bp_terms"]:
        term_index.add_term(str(term), "BP")
    for term in data["cc_terms"]:
        term_index.add_term(str(term), "CC")

    return term_index


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override from args
    if args.checkpoint is not None:
        config.inference.checkpoint_path = Path(args.checkpoint)
    if args.output is not None:
        config.inference.submission_path = Path(args.output)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dummy:
        print("\nRunning in dummy mode...")
        data_dir = config.data.embeddings_dir.parent
    else:
        data_dir = config.data.embeddings_dir.parent

    # Load test embeddings
    if args.dummy:
        embeddings_path = data_dir / "embeddings" / "test.npz"
    else:
        embeddings_path = config.data.embeddings_dir / "test.parquet"
        if not embeddings_path.exists():
            embeddings_path = config.data.embeddings_dir / "test.npz"

    print(f"Loading test embeddings from {embeddings_path}")
    test_protein_ids, test_embeddings = load_embeddings(embeddings_path)
    print(f"Loaded {len(test_protein_ids)} test proteins")

    # Load term index
    checkpoint_dir = config.inference.checkpoint_path.parent
    print(f"Loading term index from {checkpoint_dir}")
    term_index = load_term_index(checkpoint_dir)
    print(f"Term counts: MF={term_index.num_terms('MF')}, BP={term_index.num_terms('BP')}, CC={term_index.num_terms('CC')}")

    # Load GO propagator
    propagator = None
    if not args.no_propagation:
        if args.dummy:
            go_edges_path = data_dir / "go_edges.tsv"
            if go_edges_path.exists():
                print(f"Loading GO edges from {go_edges_path}")
                propagator = GOPropagator.from_file(go_edges_path)
                print(f"Loaded {len(propagator.child_to_parents)} edge relationships")
        else:
            go_obo_path = config.data.go_obo
            if go_obo_path.exists():
                print(f"Loading GO ontology from {go_obo_path}")
                propagator = GOPropagator.from_obo(go_obo_path)
                print(f"Loaded {len(propagator.child_to_parents)} edge relationships")
            else:
                print("Warning: GO ontology file not found, skipping propagation")

    # Load valid terms
    if args.dummy:
        valid_terms_path = data_dir / "valid_terms.txt"
        print(f"Loading valid terms from {valid_terms_path}")
        valid_terms = load_valid_terms(valid_terms_path)
    else:
        ia_path = config.data.ia_weights
        print(f"Loading valid terms from {ia_path}")
        valid_terms = load_valid_terms_from_ia(ia_path)
    print(f"Loaded {len(valid_terms)} valid terms")

    # Create test dataloader
    print("\nCreating test dataloader...")
    test_loader = create_test_dataloader(
        test_protein_ids=test_protein_ids,
        test_embeddings=test_embeddings,
        batch_size=config.inference.batch_size,
    )

    # Create inference pipeline
    print(f"\nLoading model from {config.inference.checkpoint_path}")
    pipeline = InferencePipeline(
        checkpoint_path=config.inference.checkpoint_path,
        term_index=term_index,
        propagator=propagator,
        device=device,
    )

    # Run inference
    print("\nRunning inference...")
    predictions = pipeline.predict(
        test_loader,
        apply_go_propagation=not args.no_propagation,
    )
    print(f"Generated predictions for {len(predictions)} proteins")

    # Write submission
    print(f"\nWriting submission to {config.inference.submission_path}")
    num_predictions = write_submission(
        predictions=predictions,
        valid_terms=valid_terms,
        output_path=config.inference.submission_path,
        max_terms_per_protein=config.inference.max_terms_per_protein,
        min_score=config.inference.min_score,
    )
    print(f"Wrote {num_predictions} predictions")

    # Summary stats
    terms_per_protein = [len(terms) for terms in predictions.values()]
    print(f"\nSubmission summary:")
    print(f"  Proteins: {len(predictions)}")
    print(f"  Total predictions: {num_predictions}")
    print(f"  Avg terms/protein: {np.mean(terms_per_protein):.1f}")
    print(f"  Max terms/protein: {max(terms_per_protein)}")

    print("\nInference complete!")


if __name__ == "__main__":
    main()

