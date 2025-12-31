#!/usr/bin/env python3
"""Generate and cache ProtT5 embeddings for protein sequences.

Usage:
    python scripts/generate_embeddings.py --config configs/default.yaml
    python scripts/generate_embeddings.py --dummy --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cafa6.config import Config, load_config
from cafa6.dummy import create_dummy_data, generate_dummy_embeddings
from cafa6.embeddings import EmbeddingGenerator, cache_embeddings_parquet, cache_embeddings_npz
from cafa6.fasta import parse_fasta


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ProtT5 embeddings for protein sequences."
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
        help="Generate dummy data instead of loading real data.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only generate embeddings for training data.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only generate embeddings for test data.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    if args.dummy:
        print("Running in dummy mode - generating synthetic data...")
        generator = create_dummy_data(
            output_dir=config.data.embeddings_dir.parent,
            num_train=80,
            num_test=20,
            seed=config.training.seed,
        )
        print("Dummy data generated with pre-computed embeddings.")
        return

    # Create embedding generator
    print(f"Loading ProtT5 model: {config.embedding.model_name}")
    emb_generator = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        batch_size=config.embedding.batch_size,
        max_length=config.embedding.max_length,
        cache_format=config.embedding.cache_format,
    )
    emb_generator.load_model()

    # Create output directory
    config.data.embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Generate train embeddings
    if not args.test_only:
        print(f"\nLoading training proteins from {config.data.train_fasta}")
        train_proteins = parse_fasta(config.data.train_fasta)
        print(f"Loaded {len(train_proteins)} training proteins")

        output_ext = "parquet" if config.embedding.cache_format == "parquet" else "npz"
        train_output = config.data.embeddings_dir / f"train.{output_ext}"

        print(f"Generating embeddings -> {train_output}")
        emb_generator.generate_and_cache(
            train_proteins,
            train_output,
            show_progress=True,
        )
        print(f"Saved training embeddings to {train_output}")

    # Generate test embeddings
    if not args.train_only:
        print(f"\nLoading test proteins from {config.data.test_fasta}")
        test_proteins = parse_fasta(config.data.test_fasta)
        print(f"Loaded {len(test_proteins)} test proteins")

        output_ext = "parquet" if config.embedding.cache_format == "parquet" else "npz"
        test_output = config.data.embeddings_dir / f"test.{output_ext}"

        print(f"Generating embeddings -> {test_output}")
        emb_generator.generate_and_cache(
            test_proteins,
            test_output,
            show_progress=True,
        )
        print(f"Saved test embeddings to {test_output}")

    print("\nEmbedding generation complete!")


if __name__ == "__main__":
    main()

