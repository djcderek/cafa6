"""Dummy data generation for testing the pipeline.

Generates synthetic proteins, GO terms, and annotations so the full
pipeline can be tested without real data.
"""

import random
import string
from pathlib import Path

import numpy as np

from .fasta import write_fasta
from .ontology import GOTermIndex, OntologyType


# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def generate_random_sequence(length: int) -> str:
    """
    Generate a random protein sequence.

    Args:
        length: Sequence length.

    Returns:
        Random amino acid sequence.
    """
    return "".join(random.choices(AMINO_ACIDS, k=length))


def generate_dummy_proteins(
    num_proteins: int = 100,
    min_length: int = 50,
    max_length: int = 500,
    seed: int = 42,
) -> dict[str, str]:
    """
    Generate dummy protein sequences.

    Args:
        num_proteins: Number of proteins to generate.
        min_length: Minimum sequence length.
        max_length: Maximum sequence length.
        seed: Random seed.

    Returns:
        Dictionary mapping protein IDs to sequences.
    """
    random.seed(seed)

    proteins = {}
    for i in range(num_proteins):
        protein_id = f"DUMMY_{i:05d}"
        length = random.randint(min_length, max_length)
        sequence = generate_random_sequence(length)
        proteins[protein_id] = sequence

    return proteins


def generate_dummy_go_terms(
    num_mf: int = 20,
    num_bp: int = 30,
    num_cc: int = 15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], dict[str, OntologyType]]:
    """
    Generate dummy GO terms for each ontology.

    Args:
        num_mf: Number of MF terms.
        num_bp: Number of BP terms.
        num_cc: Number of CC terms.
        seed: Random seed.

    Returns:
        Tuple of (mf_terms, bp_terms, cc_terms, term_to_ontology).
    """
    random.seed(seed)

    mf_terms = [f"GO:{1000000 + i}" for i in range(num_mf)]
    bp_terms = [f"GO:{2000000 + i}" for i in range(num_bp)]
    cc_terms = [f"GO:{3000000 + i}" for i in range(num_cc)]

    term_to_ontology: dict[str, OntologyType] = {}
    for term in mf_terms:
        term_to_ontology[term] = "MF"
    for term in bp_terms:
        term_to_ontology[term] = "BP"
    for term in cc_terms:
        term_to_ontology[term] = "CC"

    return mf_terms, bp_terms, cc_terms, term_to_ontology


def generate_dummy_go_edges(
    mf_terms: list[str],
    bp_terms: list[str],
    cc_terms: list[str],
    seed: int = 42,
) -> dict[str, set[str]]:
    """
    Generate dummy GO edges (child -> parents).

    Creates a simple hierarchy within each ontology.

    Args:
        mf_terms: List of MF terms.
        bp_terms: List of BP terms.
        cc_terms: List of CC terms.
        seed: Random seed.

    Returns:
        Dictionary mapping child terms to parent terms.
    """
    random.seed(seed)

    edges: dict[str, set[str]] = {}

    def add_hierarchy(terms: list[str]) -> None:
        """Add hierarchical edges within an ontology."""
        if len(terms) < 2:
            return

        # Create a simple tree structure
        # First term is root, others have random parents
        for i, term in enumerate(terms[1:], 1):
            # Each term has 1-2 parents from earlier terms
            num_parents = random.randint(1, min(2, i))
            parents = random.sample(terms[:i], num_parents)
            edges[term] = set(parents)

    add_hierarchy(mf_terms)
    add_hierarchy(bp_terms)
    add_hierarchy(cc_terms)

    return edges


def generate_dummy_annotations(
    protein_ids: list[str],
    all_terms: list[str],
    min_terms: int = 1,
    max_terms: int = 10,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Generate dummy protein-term annotations.

    Args:
        protein_ids: List of protein IDs.
        all_terms: List of all GO terms.
        min_terms: Minimum terms per protein.
        max_terms: Maximum terms per protein.
        seed: Random seed.

    Returns:
        Dictionary mapping protein IDs to GO terms.
    """
    random.seed(seed)

    annotations = {}
    for protein_id in protein_ids:
        num_terms = random.randint(min_terms, min(max_terms, len(all_terms)))
        terms = random.sample(all_terms, num_terms)
        annotations[protein_id] = terms

    return annotations


def generate_dummy_embeddings(
    protein_ids: list[str],
    embedding_dim: int = 1024,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate dummy protein embeddings.

    Args:
        protein_ids: List of protein IDs.
        embedding_dim: Embedding dimension.
        seed: Random seed.

    Returns:
        Array of shape (num_proteins, embedding_dim).
    """
    np.random.seed(seed)
    return np.random.randn(len(protein_ids), embedding_dim).astype(np.float32)


class DummyDataGenerator:
    """
    Generates a complete dummy dataset for testing.

    Creates all necessary files for running the full pipeline:
    - FASTA files for train and test
    - Term annotations
    - GO edges
    - Valid terms
    - Cached embeddings
    """

    def __init__(
        self,
        output_dir: Path | str = "data",
        num_train: int = 80,
        num_test: int = 20,
        num_mf_terms: int = 20,
        num_bp_terms: int = 30,
        num_cc_terms: int = 15,
        seed: int = 42,
    ) -> None:
        """
        Initialize the generator.

        Args:
            output_dir: Directory to save generated data.
            num_train: Number of training proteins.
            num_test: Number of test proteins.
            num_mf_terms: Number of MF terms.
            num_bp_terms: Number of BP terms.
            num_cc_terms: Number of CC terms.
            seed: Random seed.
        """
        self.output_dir = Path(output_dir)
        self.num_train = num_train
        self.num_test = num_test
        self.num_mf_terms = num_mf_terms
        self.num_bp_terms = num_bp_terms
        self.num_cc_terms = num_cc_terms
        self.seed = seed

        # Generated data
        self.train_proteins: dict[str, str] = {}
        self.test_proteins: dict[str, str] = {}
        self.mf_terms: list[str] = []
        self.bp_terms: list[str] = []
        self.cc_terms: list[str] = []
        self.term_to_ontology: dict[str, OntologyType] = {}
        self.go_edges: dict[str, set[str]] = {}
        self.train_annotations: dict[str, list[str]] = {}

    def generate(self) -> None:
        """Generate all dummy data."""
        # Generate proteins
        all_proteins = generate_dummy_proteins(
            self.num_train + self.num_test,
            seed=self.seed,
        )
        protein_ids = list(all_proteins.keys())
        self.train_proteins = {
            pid: all_proteins[pid] for pid in protein_ids[: self.num_train]
        }
        self.test_proteins = {
            pid: all_proteins[pid] for pid in protein_ids[self.num_train :]
        }

        # Generate GO terms
        self.mf_terms, self.bp_terms, self.cc_terms, self.term_to_ontology = (
            generate_dummy_go_terms(
                self.num_mf_terms,
                self.num_bp_terms,
                self.num_cc_terms,
                seed=self.seed,
            )
        )

        # Generate GO edges
        self.go_edges = generate_dummy_go_edges(
            self.mf_terms,
            self.bp_terms,
            self.cc_terms,
            seed=self.seed,
        )

        # Generate annotations for training proteins
        all_terms = self.mf_terms + self.bp_terms + self.cc_terms
        self.train_annotations = generate_dummy_annotations(
            list(self.train_proteins.keys()),
            all_terms,
            seed=self.seed,
        )

    def save(self) -> None:
        """Save generated data to files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save FASTA files
        write_fasta(self.train_proteins, self.output_dir / "train.fasta")
        write_fasta(self.test_proteins, self.output_dir / "test.fasta")

        # Save term-ontology mapping
        with open(self.output_dir / "term_ontology.tsv", "w") as f:
            for term, ontology in sorted(self.term_to_ontology.items()):
                f.write(f"{term}\t{ontology}\n")

        # Save GO edges
        with open(self.output_dir / "go_edges.tsv", "w") as f:
            for child, parents in sorted(self.go_edges.items()):
                for parent in sorted(parents):
                    f.write(f"{child}\t{parent}\n")

        # Save train annotations
        with open(self.output_dir / "train_terms.tsv", "w") as f:
            for protein_id, terms in sorted(self.train_annotations.items()):
                for term in sorted(terms):
                    f.write(f"{protein_id}\t{term}\n")

        # Save valid terms (all terms)
        all_terms = self.mf_terms + self.bp_terms + self.cc_terms
        with open(self.output_dir / "valid_terms.txt", "w") as f:
            for term in sorted(all_terms):
                f.write(f"{term}\n")

        # Save dummy embeddings
        embeddings_dir = self.output_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)

        train_ids = list(self.train_proteins.keys())
        train_embeddings = generate_dummy_embeddings(train_ids, seed=self.seed)
        np.savez(
            embeddings_dir / "train.npz",
            protein_ids=np.array(train_ids),
            embeddings=train_embeddings,
        )

        test_ids = list(self.test_proteins.keys())
        test_embeddings = generate_dummy_embeddings(test_ids, seed=self.seed + 1)
        np.savez(
            embeddings_dir / "test.npz",
            protein_ids=np.array(test_ids),
            embeddings=test_embeddings,
        )

        print(f"Generated dummy data in {self.output_dir}")
        print(f"  Train proteins: {len(self.train_proteins)}")
        print(f"  Test proteins: {len(self.test_proteins)}")
        print(f"  GO terms: MF={len(self.mf_terms)}, BP={len(self.bp_terms)}, CC={len(self.cc_terms)}")

    def get_term_index(self) -> GOTermIndex:
        """
        Build a GOTermIndex from the generated terms.

        Returns:
            GOTermIndex with all generated terms.
        """
        index = GOTermIndex()

        for term in self.mf_terms:
            index.add_term(term, "MF")
        for term in self.bp_terms:
            index.add_term(term, "BP")
        for term in self.cc_terms:
            index.add_term(term, "CC")

        return index


def create_dummy_data(
    output_dir: Path | str = "data",
    num_train: int = 80,
    num_test: int = 20,
    seed: int = 42,
) -> DummyDataGenerator:
    """
    Convenience function to create dummy data.

    Args:
        output_dir: Directory to save data.
        num_train: Number of training proteins.
        num_test: Number of test proteins.
        seed: Random seed.

    Returns:
        DummyDataGenerator with generated data.
    """
    generator = DummyDataGenerator(
        output_dir=output_dir,
        num_train=num_train,
        num_test=num_test,
        seed=seed,
    )
    generator.generate()
    generator.save()
    return generator

