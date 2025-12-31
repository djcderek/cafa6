"""GO term indexing and ontology utilities for MF, BP, CC term sets."""

from pathlib import Path
from typing import Literal

# Ontology type alias
OntologyType = Literal["MF", "BP", "CC"]

# GO term prefixes for each ontology
# These are used to filter terms by ontology when reading from a mixed file
ONTOLOGY_ROOTS = {
    "MF": "GO:0003674",  # molecular_function
    "BP": "GO:0008150",  # biological_process
    "CC": "GO:0005575",  # cellular_component
}


class GOTermIndex:
    """
    Index for GO terms, mapping between term IDs and integer indices.

    Supports separate indexing for MF, BP, and CC ontologies.
    """

    def __init__(self) -> None:
        """Initialize empty term indices for each ontology."""
        self._term_to_idx: dict[OntologyType, dict[str, int]] = {
            "MF": {},
            "BP": {},
            "CC": {},
        }
        self._idx_to_term: dict[OntologyType, dict[int, str]] = {
            "MF": {},
            "BP": {},
            "CC": {},
        }
        self._ontology_for_term: dict[str, OntologyType] = {}

    def add_term(self, term: str, ontology: OntologyType) -> int:
        """
        Add a term to the index.

        Args:
            term: GO term ID (e.g., "GO:0003674").
            ontology: Ontology type ("MF", "BP", or "CC").

        Returns:
            Index assigned to the term.
        """
        if term in self._term_to_idx[ontology]:
            return self._term_to_idx[ontology][term]

        idx = len(self._term_to_idx[ontology])
        self._term_to_idx[ontology][term] = idx
        self._idx_to_term[ontology][idx] = term
        self._ontology_for_term[term] = ontology
        return idx

    def get_idx(self, term: str, ontology: OntologyType) -> int | None:
        """
        Get the index for a term.

        Args:
            term: GO term ID.
            ontology: Ontology type.

        Returns:
            Index of the term, or None if not found.
        """
        return self._term_to_idx[ontology].get(term)

    def get_term(self, idx: int, ontology: OntologyType) -> str | None:
        """
        Get the term for an index.

        Args:
            idx: Term index.
            ontology: Ontology type.

        Returns:
            GO term ID, or None if not found.
        """
        return self._idx_to_term[ontology].get(idx)

    def get_ontology(self, term: str) -> OntologyType | None:
        """
        Get the ontology type for a term.

        Args:
            term: GO term ID.

        Returns:
            Ontology type, or None if term not found.
        """
        return self._ontology_for_term.get(term)

    def num_terms(self, ontology: OntologyType) -> int:
        """Get the number of terms for an ontology."""
        return len(self._term_to_idx[ontology])

    def get_all_terms(self, ontology: OntologyType) -> set[str]:
        """Get all terms for an ontology."""
        return set(self._term_to_idx[ontology].keys())

    def get_all_indices(self, ontology: OntologyType) -> list[int]:
        """Get all indices for an ontology."""
        return list(range(len(self._term_to_idx[ontology])))


def load_term_ontology_mapping(
    mapping_path: Path | str,
) -> dict[str, OntologyType]:
    """
    Load a mapping from GO terms to their ontology type.

    Expects a TSV file with columns: term, ontology (MF/BP/CC).

    Args:
        mapping_path: Path to the mapping file.

    Returns:
        Dictionary mapping GO terms to ontology types.
    """
    mapping_path = Path(mapping_path)
    term_to_ontology: dict[str, OntologyType] = {}

    with open(mapping_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                term, ontology = parts[0], parts[1]
                if ontology in ("MF", "BP", "CC"):
                    term_to_ontology[term] = ontology  # type: ignore

    return term_to_ontology


# Mapping from CAFA-6 aspect codes to ontology types
ASPECT_TO_ONTOLOGY: dict[str, OntologyType] = {
    "F": "MF",  # molecular Function
    "P": "BP",  # biological Process
    "C": "CC",  # cellular Component
}


def load_term_ontology_from_train_terms(
    train_terms_path: Path | str,
) -> dict[str, OntologyType]:
    """
    Extract term-to-ontology mapping from CAFA-6 train_terms.tsv file.

    Expects format: EntryID<TAB>term<TAB>aspect
    Where aspect is F (MF), P (BP), or C (CC).

    Args:
        train_terms_path: Path to the train_terms.tsv file.

    Returns:
        Dictionary mapping GO terms to ontology types.
    """
    train_terms_path = Path(train_terms_path)
    term_to_ontology: dict[str, OntologyType] = {}
    first_line = True

    with open(train_terms_path, "r") as f:
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

            if len(parts) >= 3:
                term, aspect = parts[1], parts[2]
                if aspect in ASPECT_TO_ONTOLOGY and term not in term_to_ontology:
                    term_to_ontology[term] = ASPECT_TO_ONTOLOGY[aspect]

    return term_to_ontology


def load_training_terms(
    terms_path: Path | str,
    term_to_ontology: dict[str, OntologyType],
) -> GOTermIndex:
    """
    Load training terms and build a GOTermIndex.

    Supports two formats:
    - CAFA-6 format: EntryID<TAB>term<TAB>aspect (with header row)
    - Legacy format: protein_id<TAB>GO_term (no header)

    Args:
        terms_path: Path to the training terms file.
        term_to_ontology: Mapping from GO terms to ontology types.

    Returns:
        GOTermIndex with all training terms indexed.
    """
    terms_path = Path(terms_path)
    index = GOTermIndex()
    seen_terms: set[str] = set()
    first_line = True

    with open(terms_path, "r") as f:
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
                term = parts[1]
                if term not in seen_terms and term in term_to_ontology:
                    ontology = term_to_ontology[term]
                    index.add_term(term, ontology)
                    seen_terms.add(term)

    return index


def load_valid_terms(valid_terms_path: Path | str) -> set[str]:
    """
    Load the set of valid GO terms for submission.

    Expects a text file with one GO term per line.

    Args:
        valid_terms_path: Path to the valid terms file.

    Returns:
        Set of valid GO term IDs.
    """
    valid_terms_path = Path(valid_terms_path)
    valid_terms: set[str] = set()

    with open(valid_terms_path, "r") as f:
        for line in f:
            term = line.strip()
            if term and term.startswith("GO:"):
                valid_terms.add(term)

    return valid_terms


def load_ia_weights(ia_path: Path | str) -> tuple[set[str], dict[str, float]]:
    """
    Load Information Accretion weights from IA.tsv file.

    The IA.tsv file contains GO terms and their IC (Information Content) weights
    used for scoring in the CAFA competition.

    Format: GO_term<TAB>IA_weight (no header)

    Args:
        ia_path: Path to the IA.tsv file.

    Returns:
        Tuple of:
        - valid_terms: Set of all valid GO terms (for submission filtering).
        - ia_weights: Dictionary mapping GO terms to their IA weights.
    """
    ia_path = Path(ia_path)
    valid_terms: set[str] = set()
    ia_weights: dict[str, float] = {}

    with open(ia_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                term = parts[0]
                if term.startswith("GO:"):
                    valid_terms.add(term)
                    try:
                        ia_weights[term] = float(parts[1])
                    except ValueError:
                        # If weight parsing fails, just add to valid terms
                        ia_weights[term] = 0.0

    return valid_terms, ia_weights


def load_valid_terms_from_ia(ia_path: Path | str) -> set[str]:
    """
    Load just the set of valid GO terms from IA.tsv file.

    This is a convenience wrapper around load_ia_weights() that only
    returns the valid terms set.

    Args:
        ia_path: Path to the IA.tsv file.

    Returns:
        Set of valid GO term IDs.
    """
    valid_terms, _ = load_ia_weights(ia_path)
    return valid_terms


def prune_to_training_terms_and_ancestors(
    all_terms: set[str],
    training_terms: set[str],
    parent_edges: dict[str, set[str]],
) -> set[str]:
    """
    Prune label space to terms seen in training plus their ancestors.

    Args:
        all_terms: Set of all possible GO terms.
        training_terms: Set of terms seen in training data.
        parent_edges: Dictionary mapping child terms to their parent terms.

    Returns:
        Pruned set of terms (training terms + ancestors).
    """
    pruned_terms: set[str] = set(training_terms)

    # Add all ancestors of training terms
    to_process = list(training_terms)
    while to_process:
        term = to_process.pop()
        parents = parent_edges.get(term, set())
        for parent in parents:
            if parent in all_terms and parent not in pruned_terms:
                pruned_terms.add(parent)
                to_process.append(parent)

    return pruned_terms


def build_term_index_from_terms(
    terms: set[str],
    term_to_ontology: dict[str, OntologyType],
) -> GOTermIndex:
    """
    Build a GOTermIndex from a set of terms.

    Args:
        terms: Set of GO terms to index.
        term_to_ontology: Mapping from GO terms to ontology types.

    Returns:
        GOTermIndex with all terms indexed.
    """
    index = GOTermIndex()

    # Sort terms for deterministic ordering
    for term in sorted(terms):
        if term in term_to_ontology:
            ontology = term_to_ontology[term]
            index.add_term(term, ontology)

    return index

