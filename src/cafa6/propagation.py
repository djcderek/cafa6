"""GO parent-propagation utilities.

Implements the rule: parent_score = max(parent_score, child_score)
This ensures that if a protein is annotated with a child term,
it is also annotated with all ancestor terms.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np

# Namespace mapping from OBO format to our ontology types
NAMESPACE_TO_ONTOLOGY: dict[str, Literal["MF", "BP", "CC"]] = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}


def load_go_edges_from_obo(
    obo_path: Path | str,
) -> tuple[dict[str, set[str]], dict[str, Literal["MF", "BP", "CC"]]]:
    """
    Parse an OBO file to extract GO parent-child edges and term-to-ontology mapping.

    Args:
        obo_path: Path to the go-basic.obo file.

    Returns:
        Tuple of:
        - child_to_parents: Dictionary mapping child terms to sets of parent terms.
        - term_to_ontology: Dictionary mapping GO terms to their ontology type (MF/BP/CC).
    """
    obo_path = Path(obo_path)
    child_to_parents: dict[str, set[str]] = {}
    term_to_ontology: dict[str, Literal["MF", "BP", "CC"]] = {}

    current_term: str | None = None
    current_namespace: str | None = None
    current_parents: set[str] = set()
    is_obsolete = False

    with open(obo_path, "r") as f:
        for line in f:
            line = line.strip()

            # Start of a new term
            if line == "[Term]":
                # Save previous term if valid
                if current_term is not None and not is_obsolete:
                    if current_parents:
                        child_to_parents[current_term] = current_parents
                    if current_namespace and current_namespace in NAMESPACE_TO_ONTOLOGY:
                        term_to_ontology[current_term] = NAMESPACE_TO_ONTOLOGY[current_namespace]

                # Reset for new term
                current_term = None
                current_namespace = None
                current_parents = set()
                is_obsolete = False

            elif line.startswith("[") and line.endswith("]"):
                # Non-Term stanza (e.g., [Typedef]), save current and skip
                if current_term is not None and not is_obsolete:
                    if current_parents:
                        child_to_parents[current_term] = current_parents
                    if current_namespace and current_namespace in NAMESPACE_TO_ONTOLOGY:
                        term_to_ontology[current_term] = NAMESPACE_TO_ONTOLOGY[current_namespace]

                current_term = None
                current_namespace = None
                current_parents = set()
                is_obsolete = False

            elif line.startswith("id: "):
                current_term = line[4:].strip()

            elif line.startswith("namespace: "):
                current_namespace = line[11:].strip()

            elif line.startswith("is_a: "):
                # Format: "is_a: GO:0048308 ! organelle inheritance"
                parent_part = line[6:].split("!")[0].strip()
                if parent_part.startswith("GO:"):
                    current_parents.add(parent_part)

            elif line.startswith("is_obsolete: true"):
                is_obsolete = True

        # Save the last term
        if current_term is not None and not is_obsolete:
            if current_parents:
                child_to_parents[current_term] = current_parents
            if current_namespace and current_namespace in NAMESPACE_TO_ONTOLOGY:
                term_to_ontology[current_term] = NAMESPACE_TO_ONTOLOGY[current_namespace]

    return child_to_parents, term_to_ontology


def load_go_edges(edges_path: Path | str) -> dict[str, set[str]]:
    """
    Load GO parent-child edges from a TSV file.

    Expects format: child_term<TAB>parent_term

    Args:
        edges_path: Path to the edges file.

    Returns:
        Dictionary mapping child terms to sets of parent terms.
    """
    edges_path = Path(edges_path)
    child_to_parents: dict[str, set[str]] = {}

    with open(edges_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                child, parent = parts[0], parts[1]
                if child not in child_to_parents:
                    child_to_parents[child] = set()
                child_to_parents[child].add(parent)

    return child_to_parents


def build_ancestor_map(child_to_parents: dict[str, set[str]]) -> dict[str, set[str]]:
    """
    Build a map from each term to all its ancestors (transitive closure).

    Args:
        child_to_parents: Dictionary mapping child terms to parent terms.

    Returns:
        Dictionary mapping terms to all their ancestors.
    """
    term_to_ancestors: dict[str, set[str]] = {}

    def get_ancestors(term: str, visited: set[str]) -> set[str]:
        """Recursively get all ancestors of a term."""
        if term in term_to_ancestors:
            return term_to_ancestors[term]

        if term in visited:
            return set()  # Avoid cycles

        visited.add(term)
        ancestors: set[str] = set()

        for parent in child_to_parents.get(term, set()):
            ancestors.add(parent)
            ancestors.update(get_ancestors(parent, visited))

        return ancestors

    # Compute ancestors for all terms
    all_terms = set(child_to_parents.keys())
    for parents in child_to_parents.values():
        all_terms.update(parents)

    for term in all_terms:
        term_to_ancestors[term] = get_ancestors(term, set())

    return term_to_ancestors


class GOPropagator:
    """
    Propagates prediction scores up the GO hierarchy.

    For each parent term, its score becomes the maximum of its current score
    and all its children's scores.
    """

    def __init__(self, child_to_parents: dict[str, set[str]]) -> None:
        """
        Initialize the propagator with GO edges.

        Args:
            child_to_parents: Dictionary mapping child terms to parent terms.
        """
        self.child_to_parents = child_to_parents
        self.ancestor_map = build_ancestor_map(child_to_parents)

    @classmethod
    def from_file(cls, edges_path: Path | str) -> "GOPropagator":
        """
        Create a propagator from an edges file.

        Args:
            edges_path: Path to the edges TSV file.

        Returns:
            GOPropagator instance.
        """
        child_to_parents = load_go_edges(edges_path)
        return cls(child_to_parents)

    @classmethod
    def from_obo(cls, obo_path: Path | str) -> "GOPropagator":
        """
        Create a propagator from an OBO ontology file.

        Args:
            obo_path: Path to the go-basic.obo file.

        Returns:
            GOPropagator instance.
        """
        child_to_parents, _ = load_go_edges_from_obo(obo_path)
        return cls(child_to_parents)

    def propagate_scores(
        self,
        term_scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Propagate scores up the GO hierarchy.

        For each term, all its ancestors receive at least its score.
        parent_score = max(parent_score, child_score)

        Args:
            term_scores: Dictionary mapping GO terms to their scores.

        Returns:
            Dictionary with propagated scores (includes ancestors).
        """
        propagated: dict[str, float] = dict(term_scores)

        # For each term, propagate its score to all ancestors
        for term, score in term_scores.items():
            ancestors = self.ancestor_map.get(term, set())
            for ancestor in ancestors:
                current_score = propagated.get(ancestor, 0.0)
                propagated[ancestor] = max(current_score, score)

        return propagated

    def propagate_batch(
        self,
        batch_scores: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """
        Propagate scores for a batch of proteins.

        Args:
            batch_scores: Dictionary mapping protein IDs to term scores.

        Returns:
            Dictionary with propagated scores for each protein.
        """
        return {
            protein_id: self.propagate_scores(term_scores)
            for protein_id, term_scores in batch_scores.items()
        }

    def propagate_array(
        self,
        scores: np.ndarray,
        idx_to_term: dict[int, str],
        term_to_idx: dict[str, int] | None = None,
    ) -> np.ndarray:
        """
        Propagate scores in array format.

        Args:
            scores: Array of shape (num_terms,) with prediction scores.
            idx_to_term: Mapping from indices to GO terms.
            term_to_idx: Optional mapping from GO terms to indices.
                        If None, will be derived from idx_to_term.

        Returns:
            Array with propagated scores.
        """
        if term_to_idx is None:
            term_to_idx = {term: idx for idx, term in idx_to_term.items()}

        propagated = scores.copy()

        # For each term, propagate to ancestors
        for idx, term in idx_to_term.items():
            score = scores[idx]
            ancestors = self.ancestor_map.get(term, set())

            for ancestor in ancestors:
                if ancestor in term_to_idx:
                    ancestor_idx = term_to_idx[ancestor]
                    propagated[ancestor_idx] = max(propagated[ancestor_idx], score)

        return propagated

