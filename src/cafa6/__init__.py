"""CAFA-6 Protein Function Prediction."""

__version__ = "0.1.0"

from .config import Config, DataConfig, load_config
from .fasta import extract_protein_id, iter_fasta, parse_fasta
from .ontology import (
    GOTermIndex,
    load_ia_weights,
    load_term_ontology_from_train_terms,
    load_valid_terms_from_ia,
)
from .propagation import GOPropagator, load_go_edges_from_obo

__all__ = [
    "Config",
    "DataConfig",
    "GOPropagator",
    "GOTermIndex",
    "extract_protein_id",
    "iter_fasta",
    "load_config",
    "load_go_edges_from_obo",
    "load_ia_weights",
    "load_term_ontology_from_train_terms",
    "load_valid_terms_from_ia",
    "parse_fasta",
]

