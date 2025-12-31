"""ProtT5 embedding generation and caching utilities."""

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


def load_prott5_model(
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    device: torch.device | None = None,
) -> tuple[T5Tokenizer, T5EncoderModel]:
    """
    Load the ProtT5 model and tokenizer.

    Args:
        model_name: HuggingFace model name for ProtT5.
        device: Device to load model on. If None, auto-detects GPU/CPU.

    Returns:
        Tuple of (tokenizer, model).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    return tokenizer, model


def prepare_sequence_for_prott5(sequence: str) -> str:
    """
    Prepare a protein sequence for ProtT5 tokenization.

    ProtT5 expects spaces between amino acids.

    Args:
        sequence: Raw protein sequence.

    Returns:
        Space-separated sequence.
    """
    # Replace rare/unknown amino acids with X
    sequence = sequence.upper()
    sequence = sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    # Add spaces between amino acids
    return " ".join(list(sequence))


def generate_embedding(
    sequence: str,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    max_length: int = 1024,
) -> np.ndarray:
    """
    Generate a single protein embedding using ProtT5.

    Args:
        sequence: Protein sequence.
        tokenizer: ProtT5 tokenizer.
        model: ProtT5 model.
        device: Device for computation.
        max_length: Maximum sequence length to process.

    Returns:
        Mean-pooled embedding of shape (1024,).
    """
    # Prepare sequence
    prepared = prepare_sequence_for_prott5(sequence)

    # Tokenize
    inputs = tokenizer(
        prepared,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Get last hidden state: (batch_size=1, seq_len, hidden_dim=1024)
        hidden_states = outputs.last_hidden_state

        # Mean pooling over sequence length (excluding padding)
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        embedding = sum_embeddings / sum_mask

    return embedding.cpu().numpy().squeeze()


def generate_embeddings_batch(
    sequences: list[str],
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    max_length: int = 1024,
) -> np.ndarray:
    """
    Generate embeddings for a batch of sequences.

    Args:
        sequences: List of protein sequences.
        tokenizer: ProtT5 tokenizer.
        model: ProtT5 model.
        device: Device for computation.
        max_length: Maximum sequence length.

    Returns:
        Array of embeddings of shape (batch_size, 1024).
    """
    # Prepare sequences
    prepared = [prepare_sequence_for_prott5(seq) for seq in sequences]

    # Tokenize batch
    inputs = tokenizer(
        prepared,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        embeddings = sum_embeddings / sum_mask

    return embeddings.cpu().numpy()


def cache_embeddings_parquet(
    protein_ids: list[str],
    embeddings: np.ndarray,
    output_path: Path | str,
) -> None:
    """
    Cache embeddings to a Parquet file.

    Args:
        protein_ids: List of protein IDs.
        embeddings: Array of embeddings of shape (num_proteins, embedding_dim).
        output_path: Path to save the Parquet file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame with protein_id and embedding columns
    df = pd.DataFrame({
        "protein_id": protein_ids,
        "embedding": [emb.tolist() for emb in embeddings],
    })

    df.to_parquet(output_path, index=False)


def cache_embeddings_npz(
    protein_ids: list[str],
    embeddings: np.ndarray,
    output_path: Path | str,
) -> None:
    """
    Cache embeddings to an NPZ file.

    Args:
        protein_ids: List of protein IDs.
        embeddings: Array of embeddings of shape (num_proteins, embedding_dim).
        output_path: Path to save the NPZ file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        protein_ids=np.array(protein_ids),
        embeddings=embeddings,
    )


def load_embeddings_parquet(parquet_path: Path | str) -> tuple[list[str], np.ndarray]:
    """
    Load embeddings from a Parquet file.

    Args:
        parquet_path: Path to the Parquet file.

    Returns:
        Tuple of (protein_ids, embeddings array).
    """
    df = pd.read_parquet(parquet_path)
    protein_ids = df["protein_id"].tolist()
    embeddings = np.array(df["embedding"].tolist())
    return protein_ids, embeddings


def load_embeddings_npz(npz_path: Path | str) -> tuple[list[str], np.ndarray]:
    """
    Load embeddings from an NPZ file.

    Args:
        npz_path: Path to the NPZ file.

    Returns:
        Tuple of (protein_ids, embeddings array).
    """
    data = np.load(npz_path, allow_pickle=True)
    protein_ids = data["protein_ids"].tolist()
    embeddings = data["embeddings"]
    return protein_ids, embeddings


def load_embeddings(
    embeddings_path: Path | str,
) -> tuple[list[str], np.ndarray]:
    """
    Load embeddings from a file (auto-detect format).

    Args:
        embeddings_path: Path to the embeddings file.

    Returns:
        Tuple of (protein_ids, embeddings array).
    """
    embeddings_path = Path(embeddings_path)

    if embeddings_path.suffix == ".parquet":
        return load_embeddings_parquet(embeddings_path)
    elif embeddings_path.suffix == ".npz":
        return load_embeddings_npz(embeddings_path)
    else:
        raise ValueError(f"Unknown embedding format: {embeddings_path.suffix}")


class EmbeddingGenerator:
    """
    Generates and caches ProtT5 embeddings for protein sequences.
    """

    def __init__(
        self,
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
        batch_size: int = 8,
        max_length: int = 1024,
        cache_format: str = "parquet",
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the embedding generator.

        Args:
            model_name: HuggingFace model name.
            batch_size: Batch size for embedding generation.
            max_length: Maximum sequence length.
            cache_format: Output format ("parquet" or "npz").
            device: Device for computation.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_format = cache_format
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer: T5Tokenizer | None = None
        self.model: T5EncoderModel | None = None

    def load_model(self) -> None:
        """Load the ProtT5 model and tokenizer."""
        if self.model is None:
            self.tokenizer, self.model = load_prott5_model(
                self.model_name,
                self.device,
            )

    def generate_and_cache(
        self,
        proteins: dict[str, str],
        output_path: Path | str,
        show_progress: bool = True,
    ) -> None:
        """
        Generate embeddings for all proteins and cache to file.

        Args:
            proteins: Dictionary mapping protein IDs to sequences.
            output_path: Path to save the embeddings.
            show_progress: Whether to show a progress bar.
        """
        self.load_model()
        assert self.tokenizer is not None and self.model is not None

        protein_ids = list(proteins.keys())
        sequences = list(proteins.values())

        all_embeddings: list[np.ndarray] = []

        # Process in batches
        iterator = range(0, len(sequences), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")

        for i in iterator:
            batch_seqs = sequences[i : i + self.batch_size]
            batch_embs = generate_embeddings_batch(
                batch_seqs,
                self.tokenizer,
                self.model,
                self.device,
                self.max_length,
            )
            all_embeddings.append(batch_embs)

        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)

        # Cache to file
        output_path = Path(output_path)
        if self.cache_format == "parquet":
            cache_embeddings_parquet(protein_ids, embeddings, output_path)
        else:
            cache_embeddings_npz(protein_ids, embeddings, output_path)

