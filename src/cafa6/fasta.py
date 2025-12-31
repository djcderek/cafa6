"""FASTA file parsing utilities."""

from pathlib import Path
from typing import Iterator


def extract_protein_id(header: str) -> str:
    """
    Extract the protein accession from a FASTA header.

    Handles two formats:
    - UniProt format: >sp|A0A0C5B5G6|MOTSC_HUMAN ... -> A0A0C5B5G6
    - Simple format: >A0A0C5B5G6 9606 -> A0A0C5B5G6

    Args:
        header: The FASTA header line (with or without leading '>').

    Returns:
        The protein accession ID.
    """
    # Remove leading '>' if present
    if header.startswith(">"):
        header = header[1:]

    # Get first word
    first_word = header.split()[0]

    # Check for UniProt format: sp|ACCESSION|NAME or tr|ACCESSION|NAME
    if "|" in first_word:
        parts = first_word.split("|")
        if len(parts) >= 2 and parts[0] in ("sp", "tr"):
            return parts[1]

    # Otherwise return the first word as-is
    return first_word


def parse_fasta(fasta_path: Path | str) -> dict[str, str]:
    """
    Parse a FASTA file and return a dictionary of protein IDs to sequences.

    Handles UniProt-style headers (>sp|ACCESSION|NAME ...) and simple headers (>ID ...).

    Args:
        fasta_path: Path to the FASTA file.

    Returns:
        Dictionary mapping protein IDs to their sequences.
    """
    fasta_path = Path(fasta_path)
    proteins: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous protein if exists
                if current_id is not None:
                    proteins[current_id] = "".join(current_seq)

                # Parse new protein ID
                current_id = extract_protein_id(line)
                current_seq = []
            else:
                current_seq.append(line)

        # Save last protein
        if current_id is not None:
            proteins[current_id] = "".join(current_seq)

    return proteins


def iter_fasta(fasta_path: Path | str) -> Iterator[tuple[str, str]]:
    """
    Iterate over a FASTA file, yielding (protein_id, sequence) tuples.

    This is memory-efficient for large files.
    Handles UniProt-style headers (>sp|ACCESSION|NAME ...) and simple headers (>ID ...).

    Args:
        fasta_path: Path to the FASTA file.

    Yields:
        Tuples of (protein_id, sequence).
    """
    fasta_path = Path(fasta_path)
    current_id: str | None = None
    current_seq: list[str] = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Yield previous protein if exists
                if current_id is not None:
                    yield current_id, "".join(current_seq)

                # Parse new protein ID
                current_id = extract_protein_id(line)
                current_seq = []
            else:
                current_seq.append(line)

        # Yield last protein
        if current_id is not None:
            yield current_id, "".join(current_seq)


def write_fasta(proteins: dict[str, str], fasta_path: Path | str, line_width: int = 80) -> None:
    """
    Write proteins to a FASTA file.

    Args:
        proteins: Dictionary mapping protein IDs to sequences.
        fasta_path: Path to write the FASTA file.
        line_width: Maximum characters per line for sequences.
    """
    fasta_path = Path(fasta_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(fasta_path, "w") as f:
        for protein_id, sequence in proteins.items():
            f.write(f">{protein_id}\n")
            # Write sequence in chunks
            for i in range(0, len(sequence), line_width):
                f.write(sequence[i : i + line_width] + "\n")

