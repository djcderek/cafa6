# CAFA-6 Protein Function Prediction

A minimal but solid boilerplate for the CAFA-6 Kaggle competition: Protein sequence → ProtT5 embeddings → MLP → multi-label GO term predictions.

## Features

- **ProtT5 Embeddings**: Generate and cache protein embeddings using Rostlab/prot_t5_xl_uniref50
- **Multi-Head MLP**: Shared trunk with separate heads for MF, BP, CC ontologies
- **GO Propagation**: Parent score = max(child scores) at inference
- **Label Pruning**: Restrict to training terms + ancestors
- **Submission Pipeline**: Kaggle-compliant TSV with ≤1500 terms per protein

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Quick Start (Dummy Mode)

Test the full pipeline without real data:

```bash
# Generate dummy embeddings
python scripts/generate_embeddings.py --dummy --config configs/default.yaml

# Train model
python scripts/train.py --dummy --config configs/default.yaml

# Generate predictions
python scripts/predict.py --dummy --config configs/default.yaml
```

## Full Pipeline

### 1. Prepare Data

Place your data files in the `data/` directory:
- `train.fasta`: Training protein sequences
- `test.fasta`: Test protein sequences  
- `train_terms.tsv`: Training annotations (protein_id, GO_term)
- `go_edges.tsv`: GO parent-child edges (child, parent)
- `valid_terms.txt`: Valid GO terms for submission

### 2. Generate Embeddings

```bash
python scripts/generate_embeddings.py --config configs/default.yaml
```

### 3. Train Model

```bash
python scripts/train.py --config configs/default.yaml
```

### 4. Generate Submission

```bash
python scripts/predict.py --config configs/default.yaml
```

## Configuration

All settings are in `configs/default.yaml`:

```yaml
data:
  train_fasta: data/train.fasta
  test_fasta: data/test.fasta
  embeddings_dir: data/embeddings
  # ...

model:
  embedding_dim: 1024
  hidden_dims: [512, 256]
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  # ...
```

## Project Structure

```
cafa6/
├── configs/
│   └── default.yaml
├── data/
│   └── .gitkeep
├── scripts/
│   ├── generate_embeddings.py
│   ├── train.py
│   └── predict.py
└── src/
    └── cafa6/
        ├── config.py         # Configuration dataclasses
        ├── fasta.py          # FASTA parsing
        ├── ontology.py       # GO term indexing
        ├── propagation.py    # GO parent propagation
        ├── embeddings.py     # ProtT5 embedding generation
        ├── dataset.py        # PyTorch dataset
        ├── model.py          # Multi-head MLP
        ├── trainer.py        # Training loop
        ├── inference.py      # Prediction pipeline
        ├── submission.py     # Kaggle submission writer
        ├── metrics.py        # Evaluation metrics (placeholder)
        └── dummy.py          # Dummy data generation
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## License

MIT

