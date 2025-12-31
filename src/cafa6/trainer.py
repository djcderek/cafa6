"""Training loop with checkpointing, deterministic seeds, and early stopping."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .model import MultiHeadMLP


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping handler to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Trainer for the multi-head MLP model.

    Handles training loop, validation, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: MultiHeadMLP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_dir: Path | str = "checkpoints",
        patience: int = 10,
        device: torch.device | None = None,
        use_wandb: bool = False,
        wandb_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: The model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            checkpoint_dir: Directory to save checkpoints.
            patience: Patience for early stopping.
            device: Device for training.
            use_wandb: Whether to log metrics to Weights & Biases.
            wandb_config: Configuration dict to log to wandb.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Binary cross-entropy loss for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.early_stopping = EarlyStopping(patience=patience)
        self.best_val_loss = float("inf")

        # Wandb logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_config = wandb_config or {}
        if self.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Skipping wandb logging.")

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            embeddings = batch["embedding"].to(self.device)
            targets_mf = batch["targets_mf"].to(self.device)
            targets_bp = batch["targets_bp"].to(self.device)
            targets_cc = batch["targets_cc"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(embeddings)

            # Compute loss for each head
            loss_mf = self.criterion(outputs["logits_mf"], targets_mf)
            loss_bp = self.criterion(outputs["logits_bp"], targets_bp)
            loss_cc = self.criterion(outputs["logits_cc"], targets_cc)

            # Combined loss (simple average)
            loss = (loss_mf + loss_bp + loss_cc) / 3.0

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate the model.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            embeddings = batch["embedding"].to(self.device)
            targets_mf = batch["targets_mf"].to(self.device)
            targets_bp = batch["targets_bp"].to(self.device)
            targets_cc = batch["targets_cc"].to(self.device)

            # Forward pass
            outputs = self.model(embeddings)

            # Compute loss for each head
            loss_mf = self.criterion(outputs["logits_mf"], targets_mf)
            loss_bp = self.criterion(outputs["logits_bp"], targets_bp)
            loss_cc = self.criterion(outputs["logits_cc"], targets_cc)

            loss = (loss_mf + loss_bp + loss_cc) / 3.0

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        """
        Save a model checkpoint.

        Args:
            path: Path to save the checkpoint.
            epoch: Current epoch number.
            val_loss: Current validation loss.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "model_config": self.model.get_config(),
        }
        torch.save(checkpoint, path)

    def train(self, epochs: int) -> dict[str, list[float]]:
        """
        Run the full training loop.

        Args:
            epochs: Maximum number of epochs to train.

        Returns:
            Dictionary with training history:
                - train_loss: List of training losses per epoch
                - val_loss: List of validation losses per epoch
        """
        # Initialize wandb if enabled
        if self.use_wandb:
            config = {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "epochs": epochs,
                "patience": self.early_stopping.patience,
                "batch_size": self.train_loader.batch_size,
                "model_config": self.model.get_config(),
                **self.wandb_config,
            }
            wandb.init(
                project="cafa6",
                config=config,
                reinit=True,
            )
            # Watch model for gradient logging
            wandb.watch(self.model, log="gradients", log_freq=100)

        history = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            train_loss = self.train_epoch()
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = self.validate()
            history["val_loss"].append(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": self.best_val_loss if val_loss >= self.best_val_loss else val_loss,
                    "learning_rate": self.learning_rate,
                })

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    self.checkpoint_dir / "best_model.pt",
                    epoch,
                    val_loss,
                )
                print("  -> Saved best model")

            # Save last model
            self.save_checkpoint(
                self.checkpoint_dir / "last_model.pt",
                epoch,
                val_loss,
            )

            # Check early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        # Finish wandb run
        if self.use_wandb:
            wandb.finish()

        return history


def train_model(
    model: MultiHeadMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 100,
    patience: int = 10,
    checkpoint_dir: Path | str = "checkpoints",
    seed: int = 42,
    device: torch.device | None = None,
    use_wandb: bool = False,
    wandb_config: dict[str, Any] | None = None,
) -> dict[str, list[float]]:
    """
    Convenience function to train a model.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        epochs: Maximum epochs.
        patience: Early stopping patience.
        checkpoint_dir: Checkpoint directory.
        seed: Random seed.
        device: Device for training.
        use_wandb: Whether to log metrics to Weights & Biases.
        wandb_config: Additional configuration to log to wandb.

    Returns:
        Training history.
    """
    set_seed(seed)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        device=device,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
    )

    return trainer.train(epochs)

