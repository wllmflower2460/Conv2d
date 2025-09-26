"""Training configuration with type hints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for model training.
    
    Attributes:
        batch_size: Training batch size
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints
        log_interval: Steps between logging
        eval_interval: Steps between evaluations
        save_interval: Steps between checkpoint saves
        gradient_clip: Max gradient norm for clipping
        weight_decay: L2 regularization coefficient
        seed: Random seed for reproducibility
    """
    
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    device: str = "cuda"
    
    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_interval: int = 1000
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    
    # Optimization
    gradient_clip: Optional[float] = 1.0
    weight_decay: float = 1e-4
    
    # Reproducibility
    seed: Optional[int] = 42
    
    def __post_init__(self) -> None:
        """Validate and process configuration."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)