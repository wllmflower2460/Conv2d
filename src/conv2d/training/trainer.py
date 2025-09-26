"""Trainer class with type hints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from conv2d.training.config import TrainingConfig


class Trainer:
    """Trainer for Conv2d-FSQ-HSMM models.
    
    Attributes:
        model: The model to train
        config: Training configuration
        optimizer: Optimizer instance
        device: Device to train on
        logger: Logger instance
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            optimizer: Optional optimizer (created if not provided)
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer
            
        self.current_epoch = 0
        self.global_step = 0
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        self.logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_epoch(train_loader)
            
            if val_loader is not None:
                self.evaluate(val_loader)
                
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint()
                
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            loss = self.train_step(batch)
            total_loss += loss
            
            if self.global_step % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, "
                    f"Step {self.global_step}, "
                    f"Loss: {loss:.4f}"
                )
                
            self.global_step += 1
            
        avg_loss = total_loss / len(train_loader)
        return avg_loss
        
    def train_step(self, batch: dict) -> float:
        """Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss value
        """
        # TODO: Implement actual training step
        loss = torch.tensor(0.0)
        return loss.item()
        
    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        metrics = {"val_loss": 0.0}
        
        with torch.no_grad():
            # TODO: Implement evaluation
            pass
            
        return metrics
        
    def save_checkpoint(self, path: Optional[Path] = None) -> None:
        """Save a checkpoint.
        
        Args:
            path: Optional path to save checkpoint
        """
        if path is None:
            path = self.config.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
            
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")