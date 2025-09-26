#!/usr/bin/env python3
"""Training script with Hydra configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from conv2d.config import Config, compute_config_hash, save_config_with_hash
from conv2d.utils import set_seed


logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../../conf", config_name="base")
def train(cfg: DictConfig) -> float:
    """Train model with Hydra configuration.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Final validation metric
    """
    # Convert to Pydantic model for validation
    try:
        config = Config(**OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    
    # Set seed for reproducibility
    set_seed(config.seed)
    logger.info(f"Set random seed: {config.seed}")
    
    # Compute and save config hash
    config_hash = compute_config_hash(cfg)
    logger.info(f"Configuration hash: {config_hash}")
    
    # Create output directory with hash
    output_dir = Path.cwd()  # Hydra changes to output directory
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration with hash
    config_file = save_config_with_hash(cfg, output_dir)
    logger.info(f"Configuration saved to: {config_file}")
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set up hardware
    device = torch.device(config.hardware.device.value)
    logger.info(f"Using device: {device}")
    
    if config.hardware.device.value == "cuda":
        # Configure CUDA settings
        torch.backends.cudnn.benchmark = config.hardware.cudnn_benchmark
        torch.backends.cudnn.deterministic = config.hardware.cudnn_deterministic
        if config.hardware.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    # Initialize model
    logger.info(f"Initializing model: {config.model.name}")
    # TODO: Implement model initialization
    
    # Initialize data loaders
    logger.info(f"Loading dataset: {config.data.dataset_name}")
    # TODO: Implement data loading
    
    # Initialize optimizer
    logger.info(f"Optimizer: {config.training.optimizer.type}")
    # TODO: Implement optimizer initialization
    
    # Training loop
    logger.info(f"Starting training for {config.training.epochs} epochs")
    
    best_metric = float('inf')
    for epoch in range(config.training.epochs):
        # TODO: Implement training
        logger.info(f"Epoch {epoch + 1}/{config.training.epochs}")
        
        # Dummy metric
        metric = 1.0 / (epoch + 1)
        
        if metric < best_metric:
            best_metric = metric
            logger.info(f"New best metric: {best_metric:.4f}")
            
            # Save checkpoint
            checkpoint_path = output_dir / f"best_model_hash_{config_hash[:8]}.pth"
            logger.info(f"Saving checkpoint: {checkpoint_path}")
            # TODO: Save actual model
            
    logger.info(f"Training complete. Best metric: {best_metric:.4f}")
    return best_metric


if __name__ == "__main__":
    train()