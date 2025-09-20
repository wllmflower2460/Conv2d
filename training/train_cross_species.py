"""
Cross-Species Training Script for Hailo-Compatible TCN-VAE
Implements multi-task learning for human→dog behavioral transfer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
import wandb
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tcn_vae_hailo import HailoTCNVAE, CrossSpeciesLoss
from models.device_attention import PhoneIMUAttention
from preprocessing.enhanced_pipeline import EnhancedCrossSpeciesDataset, HailoDataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossSpeciesTrainer:
    """
    Trainer for cross-species behavioral analysis with Hailo optimization
    """
    
    def __init__(self, config_path: str, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to enhanced YAML configuration
            checkpoint_dir: Directory for saving checkpoints
        """
        # Store config path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimization()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_dog_accuracy = 0.0
        self.best_human_accuracy = 0.0
    
    def _prepare_dog_labels_for_loss(self, dog_labels: torch.Tensor, 
                                     has_dog_label: torch.Tensor) -> torch.Tensor:
        """
        Helper method to prepare dog labels for loss calculation.
        Returns empty tensor when no dog labels are present to avoid None issues.
        
        Args:
            dog_labels: Dog behavior labels tensor
            has_dog_label: Boolean mask indicating valid dog labels
            
        Returns:
            Dog labels tensor for loss calculation (empty if no valid labels)
        """
        if has_dog_label.any():
            return dog_labels
        else:
            # Pass an empty tensor of the correct type and device
            return torch.empty((0,), dtype=dog_labels.dtype, device=dog_labels.device)
        
    def _setup_data(self):
        """Setup data loaders"""
        logger.info("Loading datasets...")
        
        # Create datasets
        self.train_dataset = EnhancedCrossSpeciesDataset(
            self.config_path, mode='train', enforce_hailo_constraints=True
        )
        self.val_dataset = EnhancedCrossSpeciesDataset(
            self.config_path, mode='val', enforce_hailo_constraints=True
        )
        self.test_dataset = EnhancedCrossSpeciesDataset(
            self.config_path, mode='test', enforce_hailo_constraints=True
        )
        
        # Create data loaders
        batch_size = self.config['training_config']['batch_size']
        
        self.train_loader = self.train_dataset.get_dataloader(
            batch_size=batch_size, shuffle=True
        )
        self.val_loader = self.val_dataset.get_dataloader(
            batch_size=batch_size, shuffle=False
        )
        self.test_loader = self.test_dataset.get_dataloader(
            batch_size=batch_size, shuffle=False
        )
        
        logger.info(f"✅ Data loaded - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def _setup_model(self):
        """Setup model and loss function"""
        logger.info("Initializing model...")
        
        # Create model
        self.model = HailoTCNVAE(
            input_dim=9,
            hidden_dims=[64, 128, 256],
            latent_dim=64,
            sequence_length=100,
            num_human_activities=12,
            num_dog_behaviors=3,
            use_device_attention=True
        ).to(self.device)
        
        # Validate Hailo compatibility
        validator = HailoDataValidator()
        is_valid = validator.validate_model_ops(self.model, self.config)
        
        if not is_valid:
            raise ValueError("Model is not Hailo-compatible!")
        
        # Setup loss function
        loss_weights = self.config['training_config']['loss_weights']
        self.criterion = CrossSpeciesLoss(
            beta=loss_weights['kl_divergence'],
            dog_weight=loss_weights['dog_classification']
        )
        
        logger.info(f"✅ Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _setup_optimization(self):
        """Setup optimizer and scheduler"""
        opt_config = self.config['training_config']['optimizer']
        sched_config = self.config['training_config']['scheduler']
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_config['learning_rate'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=sched_config['T_0'],
            T_mult=sched_config['T_mult'],
            eta_min=sched_config['eta_min']
        )
        
        logger.info("✅ Optimizer and scheduler configured")
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        log_config = self.config['logging']
        
        # Initialize wandb if enabled
        if log_config['wandb']['enabled']:
            wandb.init(
                project=log_config['wandb']['project'],
                tags=log_config['wandb']['tags'],
                config=self.config
            )
            wandb.watch(self.model, log='all')
        
        self.use_wandb = log_config['wandb']['enabled']
        
        logger.info("✅ Logging initialized")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0
        human_correct = 0
        dog_correct = 0
        dog_total = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            inputs = batch['input'].to(self.device)
            human_labels = batch['human_label'].to(self.device)
            dog_labels = batch['dog_label'].to(self.device)
            has_dog_label = batch['has_dog_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss
            dog_labels_for_loss = self._prepare_dog_labels_for_loss(dog_labels, has_dog_label)
            
            targets = {
                'input': inputs,
                'human_labels': human_labels,
                'dog_labels': dog_labels_for_loss
            }
            
            losses = self.criterion(outputs, targets)
            loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training_config']['gradient_clip_norm']
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Human activity accuracy
            human_pred = outputs['human_logits'].argmax(dim=1)
            human_correct += (human_pred == human_labels).sum().item()
            
            # Dog behavior accuracy (only for samples with labels)
            if has_dog_label.any():
                dog_mask = has_dog_label.bool()
                dog_pred = outputs['dog_logits'][dog_mask].argmax(dim=1)
                dog_correct += (dog_pred == dog_labels[dog_mask]).sum().item()
                dog_total += dog_mask.sum().item()
            
            total_samples += inputs.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'human_acc': human_correct / total_samples,
                'dog_acc': dog_correct / dog_total if dog_total > 0 else 0
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/reconstruction_loss': losses['reconstruction'].item(),
                    'train/kl_loss': losses['kl'].item(),
                    'train/human_loss': losses['human'].item(),
                    'train/dog_loss': losses['dog'].item() if torch.is_tensor(losses['dog']) else 0,
                    'train/attention_mean': outputs['attention_weights'].mean().item() 
                        if outputs['attention_weights'] is not None else 0
                })
        
        # Calculate epoch metrics
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'human_accuracy': human_correct / total_samples,
            'dog_accuracy': dog_correct / dog_total if dog_total > 0 else 0
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        human_correct = 0
        dog_correct = 0
        dog_total = 0
        total_samples = 0
        
        # Store predictions for confusion matrix
        all_human_preds = []
        all_human_labels = []
        all_dog_preds = []
        all_dog_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                inputs = batch['input'].to(self.device)
                human_labels = batch['human_label'].to(self.device)
                dog_labels = batch['dog_label'].to(self.device)
                has_dog_label = batch['has_dog_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                targets = {
                    'input': inputs,
                    'human_labels': human_labels,
                    'dog_labels': dog_labels if has_dog_label.any() else None
                }
                
                losses = self.criterion(outputs, targets)
                total_loss += losses['total'].item()
                
                # Human activity accuracy
                human_pred = outputs['human_logits'].argmax(dim=1)
                human_correct += (human_pred == human_labels).sum().item()
                
                all_human_preds.extend(human_pred.cpu().numpy())
                all_human_labels.extend(human_labels.cpu().numpy())
                
                # Dog behavior accuracy
                if has_dog_label.any():
                    dog_mask = has_dog_label.bool()
                    dog_pred = outputs['dog_logits'][dog_mask].argmax(dim=1)
                    dog_correct += (dog_pred == dog_labels[dog_mask]).sum().item()
                    dog_total += dog_mask.sum().item()
                    
                    all_dog_preds.extend(dog_pred.cpu().numpy())
                    all_dog_labels.extend(dog_labels[dog_mask].cpu().numpy())
                
                total_samples += inputs.size(0)
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'human_accuracy': human_correct / total_samples,
            'dog_accuracy': dog_correct / dog_total if dog_total > 0 else 0
        }
        
        # Check if we meet commercial requirements
        target_accuracy = self.config['commercial_validation']['trainer_requirements']['min_accuracy']
        if metrics['dog_accuracy'] >= target_accuracy:
            logger.info(f"🎯 Commercial target met! Dog accuracy: {metrics['dog_accuracy']:.2%} >= {target_accuracy:.2%}")
        
        return metrics
    
    def train(self, num_epochs: int):
        """
        Train model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Training curriculum from config
        curriculum = self.config['training_config']['curriculum']
        current_phase_idx = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Check curriculum phase
            if current_phase_idx < len(curriculum):
                phase = curriculum[current_phase_idx]
                if epoch >= sum(p['epochs'] for p in curriculum[:current_phase_idx+1]):
                    current_phase_idx += 1
                    if current_phase_idx < len(curriculum):
                        phase = curriculum[current_phase_idx]
                        logger.info(f"📚 Switching to curriculum phase: {phase['phase']}")
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch}: "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Human Acc: {train_metrics['human_accuracy']:.2%}, "
                       f"Train Dog Acc: {train_metrics['dog_accuracy']:.2%}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Human Acc: {val_metrics['human_accuracy']:.2%}, "
                       f"Val Dog Acc: {val_metrics['dog_accuracy']:.2%}")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'val/loss': val_metrics['loss'],
                    'val/human_accuracy': val_metrics['human_accuracy'],
                    'val/dog_accuracy': val_metrics['dog_accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint if best
            if val_metrics['dog_accuracy'] > self.best_dog_accuracy:
                self.best_dog_accuracy = val_metrics['dog_accuracy']
                self.save_checkpoint('best_dog_model.pth', epoch, val_metrics)
            
            if val_metrics['human_accuracy'] > self.best_human_accuracy:
                self.best_human_accuracy = val_metrics['human_accuracy']
                self.save_checkpoint('best_human_model.pth', epoch, val_metrics)
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_metrics)
        
        logger.info(f"✅ Training complete! Best dog accuracy: {self.best_dog_accuracy:.2%}")
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"💾 Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        
        logger.info(f"✅ Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['metrics']
    
    def export_for_hailo(self, checkpoint_path: Optional[str] = None):
        """Export model for Hailo deployment"""
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Create dummy input with static shape
        dummy_input = torch.randn(1, 9, 2, 100).to(self.device)
        
        # Export to ONNX
        output_path = self.checkpoint_dir / "tcn_vae_hailo.onnx"
        self.model.export_for_hailo(dummy_input, str(output_path))
        
        logger.info(f"✅ Model exported for Hailo: {output_path}")


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Species TCN-VAE Training")
    parser.add_argument('--config', type=str, default='configs/enhanced_dataset_schema.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--export', action='store_true',
                       help='Export model for Hailo after training')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CrossSpeciesTrainer(args.config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    trainer.train(args.epochs)
    
    # Export for Hailo if requested
    if args.export:
        trainer.export_for_hailo()
    
    logger.info("🎉 Training complete!")


if __name__ == "__main__":
    main()