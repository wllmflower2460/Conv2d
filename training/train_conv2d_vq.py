#!/usr/bin/env python3
"""
Training script for Conv2d-VQ model
Validates VQ integration with existing preprocessing pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.conv2d_vq_model import Conv2dVQModel
from preprocessing.enhanced_pipeline import get_dataset
from config.training_config import TrainingConfig


class Conv2dVQTrainer:
    """Trainer for Conv2d-VQ model with comprehensive logging"""
    
    def __init__(
        self,
        model: Conv2dVQModel,
        device: torch.device,
        learning_rate: float = 3e-4,
        beta_recon: float = 1.0,
        beta_vq: float = 0.25,
        beta_activity: float = 2.0
    ):
        self.model = model.to(device)
        self.device = device
        
        # Loss weights
        self.beta_recon = beta_recon
        self.beta_vq = beta_vq
        self.beta_activity = beta_activity
        
        # Optimizer with different learning rates for different components
        self.optimizer = optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': learning_rate},
            {'params': model.decoder.parameters(), 'lr': learning_rate},
            {'params': model.activity_classifier.parameters(), 'lr': learning_rate * 2},
            {'params': model.state_predictor.parameters(), 'lr': learning_rate}
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.activity_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'perplexity': [],
            'codebook_usage': [],
            'reconstruction_loss': [],
            'vq_loss': [],
            'activity_loss': []
        }
        
    def compute_total_loss(self, outputs, data, labels):
        """Compute combined loss with all components"""
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(outputs['reconstructed'], data)
        
        # VQ loss (already computed in model)
        vq_loss = outputs['vq_loss']
        
        # Activity classification loss
        if labels is not None:
            activity_loss = self.activity_loss(outputs['activity_logits'], labels)
        else:
            activity_loss = torch.tensor(0.0, device=self.device)
        
        # Total weighted loss
        total_loss = (
            self.beta_recon * recon_loss +
            self.beta_vq * vq_loss +
            self.beta_activity * activity_loss
        )
        
        return total_loss, {
            'reconstruction': recon_loss.item(),
            'vq': vq_loss.item(),
            'activity': activity_loss.item(),
            'total': total_loss.item()
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        batch_metrics = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, dict):
                # Enhanced pipeline returns dict format
                data = batch['input']
                # Use human_label as primary activity label
                activity_labels = batch.get('human_label', None)
                domain_labels = batch.get('species_id', None)
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    data, activity_labels, domain_labels = batch
                elif len(batch) == 2:
                    data, activity_labels = batch
                    domain_labels = None
                else:
                    data = batch[0]
                    activity_labels = None
            else:
                data = batch
                activity_labels = None
            
            # Move to device
            data = data.to(self.device).float()
            if activity_labels is not None:
                activity_labels = activity_labels.to(self.device).long()
            
            # Forward pass
            outputs = self.model(data)
            
            # Compute loss
            loss, loss_components = self.compute_total_loss(outputs, data, activity_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            if activity_labels is not None:
                _, predicted = torch.max(outputs['activity_logits'], 1)
                total_correct += (predicted == activity_labels).sum().item()
                total_samples += activity_labels.size(0)
            
            batch_metrics.append({
                **loss_components,
                'perplexity': outputs['perplexity'].item(),
                'codebook_usage': outputs['codebook_usage'].item()
            })
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:3d}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Perplexity={outputs['perplexity']:.1f}, "
                      f"Usage={outputs['codebook_usage']:.1%}")
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_metrics = {
            key: np.mean([m[key] for m in batch_metrics])
            for key in batch_metrics[0].keys()
        }
        
        return avg_loss, accuracy, avg_metrics
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        batch_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    # Enhanced pipeline returns dict format
                    data = batch['input']
                    activity_labels = batch.get('human_label', None)
                elif isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        data, activity_labels = batch[:2]
                    else:
                        data = batch[0]
                        activity_labels = None
                else:
                    data = batch
                    activity_labels = None
                
                # Move to device
                data = data.to(self.device).float()
                if activity_labels is not None:
                    activity_labels = activity_labels.to(self.device).long()
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute loss
                loss, loss_components = self.compute_total_loss(outputs, data, activity_labels)
                
                total_loss += loss.item()
                
                if activity_labels is not None:
                    _, predicted = torch.max(outputs['activity_logits'], 1)
                    total_correct += (predicted == activity_labels).sum().item()
                    total_samples += activity_labels.size(0)
                
                batch_metrics.append({
                    **loss_components,
                    'perplexity': outputs['perplexity'].item(),
                    'codebook_usage': outputs['codebook_usage'].item()
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_metrics = {
            key: np.mean([m[key] for m in batch_metrics])
            for key in batch_metrics[0].keys()
        }
        
        return avg_loss, accuracy, avg_metrics
    
    def train(self, train_loader, val_loader, num_epochs, save_dir='models'):
        """Full training loop"""
        Path(save_dir).mkdir(exist_ok=True)
        best_val_acc = 0
        
        print(f"\n🚀 Starting Conv2d-VQ Training")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Track metrics
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['train_acc'].append(train_acc)
            self.metrics_history['val_acc'].append(val_acc)
            self.metrics_history['perplexity'].append(val_metrics['perplexity'])
            self.metrics_history['codebook_usage'].append(val_metrics['codebook_usage'])
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.3f}")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.3f}")
            print(f"  VQ:    Perplexity={val_metrics['perplexity']:.1f}, "
                  f"Usage={val_metrics['codebook_usage']:.1%}")
            print(f"  Components: Recon={val_metrics['reconstruction']:.4f}, "
                  f"VQ={val_metrics['vq']:.4f}, Act={val_metrics['activity']:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'metrics': val_metrics,
                    'config': {
                        'num_codes': self.model.vq.num_codes,
                        'code_dim': self.model.vq.code_dim
                    }
                }
                torch.save(checkpoint, f"{save_dir}/best_conv2d_vq_model.pth")
                print(f"  💾 Saved best model (accuracy: {val_acc:.3f})")
            
            # Get codebook statistics
            if epoch % 5 == 0:
                stats = self.model.get_codebook_stats()
                print(f"  Codebook: mean_dist={stats['mean_min_distance']:.3f}, "
                      f"std_dist={stats['std_min_distance']:.3f}")
        
        # Save training history
        with open(f"{save_dir}/conv2d_vq_training_history.json", 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.3f}")
        return self.metrics_history


def create_dummy_dataloaders(batch_size=32, num_samples=1000):
    """Create dummy dataloaders for testing"""
    from torch.utils.data import TensorDataset
    
    # Create dummy data matching expected input format
    # (B, 9, 2, 100) - 9-axis IMU, 2 devices, 100 timesteps
    train_data = torch.randn(num_samples, 9, 2, 100)
    train_labels = torch.randint(0, 12, (num_samples,))
    
    val_data = torch.randn(num_samples // 4, 9, 2, 100)
    val_labels = torch.randint(0, 12, (num_samples // 4,))
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = Conv2dVQModel(
        input_channels=9,
        input_height=2,
        num_codes=512,
        code_dim=64,
        hidden_channels=[64, 128, 256],
        num_activities=12,
        dropout=0.2
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Initialize trainer
    trainer = Conv2dVQTrainer(
        model=model,
        device=device,
        learning_rate=3e-4,
        beta_recon=1.0,
        beta_vq=0.25,
        beta_activity=2.0
    )
    
    # Try to load real dataset, fall back to dummy data
    try:
        from preprocessing.enhanced_pipeline import get_dataset
        print("Loading real dataset...")
        
        train_dataset = get_dataset(
            approach='cross_species',
            config_path='configs/enhanced_dataset_schema.yaml',
            mode='train',
            enforce_hailo_constraints=True
        )
        val_dataset = get_dataset(
            approach='cross_species',
            config_path='configs/enhanced_dataset_schema.yaml',
            mode='val',
            enforce_hailo_constraints=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        print(f"Loaded real dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        
    except Exception as e:
        print(f"Could not load real dataset: {e}")
        print("Using dummy data for testing...")
        train_loader, val_loader = create_dummy_dataloaders()
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,  # Short test run
        save_dir='models'
    )
    
    print("\n🎉 Conv2d-VQ training complete!")
    print(f"Final metrics:")
    print(f"  Train accuracy: {history['train_acc'][-1]:.3f}")
    print(f"  Val accuracy: {history['val_acc'][-1]:.3f}")
    print(f"  Perplexity: {history['perplexity'][-1]:.1f}")
    print(f"  Codebook usage: {history['codebook_usage'][-1]:.1%}")


if __name__ == "__main__":
    main()