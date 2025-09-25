#!/usr/bin/env python3
"""
Overnight TCN-VAE Training Run
Started: Now, running through tomorrow morning
Target: >60% validation accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import json
import time
from datetime import datetime
import pickle
import signal

sys.path.append('/home/wllmflower/tcn-vae-training')
from models.tcn_vae import TCNVAE
from preprocessing.unified_pipeline import MultiDatasetHAR
from config.training_config import TrainingConfig


class OvernightTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
        # Improved optimizer with better settings
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=3e-4,  # More conservative LR
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Scheduler with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2
        )
        
        # Loss functions with improvements
        self.reconstruction_loss = nn.MSELoss()
        self.activity_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.domain_loss = nn.CrossEntropyLoss()
        
        # Better loss balancing from your successful run
        self.beta = 0.4  # Reduced KL weight
        self.lambda_act = 2.5  # Increased activity focus
        self.lambda_dom = 0.05  # Minimal domain confusion
        
        # Training state - Use configuration for baseline
        self.best_val_accuracy = TrainingConfig.get_baseline('enhanced')  # Beat current best
        self.patience = TrainingConfig.OVERNIGHT_PATIENCE  # Configurable patience 
        self.patience_counter = 0
        self.epoch_times = []
        
    def vae_loss(self, recon_x, x, mu, logvar, epoch=0):
        recon_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Progressive beta annealing
        beta = self.beta * min(1.0, epoch / 20)
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct_activities = 0
        total_samples = 0
        
        for batch_idx, (data, activity_labels, domain_labels) in enumerate(train_loader):
            data = data.to(self.device).float()
            activity_labels = activity_labels.to(self.device).long()
            domain_labels = domain_labels.to(self.device).long()
            
            # Add slight noise for regularization
            if epoch > 5:
                data += torch.randn_like(data) * 0.01
            
            # Progressive domain adaptation
            p = float(batch_idx + epoch * len(train_loader)) / (30 * len(train_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                recon_x, mu, logvar, activity_logits, domain_logits = self.model(data, alpha)
                
                # Compute losses
                vae_loss, recon_loss, kl_loss = self.vae_loss(recon_x, data, mu, logvar, epoch)
                act_loss = self.activity_loss(activity_logits, activity_labels)
                dom_loss = self.domain_loss(domain_logits, domain_labels)
                
                # Combined loss with progressive weighting
                activity_weight = self.lambda_act * (1 + epoch / 200)
                total_batch_loss = vae_loss + activity_weight * act_loss + self.lambda_dom * dom_loss
                
                # Backward with gradient clipping
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.8)
                
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
                
                # Track accuracy
                pred_activities = activity_logits.argmax(dim=1)
                correct_activities += (pred_activities == activity_labels).sum().item()
                total_samples += activity_labels.size(0)
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
                
            if batch_idx % 100 == 0:
                current_acc = correct_activities / total_samples if total_samples > 0 else 0
                print(f'Epoch {epoch+1}, Batch {batch_idx}: '
                      f'Loss: {total_batch_loss.item():.4f}, '
                      f'Train Acc: {current_acc:.4f}, '
                      f'LR: {self.scheduler.get_last_lr()[0]:.6f}')
        
        train_accuracy = correct_activities / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, train_accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct_activities = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, activity_labels, domain_labels in val_loader:
                data = data.to(self.device).float()
                activity_labels = activity_labels.to(self.device).long()
                
                recon_x, mu, logvar, activity_logits, domain_logits = self.model(data)
                
                vae_loss, _, _ = self.vae_loss(recon_x, data, mu, logvar)
                act_loss = self.activity_loss(activity_logits, activity_labels)
                
                val_loss += (vae_loss + act_loss).item()
                
                pred_activities = activity_logits.argmax(dim=1)
                correct_activities += (pred_activities == activity_labels).sum().item()
                total_samples += activity_labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_activities / total_samples
        
        return avg_val_loss, val_accuracy
    
    def log_progress(self, epoch, train_loss, train_acc, val_loss, val_acc, epoch_time):
        """Log detailed progress"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "epoch_time": float(epoch_time),
            "learning_rate": float(self.scheduler.get_last_lr()[0]),
            "best_so_far": float(self.best_val_accuracy),
            "is_best": val_acc > self.best_val_accuracy
        }
        
        # Append to log file
        with open('/home/wllmflower/tcn-vae-training/logs/overnight_training.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Console output
        improvement = "🔥 NEW BEST!" if val_acc > self.best_val_accuracy else ""
        print(f"\n[{timestamp}] Epoch {epoch+1} Complete:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f} {improvement}")
        print(f"  Time: {epoch_time:.1f}s, LR: {self.scheduler.get_last_lr()[0]:.6f}")
        print(f"  Best: {self.best_val_accuracy:.4f}")


def setup_signal_handler():
    """Handle graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, saving current state...")
        # Trainer state will be saved in main loop
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def main():
    setup_signal_handler()
    
    print("🌙 Starting NEW Overnight TCN-VAE Training Session...")
    print(f"⏰ Started at: {datetime.now()}")
    print("🎯 Target: Beat 86.53% validation accuracy (New Challenge!)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    
    # Load data
    print("📊 Loading datasets...")
    processor = MultiDatasetHAR(window_size=100, overlap=0.5)
    X_train, y_train, domains_train, X_val, y_val, domains_val = processor.preprocess_all()
    
    num_activities = len(np.unique(y_train))
    print(f"📈 Classes: {num_activities}")
    
    # Data loaders with optimizations
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train),
        torch.LongTensor(domains_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
        torch.LongTensor(domains_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = TCNVAE(input_dim=9, hidden_dims=[64, 128, 256], latent_dim=64, num_activities=num_activities)
    trainer = OvernightTrainer(model, device)
    
    print(f"🔢 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Clear log file
    os.makedirs('/home/wllmflower/tcn-vae-training/logs', exist_ok=True)
    with open('/home/wllmflower/tcn-vae-training/logs/overnight_training.jsonl', 'w') as f:
        f.write("")  # Clear file
    
    # Training loop
    start_time = time.time()
    
    try:
        for epoch in range(500):  # Long overnight run
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = trainer.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            trainer.epoch_times.append(epoch_time)
            
            # Update scheduler
            trainer.scheduler.step()
            
            # Log progress
            trainer.log_progress(epoch, train_loss, train_acc, val_loss, val_acc, epoch_time)
            
            # Save best model
            if val_acc > trainer.best_val_accuracy:
                trainer.best_val_accuracy = val_acc
                trainer.patience_counter = 0
                
                # Save best model
                torch.save(model.state_dict(), '/home/wllmflower/tcn-vae-training/models/best_overnight_v2_tcn_vae.pth')
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_val_accuracy': trainer.best_val_accuracy,
                    'timestamp': datetime.now().isoformat()
                }
                torch.save(checkpoint, '/home/wllmflower/tcn-vae-training/models/best_checkpoint_overnight_v2.pth')
                
            else:
                trainer.patience_counter += 1
            
            # Save periodic checkpoints
            if (epoch + 1) % 25 == 0:
                torch.save(model.state_dict(), f'/home/wllmflower/tcn-vae-training/models/checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping check
            if trainer.patience_counter >= trainer.patience:
                print(f"\n⏹️ Early stopping after {trainer.patience} epochs without improvement")
                break
                
            # Estimate completion time
            if epoch >= 10:
                avg_epoch_time = np.mean(trainer.epoch_times[-10:])
                remaining_epochs = 500 - (epoch + 1)
                eta_seconds = remaining_epochs * avg_epoch_time
                eta_hours = eta_seconds / 3600
                print(f"⏱️ ETA: {eta_hours:.1f} hours")
    
    except Exception as e:
        print(f"❌ Error during training: {e}")
        # Still save current best model
        torch.save(model.state_dict(), '/home/wllmflower/tcn-vae-training/models/error_recovery_model.pth')
    
    # Final save
    total_time = time.time() - start_time
    torch.save(model.state_dict(), '/home/wllmflower/tcn-vae-training/models/final_overnight_v2_tcn_vae.pth')
    
    with open('/home/wllmflower/tcn-vae-training/models/overnight_v2_processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    
    print(f"\n🏁 Training Session Complete!")
    print(f"⏱️ Total time: {total_time/3600:.2f} hours")
    print(f"🎯 Best validation accuracy: {trainer.best_val_accuracy:.4f}")
    improvement_pct = TrainingConfig.calculate_improvement(trainer.best_val_accuracy, TrainingConfig.CURRENT_BEST)
    print(f"📈 Improvement: {improvement_pct:+.1f}%")
    
    return trainer.best_val_accuracy


if __name__ == "__main__":
    main()