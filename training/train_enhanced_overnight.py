#!/usr/bin/env python3
"""
Enhanced Overnight TCN-VAE Training with WISDM + HAPT Integration
Target: Beat 86.53% validation accuracy with expanded dataset diversity
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.tcn_vae import TCNVAE
from preprocessing.enhanced_pipeline import EnhancedMultiDatasetHAR
from config.training_config import TrainingConfig


class EnhancedOvernightTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
        # Configure base paths
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        self.logs_dir = os.path.join(self.project_root, 'logs')
        self.models_dir = os.path.join(self.project_root, 'models')
        
        # Enhanced optimizer for multi-dataset training
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=2e-4,  # More conservative for diverse data
            weight_decay=2e-4,  # Stronger regularization
            betas=(0.9, 0.999)
        )
        
        # Scheduler with longer cycles for complex data
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=25, T_mult=2
        )
        
        # Loss functions optimized for transitions
        self.reconstruction_loss = nn.MSELoss()
        self.activity_loss = nn.CrossEntropyLoss(label_smoothing=0.15)  # More smoothing
        self.domain_loss = nn.CrossEntropyLoss()
        
        # Enhanced loss balancing for multi-dataset diversity
        self.beta = 0.3  # Lower KL weight for stability
        self.lambda_act = 3.5  # Higher activity focus for more classes
        self.lambda_dom = 0.03  # Lower domain confusion for diversity
        
        # Training state - use configuration for baseline
        self.best_val_accuracy = TrainingConfig.get_baseline('enhanced')
        self.patience = TrainingConfig.OVERNIGHT_PATIENCE  # Configurable patience
        self.patience_counter = 0
        self.epoch_times = []
        
        # Enhanced metrics tracking
        self.transition_classes = ['sit_to_stand', 'stand_to_sit', 'sit_to_down', 
                                 'down_to_sit', 'stand_to_down', 'down_to_stand']
        
    def vae_loss(self, recon_x, x, mu, logvar, epoch=0):
        recon_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Progressive beta annealing - slower for complex data
        beta = self.beta * min(1.0, epoch / 30)
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct_activities = 0
        total_samples = 0
        
        # Enhanced data augmentation
        augment_prob = 0.3 if epoch > 10 else 0.1
        
        for batch_idx, (data, activity_labels, domain_labels) in enumerate(train_loader):
            data = data.to(self.device).float()
            activity_labels = activity_labels.to(self.device).long()
            domain_labels = domain_labels.to(self.device).long()
            
            # Enhanced data augmentation for multi-dataset robustness
            if np.random.random() < augment_prob:
                # Sensor noise injection
                data += torch.randn_like(data) * 0.015
                
                # Time warping simulation (slight temporal shifts)
                if np.random.random() < 0.2:
                    shift = np.random.randint(-2, 3)
                    if shift != 0:
                        data = torch.roll(data, shift, dims=1)
            
            # Progressive domain adaptation - slower ramp for diversity
            p = float(batch_idx + epoch * len(train_loader)) / (50 * len(train_loader))
            alpha = 2. / (1. + np.exp(-8 * p)) - 1
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                recon_x, mu, logvar, activity_logits, domain_logits = self.model(data, alpha)
                
                # Compute losses
                vae_loss, recon_loss, kl_loss = self.vae_loss(recon_x, data, mu, logvar, epoch)
                act_loss = self.activity_loss(activity_logits, activity_labels)
                dom_loss = self.domain_loss(domain_logits, domain_labels)
                
                # Enhanced loss weighting that increases activity focus over time
                activity_weight = self.lambda_act * (1 + epoch / 300)
                domain_weight = self.lambda_dom * max(0.5, 1 - epoch / 200)  # Reduce domain focus over time
                
                total_batch_loss = vae_loss + activity_weight * act_loss + domain_weight * dom_loss
                
                # Enhanced gradient clipping
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
                
                # Track accuracy
                pred_activities = activity_logits.argmax(dim=1)
                correct_activities += (pred_activities == activity_labels).sum().item()
                total_samples += activity_labels.size(0)
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
                
            if batch_idx % 50 == 0:  # More frequent logging for complex training
                current_acc = correct_activities / total_samples if total_samples > 0 else 0
                print(f'Epoch {epoch+1}, Batch {batch_idx}: '
                      f'Loss: {total_batch_loss.item():.4f}, '
                      f'Train Acc: {current_acc:.4f}, '
                      f'Recon: {recon_loss.item():.4f}, '
                      f'Act: {act_loss.item():.4f}, '
                      f'Dom: {dom_loss.item():.4f}, '
                      f'LR: {self.scheduler.get_last_lr()[0]:.6f}')
        
        train_accuracy = correct_activities / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, train_accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct_activities = 0
        total_samples = 0
        
        # Per-class accuracy tracking
        class_correct = {}
        class_total = {}
        
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
                
                # Per-class tracking
                for i in range(len(activity_labels)):
                    label = activity_labels[i].item()
                    pred = pred_activities[i].item()
                    
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                    
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_activities / total_samples
        
        # Print per-class performance for insights
        if len(class_correct) > 0:
            print("\nPer-class accuracy:")
            for class_id in sorted(class_correct.keys()):
                if class_total[class_id] > 0:
                    acc = class_correct[class_id] / class_total[class_id]
                    print(f"  Class {class_id}: {acc:.3f} ({class_correct[class_id]}/{class_total[class_id]})")
        
        return avg_val_loss, val_accuracy
    
    def log_progress(self, epoch, train_loss, train_acc, val_loss, val_acc, epoch_time):
        """Enhanced logging for multi-dataset training"""
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
            "is_best": val_acc > self.best_val_accuracy,
            "datasets": "pamap2+uci_har+wisdm+hapt+tartan",
            "enhanced_pipeline": True
        }
        
        # Append to log file
        os.makedirs(self.logs_dir, exist_ok=True)
        with open(os.path.join(self.logs_dir, 'enhanced_training.jsonl'), 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Enhanced console output
        improvement = "🔥 NEW BEST!" if val_acc > self.best_val_accuracy else ""
        target_progress_pct = TrainingConfig.calculate_target_progress(val_acc, TrainingConfig.CURRENT_BEST)
        target_progress = f"({target_progress_pct:.1f}% of {TrainingConfig.CURRENT_BEST*100:.1f}% target)"
        
        print(f"\n[{timestamp}] Epoch {epoch+1} Complete:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f} {improvement}")
        print(f"  Progress: {target_progress}, Best: {self.best_val_accuracy:.4f}")
        print(f"  Time: {epoch_time:.1f}s, LR: {self.scheduler.get_last_lr()[0]:.6f}")


def main():
    print("🌙 Starting Enhanced Overnight TCN-VAE Training...")
    print(f"⏰ Started at: {datetime.now()}")
    print("🎯 Target: Beat 86.53% with WISDM+HAPT integration")
    print("📊 Datasets: PAMAP2 + UCI-HAR + WISDM + HAPT + TartanIMU")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
    
    # Load enhanced data
    print("\\n📊 Loading enhanced multi-dataset pipeline...")
    processor = EnhancedMultiDatasetHAR(window_size=100, overlap=0.5)
    X_train, y_train, domains_train, X_val, y_val, domains_val = processor.preprocess_all_enhanced()
    
    num_activities = len(np.unique(y_train))
    num_domains = len(np.unique(domains_train))
    print(f"📈 Activities: {num_activities}, Domains: {num_domains}")
    
    # Create data loaders
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
    
    # Enhanced batch size for multi-dataset training
    batch_size = 48 if device.type == 'cuda' else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=2, pin_memory=True)
    
    # Enhanced model architecture
    model = TCNVAE(input_dim=9, hidden_dims=[64, 128, 256, 128], latent_dim=64, 
                   num_activities=num_activities)
    trainer = EnhancedOvernightTrainer(model, device)
    
    print(f"🔢 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Clear log file
    with open(os.path.join(trainer.logs_dir, 'enhanced_training.jsonl'), 'w') as f:
        f.write("")
    
    # Enhanced training loop
    start_time = time.time()
    
    try:
        for epoch in range(400):  # Extended for complex multi-dataset learning
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
                
                # Save best model with enhanced naming
                torch.save(model.state_dict(), os.path.join(trainer.models_dir, 'best_enhanced_tcn_vae.pth'))
                
                # Enhanced checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_val_accuracy': trainer.best_val_accuracy,
                    'datasets': 'pamap2+uci_har+wisdm+hapt+tartan',
                    'num_activities': num_activities,
                    'enhanced_pipeline': True,
                    'timestamp': datetime.now().isoformat()
                }
                torch.save(checkpoint, os.path.join(trainer.models_dir, 'best_enhanced_checkpoint.pth'))
                
                print(f"🎯 New best model saved! Accuracy: {trainer.best_val_accuracy:.4f}")
                
            else:
                trainer.patience_counter += 1
            
            # Periodic saves
            if (epoch + 1) % 50 == 0:
                torch.save(model.state_dict(), os.path.join(trainer.models_dir, f'enhanced_epoch_{epoch+1}.pth'))
            
            # Early stopping
            if trainer.patience_counter >= trainer.patience:
                print(f"\n⏹️ Early stopping after {trainer.patience} epochs without improvement")
                break
                
            # ETA estimation
            if epoch >= 10:
                avg_epoch_time = np.mean(trainer.epoch_times[-10:])
                remaining_epochs = 400 - (epoch + 1)
                eta_seconds = remaining_epochs * avg_epoch_time
                eta_hours = eta_seconds / 3600
                print(f"⏱️ ETA: {eta_hours:.1f} hours")
    
    except Exception as e:
        print(f"❌ Error during training: {e}")
        torch.save(model.state_dict(), os.path.join(trainer.models_dir, 'enhanced_recovery_model.pth'))
    
    # Final saves
    total_time = time.time() - start_time
    torch.save(model.state_dict(), os.path.join(trainer.models_dir, 'final_enhanced_tcn_vae.pth'))
    
    with open(os.path.join(trainer.models_dir, 'enhanced_processor.pkl'), 'wb') as f:
        pickle.dump(processor, f)
    
    print(f"\n🏁 Enhanced Training Complete!")
    print(f"⏱️ Total time: {total_time/3600:.2f} hours")
    print(f"🎯 Best validation accuracy: {trainer.best_val_accuracy:.4f}")
    target_improvement = TrainingConfig.calculate_improvement(trainer.best_val_accuracy, TrainingConfig.CURRENT_BEST)
    print(f"📈 vs {TrainingConfig.CURRENT_BEST*100:.1f}% target: {target_improvement:+.1f}%")
    print(f"📊 Enhanced with WISDM + HAPT integration including transitions")
    
    return trainer.best_val_accuracy


if __name__ == "__main__":
    main()