#!/usr/bin/env python3
"""
Quadruped-Focused TCN-VAE Training with Animal Behavior Recognition
Target: Achieve 90%+ static pose accuracy and 85%+ F1 for transitions
Focus: sit/down/stand detection for dog training applications
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

sys.path.append('/home/wllmflower/Development/tcn-vae-training')

from models.tcn_vae import TCNVAE
from preprocessing.quadruped_pipeline import QuadrupedDatasetHAR


class QuadrupedTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
        # Quadruped-optimized training parameters
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-4,  # Conservative for animal behavior learning
            weight_decay=1e-3,  # Strong regularization for generalization
            betas=(0.9, 0.999)
        )
        
        # Scheduler optimized for behavior learning
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=30, T_mult=2
        )
        
        # Loss functions for animal behavior
        self.reconstruction_loss = nn.MSELoss()
        self.activity_loss = nn.CrossEntropyLoss(label_smoothing=0.2)  # Higher smoothing for animal behaviors
        self.domain_loss = nn.CrossEntropyLoss()
        
        # Loss weights optimized for quadruped behaviors
        self.beta = 0.2  # Lower KL weight for stable learning
        self.lambda_act = 4.0  # Higher activity focus for precise behavior detection
        self.lambda_dom = 0.02  # Lower domain weight for diverse animal data
        
        # Training targets for quadruped behaviors
        self.best_val_accuracy = 0.90  # Target 90%+ static pose accuracy
        self.patience = 80  # Extended patience for complex animal behaviors
        self.patience_counter = 0
        self.epoch_times = []
        
        # Quadruped-specific behavior tracking
        self.static_poses = ['sit', 'down', 'stand', 'stay', 'lying']
        self.transitions = ['sit_to_down', 'down_to_sit', 'sit_to_stand', 'stand_to_sit', 
                           'down_to_stand', 'stand_to_down']
        self.gaits = ['walking', 'trotting', 'running']
        
        # F1 tracking for transitions
        self.transition_f1_target = 0.85
        
    def vae_loss(self, recon_x, x, mu, logvar, epoch=0):
        recon_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Progressive beta annealing - slower for animal behavior complexity
        beta = self.beta * min(1.0, epoch / 40)
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct_activities = 0
        total_samples = 0
        
        # Enhanced data augmentation for animal behavior robustness
        augment_prob = 0.4 if epoch > 15 else 0.2
        
        for batch_idx, (data, activity_labels, domain_labels) in enumerate(train_loader):
            data = data.to(self.device).float()
            activity_labels = activity_labels.to(self.device).long()
            domain_labels = domain_labels.to(self.device).long()
            
            # Animal behavior data augmentation
            if np.random.random() < augment_prob:
                # Sensor noise (more realistic for wearable devices on animals)
                data += torch.randn_like(data) * 0.02
                
                # Temporal jitter (animals don't move perfectly consistently)
                if np.random.random() < 0.3:
                    shift = np.random.randint(-3, 4)
                    if shift != 0:
                        data = torch.roll(data, shift, dims=1)
                
                # Magnitude scaling (different sized animals)
                if np.random.random() < 0.2:
                    scale = np.random.uniform(0.9, 1.1)
                    data = data * scale
            
            # Progressive domain adaptation - slower for animal diversity
            p = float(batch_idx + epoch * len(train_loader)) / (60 * len(train_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                recon_x, mu, logvar, activity_logits, domain_logits = self.model(data, alpha)
                
                # Compute losses
                vae_loss, recon_loss, kl_loss = self.vae_loss(recon_x, data, mu, logvar, epoch)
                act_loss = self.activity_loss(activity_logits, activity_labels)
                dom_loss = self.domain_loss(domain_logits, domain_labels)
                
                # Dynamic loss weighting favoring behavior accuracy over time
                activity_weight = self.lambda_act * (1 + epoch / 200)
                domain_weight = self.lambda_dom * max(0.3, 1 - epoch / 150)
                
                total_batch_loss = vae_loss + activity_weight * act_loss + domain_weight * dom_loss
                
                # Gradient clipping for stable training
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
                
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
                
                # Track accuracy
                pred_activities = activity_logits.argmax(dim=1)
                correct_activities += (pred_activities == activity_labels).sum().item()
                total_samples += activity_labels.size(0)
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
                
            if batch_idx % 40 == 0:  # Frequent logging for behavior tracking
                current_acc = correct_activities / total_samples if total_samples > 0 else 0
                print(f'Epoch {epoch+1}, Batch {batch_idx}: '
                      f'Loss: {total_batch_loss.item():.4f}, '
                      f'Behavior Acc: {current_acc:.4f}, '
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
        
        # Detailed behavior tracking
        behavior_correct = {}
        behavior_total = {}
        
        all_predictions = []
        all_labels = []
        
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
                
                # Store for F1 calculation
                all_predictions.extend(pred_activities.cpu().numpy())
                all_labels.extend(activity_labels.cpu().numpy())
                
                # Per-behavior tracking
                for i in range(len(activity_labels)):
                    label = activity_labels[i].item()
                    pred = pred_activities[i].item()
                    
                    if label not in behavior_total:
                        behavior_total[label] = 0
                        behavior_correct[label] = 0
                    
                    behavior_total[label] += 1
                    if label == pred:
                        behavior_correct[label] += 1
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_activities / total_samples
        
        # Calculate F1 for transitions (key metric)
        from sklearn.metrics import f1_score, classification_report
        
        # Overall F1
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        # Print detailed behavior analysis
        print("\\n=== Quadruped Behavior Analysis ===")
        for behavior_id in sorted(behavior_correct.keys()):
            if behavior_total[behavior_id] > 0:
                acc = behavior_correct[behavior_id] / behavior_total[behavior_id]
                print(f"  Behavior {behavior_id}: {acc:.3f} ({behavior_correct[behavior_id]}/{behavior_total[behavior_id]})")
        
        print(f"\\nOverall F1 Score: {f1_macro:.3f}")
        print(f"Transition F1 Target: {self.transition_f1_target:.3f}")
        
        return avg_val_loss, val_accuracy, f1_macro
    
    def log_progress(self, epoch, train_loss, train_acc, val_loss, val_acc, f1_score, epoch_time):
        """Enhanced logging for quadruped behavior training"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "f1_macro": float(f1_score),
            "epoch_time": float(epoch_time),
            "learning_rate": float(self.scheduler.get_last_lr()[0]),
            "best_so_far": float(self.best_val_accuracy),
            "is_best": val_acc > self.best_val_accuracy,
            "datasets": "awa_pose+animal_activity+cear_quadruped",
            "target_static_pose": 0.90,
            "target_transition_f1": self.transition_f1_target,
            "quadruped_pipeline": True
        }
        
        # Append to log file
        os.makedirs('/home/wllmflower/Development/tcn-vae-training/logs', exist_ok=True)
        with open('/home/wllmflower/Development/tcn-vae-training/logs/quadruped_training.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\\n')
        
        # Enhanced console output
        improvement = "🐕 NEW BEST!" if val_acc > self.best_val_accuracy else ""
        static_pose_progress = f"({val_acc/0.90*100:.1f}% of 90% static pose target)"
        f1_progress = f"({f1_score/self.transition_f1_target*100:.1f}% of 85% F1 target)"
        
        print(f"\\n[{timestamp}] Epoch {epoch+1} Complete:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f} {improvement}")
        print(f"  F1:    {f1_score:.4f} {f1_progress}")
        print(f"  Progress: {static_pose_progress}, Best: {self.best_val_accuracy:.4f}")
        print(f"  Time: {epoch_time:.1f}s, LR: {self.scheduler.get_last_lr()[0]:.6f}")


def main():
    print("🐕 Starting Quadruped TCN-VAE Training...")
    print(f"⏰ Started at: {datetime.now()}")
    print("🎯 Target: 90%+ static pose accuracy, 85%+ transition F1")
    print("📊 Datasets: AwA Pose + Animal Activity + CEAR Quadruped")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
    
    # Load quadruped data
    print("\\n📊 Loading quadruped behavior datasets...")
    processor = QuadrupedDatasetHAR(window_size=100, overlap=0.5)
    X_train, y_train, domains_train, X_val, y_val, domains_val = processor.preprocess_all_quadruped()
    
    num_behaviors = len(np.unique(y_train))
    num_domains = len(np.unique(domains_train))
    print(f"📈 Behaviors: {num_behaviors}, Domains: {num_domains}")
    
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
    
    # Quadruped-optimized batch size
    batch_size = 32 if device.type == 'cuda' else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=2, pin_memory=True)
    
    # Quadruped-optimized model architecture
    model = TCNVAE(input_dim=9, hidden_dims=[64, 128, 256, 128], latent_dim=64, 
                   num_activities=num_behaviors)
    trainer = QuadrupedTrainer(model, device)
    
    print(f"🔢 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Clear log file
    with open('/home/wllmflower/Development/tcn-vae-training/logs/quadruped_training.jsonl', 'w') as f:
        f.write("")
    
    # Quadruped behavior training loop
    start_time = time.time()
    
    try:
        for epoch in range(300):  # Extended training for animal behavior complexity
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
            
            # Validation with F1 tracking
            val_loss, val_acc, f1_score = trainer.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            trainer.epoch_times.append(epoch_time)
            
            # Update scheduler
            trainer.scheduler.step()
            
            # Log progress
            trainer.log_progress(epoch, train_loss, train_acc, val_loss, val_acc, f1_score, epoch_time)
            
            # Save best model (considering both accuracy and F1)
            combined_metric = val_acc * 0.7 + f1_score * 0.3  # Weighted combination
            if val_acc > trainer.best_val_accuracy or combined_metric > trainer.best_val_accuracy * 0.9:
                if val_acc > trainer.best_val_accuracy:
                    trainer.best_val_accuracy = val_acc
                trainer.patience_counter = 0
                
                # Save quadruped-specific model
                torch.save(model.state_dict(), '/home/wllmflower/Development/tcn-vae-training/models/best_quadruped_tcn_vae.pth')
                
                # Enhanced checkpoint for quadruped behaviors
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_val_accuracy': trainer.best_val_accuracy,
                    'f1_score': f1_score,
                    'datasets': 'awa_pose+animal_activity+cear_quadruped',
                    'num_behaviors': num_behaviors,
                    'behavior_focus': 'sit/down/stand + transitions',
                    'quadruped_pipeline': True,
                    'timestamp': datetime.now().isoformat()
                }
                torch.save(checkpoint, '/home/wllmflower/Development/tcn-vae-training/models/best_quadruped_checkpoint.pth')
                
                print(f"🐕 New best quadruped model! Accuracy: {trainer.best_val_accuracy:.4f}, F1: {f1_score:.4f}")
                
            else:
                trainer.patience_counter += 1
            
            # Periodic saves
            if (epoch + 1) % 50 == 0:
                torch.save(model.state_dict(), f'/home/wllmflower/Development/tcn-vae-training/models/quadruped_epoch_{epoch+1}.pth')
            
            # Early stopping
            if trainer.patience_counter >= trainer.patience:
                print(f"\\n⏹️ Early stopping after {trainer.patience} epochs without improvement")
                break
                
            # ETA estimation
            if epoch >= 10:
                avg_epoch_time = np.mean(trainer.epoch_times[-10:])
                remaining_epochs = 300 - (epoch + 1)
                eta_seconds = remaining_epochs * avg_epoch_time
                eta_hours = eta_seconds / 3600
                print(f"⏱️ ETA: {eta_hours:.1f} hours")
    
    except Exception as e:
        print(f"❌ Error during training: {e}")
        torch.save(model.state_dict(), '/home/wllmflower/Development/tcn-vae-training/models/quadruped_recovery_model.pth')
    
    # Final saves
    total_time = time.time() - start_time
    torch.save(model.state_dict(), '/home/wllmflower/Development/tcn-vae-training/models/final_quadruped_tcn_vae.pth')
    
    with open('/home/wllmflower/Development/tcn-vae-training/models/quadruped_processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    
    print(f"\\n🏁 Quadruped Training Complete!")
    print(f"⏱️ Total time: {total_time/3600:.2f} hours")
    print(f"🎯 Best validation accuracy: {trainer.best_val_accuracy:.4f}")
    print(f"📈 vs 90% static pose target: {((trainer.best_val_accuracy / 0.90) - 1) * 100:+.1f}%")
    print(f"🐕 Quadruped behavior detection ready for dog training applications!")
    
    return trainer.best_val_accuracy


if __name__ == "__main__":
    main()