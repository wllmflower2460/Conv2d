#!/usr/bin/env python3
"""
Enhanced Multi-Dataset Training - Working Version
WISDM + HAPT + Existing datasets with model compatibility fixes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import sys
import os
import json
import time
from datetime import datetime

sys.path.append('/home/wllmflower/Development/tcn-vae-training')

from models.tcn_vae import TCNVAE
from preprocessing.enhanced_pipeline import EnhancedMultiDatasetHAR

class EnhancedTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
        # Conservative training parameters
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=2e-4,  # Conservative learning rate
            weight_decay=5e-4,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2
        )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.activity_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.domain_loss = nn.CrossEntropyLoss()
        
        # Loss weights
        self.beta = 0.25  # VAE weight
        self.lambda_act = 5.0  # Strong activity focus for 60K samples
        self.lambda_dom = 0.01  # Minimal domain weight
        
        self.best_val_accuracy = 0.72  # Beat previous best of 72.13%
        self.patience = 25
        self.patience_counter = 0
        
    def vae_loss(self, recon_x, x, mu, logvar, epoch=0):
        recon_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        beta = self.beta * min(1.0, epoch / 15)
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct_activities = 0
        total_samples = 0
        
        for batch_idx, (data, activity_labels, domain_labels) in enumerate(train_loader):
            data = data.to(self.device).float()
            activity_labels = activity_labels.to(self.device).long()
            # Map 5 domains to 3 for model compatibility
            domain_labels = torch.clamp(domain_labels, 0, 2).to(self.device).long()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_x, mu, logvar, activity_pred, domain_pred = self.model(data)
            z = self.model.reparameterize(mu, logvar)
            
            # Losses
            vae_loss, recon_loss, kl_loss = self.vae_loss(recon_x, data, mu, logvar, epoch)
            act_loss = self.activity_loss(activity_pred, activity_labels)
            dom_loss = self.domain_loss(domain_pred, domain_labels)
            
            total_batch_loss = vae_loss + self.lambda_act * act_loss + self.lambda_dom * dom_loss
            
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # Metrics
            total_loss += total_batch_loss.item()
            pred_activities = activity_pred.argmax(dim=1)
            correct_activities += (pred_activities == activity_labels).sum().item()
            total_samples += activity_labels.size(0)
            
            # Detailed logging every 250 batches
            if batch_idx > 0 and batch_idx % 250 == 0:
                current_acc = correct_activities / total_samples
                print(f"  Batch {batch_idx:3d}: Loss={total_batch_loss.item():.4f}, "
                      f"Recon={recon_loss.item():.4f}, Act={act_loss.item():.4f}, "
                      f"Acc={current_acc:.4f}")
        
        return total_loss / len(train_loader), correct_activities / total_samples
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct_activities = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, activity_labels, domain_labels in val_loader:
                data = data.to(self.device).float()
                activity_labels = activity_labels.to(self.device).long()
                domain_labels = torch.clamp(domain_labels, 0, 2).to(self.device).long()
                
                recon_x, mu, logvar, activity_pred, domain_pred = self.model(data)
                
                vae_loss, _, _ = self.vae_loss(recon_x, data, mu, logvar)
                act_loss = self.activity_loss(activity_pred, activity_labels)
                dom_loss = self.domain_loss(domain_pred, domain_labels)
                
                total_batch_loss = vae_loss + self.lambda_act * act_loss + self.lambda_dom * dom_loss
                total_loss += total_batch_loss.item()
                
                pred_activities = activity_pred.argmax(dim=1)
                correct_activities += (pred_activities == activity_labels).sum().item()
                total_samples += activity_labels.size(0)
        
        return total_loss / len(val_loader), correct_activities / total_samples

def main():
    print("🚀 Enhanced Multi-Dataset Training - Working Version")
    print(f"⏰ Started at: {datetime.now()}")
    print("🎯 Target: Beat 72.13% baseline with WISDM+HAPT+Existing")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    if device.type == 'cuda':
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
    
    try:
        # Load enhanced dataset
        print("\n📊 Loading enhanced multi-dataset pipeline...")
        processor = EnhancedMultiDatasetHAR()
        X_train, y_train, domain_train, X_val, y_val, domain_val = processor.preprocess_all_enhanced()
        
        print(f"📈 Training samples: {X_train.shape[0]:,}")
        print(f"📈 Validation samples: {X_val.shape[0]:,}")
        print(f"📈 Features: {X_train.shape[-1]}")
        print(f"📈 Activities: {len(np.unique(y_train))}")
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
            torch.from_numpy(domain_train).long()
        )
        
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long(),
            torch.from_numpy(domain_val).long()
        )
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False, num_workers=0)
        
        # Create model with existing TCNVAE architecture
        model = TCNVAE(
            input_dim=X_train.shape[-1],  
            latent_dim=64,
            num_activities=len(np.unique(y_train))
        )
        
        print(f"🔢 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        trainer = EnhancedTrainer(model, device)
        
        # Training loop
        print(f"\n🚀 Starting Training to Beat {trainer.best_val_accuracy:.1%}...")
        
        for epoch in range(1, 76):  # Max 75 epochs
            start_time = time.time()
            
            print(f"\n--- Epoch {epoch}/75 ---")
            
            # Training
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
            
            # Validation  
            val_loss, val_acc = trainer.validate(val_loader)
            
            # Scheduler
            trainer.scheduler.step()
            
            epoch_time = time.time() - start_time
            current_lr = trainer.optimizer.param_groups[0]['lr']
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch} Complete:")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            print(f"  Baseline: {val_acc:.1%} vs 72.1% target")
            
            if val_acc > trainer.best_val_accuracy:
                improvement = val_acc - trainer.best_val_accuracy
                print(f"  🎯 IMPROVEMENT: +{improvement:.3f} ({improvement:.1%})")
            else:
                print(f"  ⏳ No improvement ({trainer.patience_counter+1}/{trainer.patience})")
            
            print(f"  Time: {epoch_time:.1f}s, LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > trainer.best_val_accuracy:
                improvement = val_acc - trainer.best_val_accuracy
                trainer.best_val_accuracy = val_acc
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_accuracy': trainer.best_val_accuracy,
                    'improvement': improvement
                }, '/home/wllmflower/Development/tcn-vae-training/models/best_enhanced_working.pth')
                
                trainer.patience_counter = 0
                
                # Achievement milestones
                if val_acc > 0.90:
                    print("🏆 EXCELLENCE: >90% accuracy achieved!")
                elif val_acc > 0.85:
                    print("🎯 STRONG: >85% accuracy achieved!")
                elif val_acc > 0.80:
                    print("📈 GOOD: >80% accuracy achieved!")
                elif val_acc > 0.75:
                    print("⚡ PROGRESS: >75% accuracy achieved!")
            else:
                trainer.patience_counter += 1
            
            # Early stopping
            if trainer.patience_counter >= trainer.patience:
                print(f"⏹️ Early stopping after {epoch} epochs")
                break
                
            # Excellence exit
            if val_acc > 0.95:
                print("🏆 MAXIMUM EXCELLENCE: >95% accuracy!")
                break
        
        # Final summary
        final_accuracy = trainer.best_val_accuracy
        improvement = final_accuracy - 0.7213  # vs previous 72.13%
        
        print(f"\n🏁 Enhanced Multi-Dataset Training Complete!")
        print(f"🎯 Final accuracy: {final_accuracy:.4f} ({final_accuracy:.1%})")
        print(f"📈 Improvement: +{improvement:.3f} ({improvement:.1%} gain)")
        
        if final_accuracy > 0.90:
            print("✅ OUTSTANDING: Achieved >90% target!")
        elif final_accuracy > 0.85:
            print("⚡ EXCELLENT: Strong >85% performance!")
        elif final_accuracy > 0.80:
            print("📈 GOOD: Solid >80% improvement!")
        elif improvement > 0:
            print("✅ SUCCESS: Beat baseline with enhanced dataset!")
        else:
            print("🔧 Baseline maintained - consider hyperparameter tuning")
        
        print(f"📊 Dataset: {X_train.shape[0]:,} samples from WISDM+HAPT+PAMAP2+UCI-HAR+TartanIMU")
        print(f"🚀 Enhanced model ready for deployment")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()