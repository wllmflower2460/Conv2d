import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import time
import json
from datetime import datetime
sys.path.append('/home/wllmflower/tcn-vae-training')

from models.tcn_vae import TCNVAE
from preprocessing.unified_pipeline import MultiDatasetHAR

class ExtendedTCNVAETrainer:
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.activity_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss()
        
        # Loss weights - adjusted for better balance
        self.beta = 0.5          # VAE KL weight (reduced)
        self.lambda_act = 2.0    # Activity classification weight (increased)
        self.lambda_dom = 0.1    # Domain adaptation weight
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # Normalize by batch size
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_act_loss = 0
        total_dom_loss = 0
        
        for batch_idx, (data, activity_labels, domain_labels) in enumerate(train_loader):
            data = data.to(self.device).float()
            activity_labels = activity_labels.to(self.device).long()
            domain_labels = domain_labels.to(self.device).long()
            
            # Progressive adversarial training
            p = float(batch_idx + epoch * len(train_loader)) / (20 * len(train_loader))  # Slower progression
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_x, mu, logvar, activity_logits, domain_logits = self.model(data, alpha)
            
            # Compute losses
            vae_loss, recon_loss, kl_loss = self.vae_loss(recon_x, data, mu, logvar)
            act_loss = self.activity_loss(activity_logits, activity_labels)
            dom_loss = self.domain_loss(domain_logits, domain_labels)
            
            # Combined loss with adaptive weights
            epoch_factor = min(1.0, epoch / 50)  # Gradually increase activity focus
            adapted_lambda_act = self.lambda_act * (1 + epoch_factor)
            
            total_batch_loss = vae_loss + adapted_lambda_act * act_loss + self.lambda_dom * dom_loss
            
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_recon_loss += recon_loss.item()
            total_act_loss += act_loss.item()
            total_dom_loss += dom_loss.item()
            
            if batch_idx % 50 == 0:  # More frequent logging
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Total: {total_batch_loss.item():.4f}, '
                      f'Recon: {recon_loss.item():.4f}, '
                      f'Activity: {act_loss.item():.4f}, '
                      f'Domain: {dom_loss.item():.4f}, '
                      f'Alpha: {alpha:.3f}')
        
        avg_losses = {
            'total': total_loss / len(train_loader),
            'recon': total_recon_loss / len(train_loader),
            'activity': total_act_loss / len(train_loader),
            'domain': total_dom_loss / len(train_loader)
        }
        
        return avg_losses
    
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
                
                # Compute losses
                vae_loss, _, _ = self.vae_loss(recon_x, data, mu, logvar)
                act_loss = self.activity_loss(activity_logits, activity_labels)
                
                val_loss += (vae_loss + act_loss).item()
                
                # Activity accuracy
                pred_activities = activity_logits.argmax(dim=1)
                correct_activities += (pred_activities == activity_labels).sum().item()
                total_samples += activity_labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        activity_accuracy = correct_activities / total_samples
        
        return avg_val_loss, activity_accuracy
    
    def save_checkpoint(self, epoch, val_accuracy, save_path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_accuracy,
            'history': self.history
        }
        torch.save(checkpoint, save_path)

def main():
    print("=" * 60)
    print("EXTENDED TCN-VAE TRAINING SESSION")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    print("\nLoading and preprocessing datasets...")
    processor = MultiDatasetHAR(window_size=100, overlap=0.5)
    X_train, y_train, domains_train, X_val, y_val, domains_val = processor.preprocess_all()
    
    # Get number of unique activities for model
    num_activities = len(np.unique(y_train))
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of activity classes: {num_activities}")
    
    # Create data loaders with larger batch size for better GPU utilization
    batch_size = 64 if torch.cuda.is_available() else 32
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model and trainer
    model = TCNVAE(input_dim=9, hidden_dims=[64, 128, 256], latent_dim=64, num_activities=num_activities)
    trainer = ExtendedTCNVAETrainer(model, device, learning_rate=1e-3)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Extended training loop - 200 epochs for overnight training
    num_epochs = 200
    best_val_accuracy = 0
    patience_counter = 0
    early_stop_patience = 20
    
    print(f"\nStarting extended training for {num_epochs} epochs...")
    print("Progress will be saved every 10 epochs")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
        
        # Training
        train_losses = trainer.train_epoch(train_loader, epoch)
        
        # Validation
        val_loss, val_accuracy = trainer.validate(val_loader)
        
        # Update scheduler
        trainer.scheduler.step(val_accuracy)
        
        # Update history
        trainer.history['train_loss'].append(train_losses['total'])
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_accuracy'].append(val_accuracy)
        trainer.history['learning_rate'].append(trainer.optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Train Loss: {train_losses["total"]:.4f} (Recon: {train_losses["recon"]:.4f}, Act: {train_losses["activity"]:.4f})')
        print(f'  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print(f'  Learning Rate: {trainer.optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Epoch Time: {epoch_time:.1f}s, Total Time: {elapsed_time/3600:.1f}h')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            torch.save(model.state_dict(), '/home/wllmflower/tcn-vae-training/models/best_extended_tcn_vae.pth')
            trainer.save_checkpoint(epoch, val_accuracy, '/home/wllmflower/tcn-vae-training/models/best_checkpoint.pth')
            print(f"  *** NEW BEST MODEL SAVED! Accuracy: {best_val_accuracy:.4f} ***")
        else:
            patience_counter += 1
        
        # Periodic saves
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch, val_accuracy, f'/home/wllmflower/tcn-vae-training/models/checkpoint_epoch_{epoch+1}.pth')
            
            # Save training history
            with open('/home/wllmflower/tcn-vae-training/models/training_history.json', 'w') as f:
                json.dump(trainer.history, f, indent=2)
            
            print(f"  Checkpoint saved at epoch {epoch+1}")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {early_stop_patience} epochs without improvement")
            break
        
        # Progress estimate
        if epoch > 0:
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = num_epochs - epoch - 1
            estimated_finish = datetime.fromtimestamp(time.time() + remaining_epochs * avg_epoch_time)
            print(f"  Estimated finish: {estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f'Best validation accuracy: {best_val_accuracy:.4f}')
    print(f'Total training time: {total_time/3600:.2f} hours')
    print(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Save final model and artifacts
    torch.save(model.state_dict(), '/home/wllmflower/tcn-vae-training/models/final_extended_tcn_vae.pth')
    trainer.save_checkpoint(epoch, val_accuracy, '/home/wllmflower/tcn-vae-training/models/final_checkpoint.pth')
    
    # Save preprocessing artifacts
    import pickle
    with open('/home/wllmflower/tcn-vae-training/models/processor_extended.pkl', 'wb') as f:
        pickle.dump(processor, f)
    
    # Save final training history
    with open('/home/wllmflower/tcn-vae-training/models/final_training_history.json', 'w') as f:
        json.dump(trainer.history, f, indent=2)
    
    print("All artifacts saved successfully!")
    print(f"Best model: /home/wllmflower/tcn-vae-training/models/best_extended_tcn_vae.pth")

if __name__ == "__main__":
    main()