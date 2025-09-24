import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
sys.path.append('/home/wllmflower/tcn-vae-training')

from models.tcn_vae import TCNVAE
from preprocessing.unified_pipeline import MultiDatasetHAR

class TCNVAETrainer:
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.activity_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss()
        
        # Loss weights
        self.beta = 1.0          # VAE KL weight
        self.lambda_act = 1.0    # Activity classification weight  
        self.lambda_dom = 0.1    # Domain adaptation weight
    
    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, activity_labels, domain_labels) in enumerate(train_loader):
            data = data.to(self.device).float()
            activity_labels = activity_labels.to(self.device).long()
            domain_labels = domain_labels.to(self.device).long()
            
            # Progressive adversarial training
            p = float(batch_idx + epoch * len(train_loader)) / (10 * len(train_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_x, mu, logvar, activity_logits, domain_logits = self.model(data, alpha)
            
            # Compute losses
            vae_loss, recon_loss, kl_loss = self.vae_loss(recon_x, data, mu, logvar)
            act_loss = self.activity_loss(activity_logits, activity_labels)
            dom_loss = self.domain_loss(domain_logits, domain_labels)
            
            # Combined loss
            total_batch_loss = vae_loss + self.lambda_act * act_loss + self.lambda_dom * dom_loss
            
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'VAE Loss: {vae_loss.item():.4f}, '
                      f'Activity Loss: {act_loss.item():.4f}, '
                      f'Domain Loss: {dom_loss.item():.4f}')
        
        return total_loss / len(train_loader)
    
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

def main():
    # Check if we can run without wandb for now
    print("Starting TCN-VAE training session...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading and preprocessing datasets...")
    processor = MultiDatasetHAR(window_size=100, overlap=0.5)
    X_train, y_train, domains_train, X_val, y_val, domains_val = processor.preprocess_all()
    
    # Get number of unique activities for model
    num_activities = len(np.unique(y_train))
    print(f"Number of activity classes: {num_activities}")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and trainer
    model = TCNVAE(input_dim=9, hidden_dims=[64, 128, 256], latent_dim=64, num_activities=num_activities)
    trainer = TCNVAETrainer(model, device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    best_val_accuracy = 0
    for epoch in range(50):
        print(f"\n--- Epoch {epoch+1}/50 ---")
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss, val_accuracy = trainer.validate(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), '/home/wllmflower/tcn-vae-training/models/best_tcn_vae.pth')
            print(f"New best model saved! Accuracy: {best_val_accuracy:.4f}")
    
    print(f'\nTraining completed. Best validation accuracy: {best_val_accuracy:.4f}')
    
    # Save final model and processor
    torch.save(model.state_dict(), '/home/wllmflower/tcn-vae-training/models/final_tcn_vae.pth')
    
    # Save preprocessing artifacts
    import pickle
    with open('/home/wllmflower/tcn-vae-training/models/processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    
    print("Training session completed successfully!")

if __name__ == "__main__":
    main()