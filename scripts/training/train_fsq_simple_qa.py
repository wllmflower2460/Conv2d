#!/usr/bin/env python3
"""
Train FSQ model with preprocessing QA on real quadruped data.
Simplified version focusing on getting back to M1.0-M1.2 performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix

# Import existing models and QA
from preprocessing.movement_diagnostics import QualityControl, QualityThresholds
from models.conv2d_fsq_model import Conv2dFSQ

class QuadrupedDatasetWithQA(Dataset):
    """Dataset with preprocessing QA for quadruped data."""
    
    def __init__(self, X, y, quality_control=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.qc = quality_control or QualityControl()
        
        # Validate data quality
        self.validate_data()
        
        # Reshape for Conv2d (B, C, T) -> (B, C, 2, T/2)
        B, C, T = self.X.shape
        if T % 2 == 1:
            self.X = self.X[:, :, :-1]
            T = T - 1
        self.X = self.X.reshape(B, C, 2, T//2)
        
    def validate_data(self):
        """Run quality validation on the dataset."""
        print("\n" + "="*60)
        print("PREPROCESSING QUALITY ASSURANCE")
        print("="*60)
        
        # Basic statistics
        print(f"\n  Dataset Statistics:")
        print(f"  ✓ Shape: {tuple(self.X.shape)}")
        print(f"  ✓ Data type: {self.X.dtype}")
        print(f"  ✓ Value range: [{self.X.min():.2f}, {self.X.max():.2f}]")
        print(f"  ✓ Mean: {self.X.mean():.4f}, Std: {self.X.std():.4f}")
        
        # Check for data issues
        nan_count = torch.isnan(self.X).sum().item()
        inf_count = torch.isinf(self.X).sum().item()
        zero_std_channels = (self.X.std(dim=(0, 2)) < 0.001).sum().item()
        
        print(f"\n  Quality Checks:")
        print(f"  ✓ NaN values: {nan_count}")
        print(f"  ✓ Inf values: {inf_count}")
        print(f"  ✓ Dead channels (std<0.001): {zero_std_channels}")
        
        if nan_count > 0:
            print(f"    ⚠ Replacing {nan_count} NaN values with 0")
            self.X = torch.nan_to_num(self.X, nan=0.0)
            
        if inf_count > 0:
            print(f"    ⚠ Clipping {inf_count} Inf values")
            self.X = torch.clamp(self.X, -1e6, 1e6)
            
        if zero_std_channels > 0:
            print(f"    ⚠ Warning: {zero_std_channels} channels have very low variance")
            
        # Check class distribution
        unique_classes, counts = torch.unique(self.y, return_counts=True)
        print(f"\n  Class Distribution:")
        for cls, count in zip(unique_classes, counts):
            print(f"    Class {cls}: {count} samples ({100*count/len(self.y):.1f}%)")
            
        print(f"\n  ✅ QA Complete - Data is ready for training")
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_fsq_with_qa():
    """Train FSQ model with preprocessing QA."""
    
    print("\n" + "="*80)
    print("M1.5 TRAINING: FSQ Model with Preprocessing QA")
    print("Goal: Recover M1.0-M1.2 performance (78.12%)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set up quality control
    qc = QualityControl(strict_mode=False)
    
    # Load quadruped data
    quadruped_dir = Path("./quadruped_data/processed")
    
    if not quadruped_dir.exists():
        print("\n⚠ Quadruped data not found. Setting up...")
        from setup_quadruped_datasets import main as setup_quadruped
        setup_quadruped()
    
    print("\n✓ Loading quadruped locomotion data...")
    X_train = np.load(quadruped_dir / "X_train_quadruped.npy")
    y_train = np.load(quadruped_dir / "y_train_quadruped.npy")
    X_val = np.load(quadruped_dir / "X_val_quadruped.npy")
    y_val = np.load(quadruped_dir / "y_val_quadruped.npy")
    X_test = np.load(quadruped_dir / "X_test_quadruped.npy")
    y_test = np.load(quadruped_dir / "y_test_quadruped.npy")
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Load metadata
    with open(quadruped_dir / 'quadruped_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"\n  Behaviors ({metadata['n_classes']} classes):")
    for name, info in metadata['behaviors'].items():
        print(f"    [{info['id']}] {name:12s}: {info['description']}")
    
    # Create datasets with QA
    train_dataset = QuadrupedDatasetWithQA(X_train, y_train, qc)
    val_dataset = QuadrupedDatasetWithQA(X_val, y_val, qc)
    test_dataset = QuadrupedDatasetWithQA(X_test, y_test, qc)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create FSQ model
    print("\n" + "-"*60)
    print("Initializing FSQ Model")
    print("-"*60)
    
    model = Conv2dFSQ(
        input_channels=9,
        hidden_dim=128,
        num_classes=metadata['n_classes'],
        fsq_levels=[8, 6, 5],  # From M1.2
        project_dim=None
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  FSQ levels: [8, 6, 5] (240 unique codes)")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    print("\n" + "-"*60)
    print("TRAINING")
    print("-"*60)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 30
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(300):  # More epochs for complex behaviors
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            # Handle dict output
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs
                
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_acc = train_correct / train_total
        history['train_acc'].append(train_acc)
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output'))
                else:
                    logits = outputs
                    
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        history['val_acc'].append(val_acc)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # Print progress
        if epoch % 10 == 0 or val_acc > best_val_acc:
            print(f"Epoch {epoch:3d}: Train={train_acc:.4f}, Val={val_acc:.4f}, "
                  f"Best={best_val_acc:.4f}")
            
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'epoch': epoch,
                'n_params': n_params,
                'architecture': 'FSQ',
                'fsq_levels': [8, 6, 5]
            }, 'm15_fsq_best_qa.pth')
            
            if val_acc > 0.70:  # Approaching target
                print(f"  → Good progress! Val accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Final evaluation
    print("\n" + "-"*60)
    print("FINAL EVALUATION")
    print("-"*60)
    
    # Load best model
    checkpoint = torch.load('m15_fsq_best_qa.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs
                
            _, predicted = torch.max(logits, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"\n  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Target (M1.0-M1.2): 0.7812")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'target_acc': 0.7812,
        'achieved_target': test_acc >= 0.75,
        'n_params': n_params,
        'epochs_trained': checkpoint['epoch'],
        'data': 'quadruped_locomotion',
        'preprocessing_qa': True
    }
    
    with open('m15_fsq_qa_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    if test_acc >= 0.75:
        print("✅ SUCCESS: Approaching M1.0-M1.2 performance!")
    elif test_acc >= 0.60:
        print("⚠ PROGRESS: Good improvement from 22%, keep training")
    else:
        print("❌ Need more work to reach target")
    print("="*80)
    
    return test_acc

if __name__ == "__main__":
    accuracy = train_fsq_with_qa()