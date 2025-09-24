#!/usr/bin/env python3
"""
Train FSQ+HSMM model with proper preprocessing QA on real quadruped data.
This combines M1.0-M1.2 architecture with proper data handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from torch.utils.data import DataLoader, Dataset

# Import existing preprocessing with QA
from preprocessing.movement_diagnostics import QualityControl, QualityThresholds
from preprocessing.enhanced_pipeline import EnhancedCrossSpeciesDataset
from models.conv2d_fsq_model import Conv2dFSQ
from models.hsmm_components import HSMM

class FSQHSMMModel(nn.Module):
    """FSQ+HSMM model that worked best in M1.2 ablation (100% accuracy)."""
    
    def __init__(self, input_channels=9, num_classes=10, levels=[8, 6, 5], hidden_dim=64):
        super().__init__()
        
        # FSQ component
        self.fsq = Conv2dFSQ(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            fsq_levels=levels,
            project_dim=None
        )
        
        # HSMM component for temporal dynamics
        self.hsmm = HSMM(
            input_dim=len(levels),  # FSQ code dimension
            num_states=num_classes,
            duration_type='negative_binomial'
        )
        
    def forward(self, x):
        # Get FSQ outputs
        fsq_out = self.fsq(x)
        
        # Extract logits and codes
        if isinstance(fsq_out, dict):
            logits = fsq_out.get('logits', fsq_out.get('output'))
            codes = fsq_out.get('codes', None)
        else:
            logits = fsq_out
            codes = None
            
        # Apply HSMM if we have codes
        if codes is not None:
            # Reshape codes for HSMM: (B, T, D)
            B, D, H, W = codes.shape
            codes_seq = codes.view(B, D, -1).permute(0, 2, 1)  # (B, T, D)
            
            # Get HSMM predictions
            hsmm_out = self.hsmm(codes_seq)
            if isinstance(hsmm_out, dict):
                hsmm_logits = hsmm_out.get('logits', logits)
            else:
                hsmm_logits = hsmm_out
                
            # Combine FSQ and HSMM predictions
            logits = 0.7 * logits + 0.3 * hsmm_logits.mean(dim=1)
            
        return logits

class QuadrupedDatasetWithQA(Dataset):
    """Dataset with preprocessing QA for quadruped data."""
    
    def __init__(self, X, y, quality_control=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.qc = quality_control or QualityControl()
        
        # Validate data quality
        self.validate_data()
        
        # Reshape for Conv2d
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
        
        # Sample validation (check first 100 samples)
        n_check = min(100, len(self.X))
        sample_indices = np.random.choice(len(self.X), n_check, replace=False)
        
        failures = []
        warnings = []
        
        for idx in sample_indices:
            sample = self.X[idx].unsqueeze(0)  # Add batch dimension
            
            # Run QC validation
            validation_result = self.qc.validate_input_tensor(sample)
            
            if not validation_result['pass']:
                failures.extend(validation_result['failures'])
            if validation_result['warnings']:
                warnings.extend(validation_result['warnings'])
        
        # Report results
        print(f"\n  Checked {n_check} samples:")
        print(f"  ✓ Shape: {tuple(self.X.shape)}")
        print(f"  ✓ Data type: {self.X.dtype}")
        print(f"  ✓ Value range: [{self.X.min():.2f}, {self.X.max():.2f}]")
        print(f"  ✓ Mean: {self.X.mean():.4f}, Std: {self.X.std():.4f}")
        
        if failures:
            print(f"\n  ⚠ Quality failures: {len(set(failures))}")
            for failure in set(failures):
                print(f"    - {failure}")
        else:
            print("\n  ✅ All quality checks PASSED")
            
        if warnings:
            print(f"\n  ⚠ Warnings: {len(set(warnings))}")
            for warning in set(warnings)[:5]:  # Show first 5
                print(f"    - {warning}")
                
        # Check for data issues
        nan_count = torch.isnan(self.X).sum().item()
        inf_count = torch.isinf(self.X).sum().item()
        
        if nan_count > 0:
            print(f"\n  ⚠ Found {nan_count} NaN values - replacing with 0")
            self.X = torch.nan_to_num(self.X, nan=0.0)
            
        if inf_count > 0:
            print(f"\n  ⚠ Found {inf_count} Inf values - clipping")
            self.X = torch.clamp(self.X, -1e6, 1e6)
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_with_qa():
    """Train model with preprocessing QA."""
    
    print("\n" + "="*80)
    print("M1.5 TRAINING: FSQ+HSMM with Preprocessing QA")
    print("Target: Match M1.0-M1.2 78.12% accuracy")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set up quality control
    qc_thresholds = QualityThresholds(
        max_nan_percentage=5.0,
        min_signal_std=0.01,
        max_signal_std=50.0,
        min_codebook_usage=0.2,
        min_perplexity=4.0
    )
    qc = QualityControl(thresholds=qc_thresholds, strict_mode=False)
    
    # Check for quadruped data first
    quadruped_dir = Path("./quadruped_data/processed")
    eval_dir = Path("./evaluation_data")
    
    if quadruped_dir.exists() and (quadruped_dir / "X_train_quadruped.npy").exists():
        print("\n✓ Loading quadruped locomotion data...")
        X_train = np.load(quadruped_dir / "X_train_quadruped.npy")
        y_train = np.load(quadruped_dir / "y_train_quadruped.npy")
        X_val = np.load(quadruped_dir / "X_val_quadruped.npy")
        y_val = np.load(quadruped_dir / "y_val_quadruped.npy")
        X_test = np.load(quadruped_dir / "X_test_quadruped.npy")
        y_test = np.load(quadruped_dir / "y_test_quadruped.npy")
        n_classes = 10  # Quadruped behaviors
    elif eval_dir.exists():
        print("\n✓ Loading simple behavioral data...")
        X_train = np.load(eval_dir / "X_train.npy")
        y_train = np.load(eval_dir / "y_train.npy")
        X_val = np.load(eval_dir / "X_val.npy")
        y_val = np.load(eval_dir / "y_val.npy")
        X_test = np.load(eval_dir / "X_test.npy")
        y_test = np.load(eval_dir / "y_test.npy")
        n_classes = 5  # Simple behaviors
    else:
        print("\n⚠ No data found. Running setup...")
        from setup_quadruped_datasets import main as setup_quadruped
        setup_quadruped()
        
        # Try loading again
        if quadruped_dir.exists():
            X_train = np.load(quadruped_dir / "X_train_quadruped.npy")
            y_train = np.load(quadruped_dir / "y_train_quadruped.npy")
            X_val = np.load(quadruped_dir / "X_val_quadruped.npy")
            y_val = np.load(quadruped_dir / "y_val_quadruped.npy")
            X_test = np.load(quadruped_dir / "X_test_quadruped.npy")
            y_test = np.load(quadruped_dir / "y_test_quadruped.npy")
            n_classes = 10
        else:
            raise RuntimeError("Failed to set up data")
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Classes: {n_classes}")
    
    # Create datasets with QA
    train_dataset = QuadrupedDatasetWithQA(X_train, y_train, qc)
    val_dataset = QuadrupedDatasetWithQA(X_val, y_val, qc)
    test_dataset = QuadrupedDatasetWithQA(X_test, y_test, qc)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create FSQ+HSMM model (best from M1.2)
    print("\n" + "-"*60)
    print("Initializing FSQ+HSMM model (M1.2 best architecture)")
    print("-"*60)
    
    model = FSQHSMMModel(
        input_channels=9,
        num_classes=n_classes,
        levels=[8, 6, 5],  # FSQ levels from M1.2
        hidden_dim=64
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print("  Architecture: FSQ + HSMM (100% in M1.2 ablation)")
    
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
    max_patience = 20
    
    for epoch in range(200):  # More epochs for quadruped data
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'n_params': n_params,
                'architecture': 'FSQ+HSMM'
            }, 'm15_fsq_hsmm_best.pth')
            
            if val_acc > 0.75:  # Approaching M1.2 target
                print(f"  → Approaching target! Val accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
            
    print(f"\n✓ Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Target was: 0.7812 (78.12% from M1.0-M1.2)")
    
    if best_val_acc >= 0.75:
        print("  🎉 Successfully approaching M1.2 performance!")
    else:
        print("  Need more training or better data")
        
    return best_val_acc

if __name__ == "__main__":
    accuracy = train_with_qa()