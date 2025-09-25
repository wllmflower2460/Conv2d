#!/usr/bin/env python3
"""
Debug script for FSQ ablation - finding the 0% accuracy issue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_quantize_pytorch import FSQ

# Simple test without any components
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=(1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(64, hidden_dim, kernel_size=(1, 5), padding=(0, 2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        logits = self.classifier(x)
        return logits

def create_simple_data(n=300):
    """Create very simple data to test."""
    X = torch.randn(n, 9, 2, 100) * 0.1  # Small noise
    y = torch.zeros(n, dtype=torch.long)
    
    # Make it very simple - just use first channel
    for i in range(10):
        start = i * (n // 10)
        end = (i + 1) * (n // 10) if i < 9 else n
        y[start:end] = i
        # Strong signal in first channel
        X[start:end, 0, :, :] += (i + 1) * 2.0  
    
    return X, y

def test_simple_model():
    """Test without FSQ first."""
    print("="*60)
    print("Testing Simple Classifier (No FSQ)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    X, y = create_simple_data(300)
    train_loader = DataLoader(TensorDataset(X[:200], y[:200]), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[200:], y[200:]), batch_size=32)
    
    # Model
    model = SimpleClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    for epoch in range(20):
        model.train()
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(device), y_batch.to(device)
                logits = model(x)
                _, predicted = logits.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
        
        acc = 100. * correct / total
        print(f"Epoch {epoch}: Val Acc = {acc:.1f}%")
    
    return acc > 50  # Should get much better than 50%

def test_fsq_dimensions():
    """Test FSQ dimension handling."""
    print("\n" + "="*60)
    print("Testing FSQ Dimensions")
    print("="*60)
    
    # Test FSQ directly
    fsq = FSQ(levels=[8, 5, 5, 4])
    fsq_dim = len([8, 5, 5, 4])  # 4
    
    # Create test input
    batch_size = 2
    features = torch.randn(batch_size, 128)  # From encoder
    
    print(f"Encoder output shape: {features.shape}")
    
    # Need to project from 128 to fsq_dim=4
    projection = nn.Linear(128, fsq_dim)
    features_proj = projection(features)
    print(f"After projection: {features_proj.shape}")
    
    # FSQ expects (batch, seq, dim)
    features_3d = features_proj.unsqueeze(1)
    print(f"After unsqueeze: {features_3d.shape}")
    
    # Apply FSQ
    quantized, codes = fsq(features_3d)
    print(f"Quantized shape: {quantized.shape}")
    print(f"Codes shape: {codes.shape}")
    
    # Remove sequence dim
    quantized = quantized.squeeze(1)
    print(f"Final quantized: {quantized.shape}")
    
    # Check values
    print(f"Quantized values range: [{quantized.min():.2f}, {quantized.max():.2f}]")
    print(f"Unique codes: {torch.unique(codes).tolist()}")
    
    return True

class DebugFSQModel(nn.Module):
    """FSQ model with extensive debugging."""
    def __init__(self, use_fsq=True):
        super().__init__()
        self.use_fsq = use_fsq
        
        # Encoder
        self.conv1 = nn.Conv2d(9, 64, kernel_size=(1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 5), padding=(0, 2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if use_fsq:
            # FSQ with projection
            self.fsq_levels = [8, 5, 5, 4]
            self.fsq_dim = len(self.fsq_levels)
            self.projection = nn.Linear(128, self.fsq_dim)
            self.fsq = FSQ(levels=self.fsq_levels)
            classifier_input = self.fsq_dim
        else:
            classifier_input = 128
            
        self.classifier = nn.Linear(classifier_input, 10)
        
    def forward(self, x):
        # Encode
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        features = self.pool(x).squeeze(-1).squeeze(-1)
        
        if self.use_fsq:
            # Project and quantize
            features = self.projection(features)
            features_3d = features.unsqueeze(1)
            quantized, codes = self.fsq(features_3d)
            features = quantized.squeeze(1)
            
            # Debug prints
            if hasattr(self, 'debug') and self.debug:
                print(f"Features shape: {features.shape}")
                print(f"Features range: [{features.min():.2f}, {features.max():.2f}]")
                print(f"Unique codes: {torch.unique(codes).numel()}")
        
        # Classify
        logits = self.classifier(features)
        return logits

def test_fsq_model():
    """Test FSQ model with debugging."""
    print("\n" + "="*60)
    print("Testing FSQ Model")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    X, y = create_simple_data(300)
    train_loader = DataLoader(TensorDataset(X[:200], y[:200]), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[200:], y[200:]), batch_size=32)
    
    # Model with FSQ
    model = DebugFSQModel(use_fsq=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Debug first forward pass
    model.debug = True
    with torch.no_grad():
        x_test = X[:2].to(device)
        logits = model(x_test)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits: {logits}")
    model.debug = False
    
    # Train
    for epoch in range(20):
        model.train()
        train_loss = 0
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(device), y_batch.to(device)
                logits = model(x)
                _, predicted = logits.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
        
        acc = 100. * correct / total
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch}: Loss={avg_loss:.3f}, Val Acc = {acc:.1f}%")
    
    return acc

def main():
    print("FSQ Ablation Debugging")
    print("="*60)
    
    # Test 1: Simple model without FSQ
    simple_works = test_simple_model()
    print(f"\n✅ Simple model works: {simple_works}")
    
    # Test 2: FSQ dimensions
    fsq_dims_ok = test_fsq_dimensions()
    print(f"\n✅ FSQ dimensions OK: {fsq_dims_ok}")
    
    # Test 3: FSQ model
    fsq_acc = test_fsq_model()
    print(f"\n✅ FSQ model accuracy: {fsq_acc:.1f}%")
    
    if fsq_acc > 50:
        print("\n🎉 FSQ model is working! The issue is in the ablation script.")
    else:
        print("\n❌ FSQ model still has issues.")

if __name__ == "__main__":
    main()