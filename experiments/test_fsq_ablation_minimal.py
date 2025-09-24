#!/usr/bin/env python3
"""
Minimal test of FSQ ablation - just 2 configs, 10 epochs.
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

class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 32, kernel_size=(1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)

class TestModel(nn.Module):
    def __init__(self, use_fsq=False):
        super().__init__()
        self.encoder = SimpleEncoder()
        self.use_fsq = use_fsq
        
        if use_fsq:
            # 8D FSQ
            self.fsq_proj = nn.Linear(64, 8)
            nn.init.xavier_normal_(self.fsq_proj.weight, gain=2.0)
            self.fsq = FSQ(levels=[8]*8)
            feat_dim = 8
        else:
            feat_dim = 64
            
        self.classifier = nn.Linear(feat_dim, 10)
        
    def forward(self, x):
        features = self.encoder(x)
        
        if self.use_fsq:
            features = self.fsq_proj(features) * 2.0  # Scale up
            features = features.unsqueeze(1)
            features, _ = self.fsq(features)
            features = features.squeeze(1)
            
        return self.classifier(features)

def create_data(n=600):
    """Simple data."""
    X = torch.randn(n, 9, 2, 100)
    y = torch.zeros(n, dtype=torch.long)
    
    for i in range(10):
        start = i * (n // 10)
        end = (i + 1) * (n // 10) if i < 9 else n
        y[start:end] = i
        # Add strong signal
        X[start:end, 0, 0, :] += i * 2.0
    
    return X, y

def test_config(name, use_fsq, train_loader, val_loader, device):
    print(f"\nTesting {name}...")
    model = TestModel(use_fsq=use_fsq).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(10):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        
        # Validate
        if epoch == 9:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    _, pred = logits.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            
            acc = 100. * correct / total
            print(f"  Final accuracy: {acc:.1f}%")
            return acc
    
    return 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data
    X, y = create_data(600)
    train_loader = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[400:], y[400:]), batch_size=32)
    
    print("="*50)
    print("MINIMAL FSQ TEST")
    print("="*50)
    
    # Test baseline
    baseline_acc = test_config("Baseline", use_fsq=False, 
                               train_loader=train_loader, 
                               val_loader=val_loader, 
                               device=device)
    
    # Test FSQ
    fsq_acc = test_config("FSQ", use_fsq=True,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         device=device)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Baseline: {baseline_acc:.1f}%")
    print(f"FSQ:      {fsq_acc:.1f}%")
    
    if baseline_acc > 80 and fsq_acc > 80:
        print("\n✅ Both models work! The fixed ablation should work too.")
    else:
        print(f"\n⚠️ Low accuracy detected. Baseline: {baseline_acc:.1f}%, FSQ: {fsq_acc:.1f}%")

if __name__ == "__main__":
    main()