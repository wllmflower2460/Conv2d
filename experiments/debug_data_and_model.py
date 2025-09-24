#!/usr/bin/env python3
"""
Debug data creation and simple model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_data_creation():
    """Test if data is created correctly."""
    print("="*50)
    print("Testing Data Creation")
    print("="*50)
    
    n = 100
    X = torch.randn(n, 9, 2, 100)
    y = torch.zeros(n, dtype=torch.long)
    
    for i in range(10):
        start = i * 10
        end = (i + 1) * 10
        y[start:end] = i
        X[start:end, 0, 0, :] += i * 5.0  # Strong signal
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y unique values: {torch.unique(y).tolist()}")
    print(f"y distribution: {[(y == i).sum().item() for i in range(10)]}")
    
    # Check if signal is present
    for i in range(10):
        class_data = X[y == i]
        mean_val = class_data[:, 0, 0, :].mean().item()
        print(f"Class {i}: mean channel 0 = {mean_val:.2f}")
    
    return X, y

def test_simple_linear():
    """Test with simplest possible model."""
    print("\n" + "="*50)
    print("Testing Simple Linear Model")
    print("="*50)
    
    # Even simpler data
    n = 100
    X = torch.randn(n, 100)
    y = torch.zeros(n, dtype=torch.long)
    
    for i in range(10):
        start = i * 10
        end = (i + 1) * 10
        y[start:end] = i
        X[start:end, i] += 10.0  # Very strong signal in feature i
    
    print(f"Created simple data: X={X.shape}, y={y.shape}")
    
    # Simple linear model
    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Train
    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                _, pred = logits.max(1)
                acc = (pred == y).float().mean() * 100
                print(f"Epoch {epoch}: Loss={loss:.3f}, Acc={acc:.1f}%")
    
    # Final accuracy
    with torch.no_grad():
        logits = model(X)
        _, pred = logits.max(1)
        acc = (pred == y).float().mean() * 100
        print(f"Final accuracy: {acc:.1f}%")
    
    return acc > 90

def test_conv_model():
    """Test with Conv2d model."""
    print("\n" + "="*50)
    print("Testing Conv2d Model")
    print("="*50)
    
    # Create data with clear patterns
    n = 200
    X = torch.zeros(n, 9, 2, 100)
    y = torch.zeros(n, dtype=torch.long)
    
    for i in range(10):
        start = i * 20
        end = (i + 1) * 20
        y[start:end] = i
        # Put different patterns in different channels
        if i < 9:
            X[start:end, i, :, :] = 1.0 + torch.randn(20, 2, 100) * 0.1
        else:
            X[start:end, :, :, :] = torch.randn(20, 9, 2, 100) * 0.5
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Check patterns
    for i in range(min(3, 10)):
        class_mean = X[y == i].mean(dim=(0, 2, 3))  # Mean across batch and spatial
        print(f"Class {i} channel means: {class_mean[:3].tolist()}")
    
    # Simple Conv model
    class SimpleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(9, 32, kernel_size=(2, 5), padding=(0, 2))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)
            
        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleConv().to(device)
    X, y = X.to(device), y.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            with torch.no_grad():
                _, pred = logits.max(1)
                acc = (pred == y).float().mean() * 100
                print(f"Epoch {epoch}: Loss={loss:.3f}, Acc={acc:.1f}%")
    
    # Check predictions
    with torch.no_grad():
        logits = model(X)
        _, pred = logits.max(1)
        acc = (pred == y).float().mean() * 100
        print(f"\nFinal accuracy: {acc:.1f}%")
        
        # Check per-class accuracy
        for i in range(10):
            class_mask = y == i
            if class_mask.sum() > 0:
                class_acc = (pred[class_mask] == y[class_mask]).float().mean() * 100
                print(f"Class {i}: {class_acc:.1f}%")
    
    return acc

def main():
    # Test data creation
    X, y = test_data_creation()
    
    # Test simple linear (should work)
    linear_works = test_simple_linear()
    print(f"\n✅ Linear model works: {linear_works}")
    
    # Test conv model
    conv_acc = test_conv_model()
    print(f"\n📊 Conv model accuracy: {conv_acc:.1f}%")
    
    if conv_acc > 50:
        print("\n✅ Conv model works! Issue might be in FSQ integration.")
    else:
        print("\n⚠️ Conv model has issues too. Problem is in the architecture or data.")

if __name__ == "__main__":
    main()