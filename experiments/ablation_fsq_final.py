#!/usr/bin/env python3
"""
Final FSQ ablation study with proper data shuffling and splits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from dataclasses import dataclass
from typing import Optional, Dict, List
import json
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_quantize_pytorch import FSQ

@dataclass
class Config:
    name: str
    use_fsq: bool = False
    use_hdp: bool = False
    use_hsmm: bool = False
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    def __str__(self):
        components = []
        if self.use_fsq: components.append("FSQ")
        if self.use_hdp: components.append("HDP")
        if self.use_hsmm: components.append("HSMM")
        return "+".join(components) if components else "Baseline"

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 32, kernel_size=(1, 7), padding=(0, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (1, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (1, 2))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)

class AblationModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder()
        
        feature_dim = 128
        
        # FSQ
        if config.use_fsq:
            self.fsq_proj = nn.Linear(128, 8)
            nn.init.xavier_normal_(self.fsq_proj.weight, gain=2.0)
            self.fsq = FSQ(levels=[8]*8)
            feature_dim = 8
            self.register_buffer('code_usage', torch.zeros(1000))
        else:
            self.fsq = None
            
        # HDP
        if config.use_hdp:
            self.hdp = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(feature_dim, 20),
                nn.Softmax(dim=-1)
            )
            feature_dim = 20
        else:
            self.hdp = None
            
        # HSMM
        if config.use_hsmm:
            self.hsmm = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(feature_dim, 100)
            )
            feature_dim = 100
        else:
            self.hsmm = None
            
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        
        if self.fsq:
            features = self.fsq_proj(features) * 2.0
            features = features.unsqueeze(1)
            features, codes = self.fsq(features)
            features = features.squeeze(1)
            
            # Track code usage
            if self.training:
                unique_codes = torch.unique(codes)
                for c in unique_codes:
                    if c < 1000:
                        self.code_usage[c] += 1
        
        if self.hdp:
            features = self.hdp(features)
            
        if self.hsmm:
            features = self.hsmm(features)
            
        return self.classifier(features)
    
    def get_stats(self):
        if not self.fsq:
            return {"codes": 0, "perplexity": 0}
        
        used = (self.code_usage > 0).sum().item()
        if self.code_usage.sum() > 0:
            probs = self.code_usage / self.code_usage.sum()
            probs = probs[probs > 0]
            entropy = -(probs * probs.log()).sum()
            perplexity = entropy.exp().item()
        else:
            perplexity = 0
            
        return {"codes": used, "perplexity": perplexity}

def create_dataset(n_samples=3000):
    """Create dataset with proper patterns."""
    X = torch.zeros(n_samples, 9, 2, 100)
    y = torch.zeros(n_samples, dtype=torch.long)
    
    samples_per_class = n_samples // 10
    
    for i in range(10):
        start = i * samples_per_class
        end = (i + 1) * samples_per_class if i < 9 else n_samples
        n = end - start
        
        # Base noise
        X[start:end] = torch.randn(n, 9, 2, 100) * 0.3
        
        # Add class-specific patterns
        if i == 0:  # Stationary
            X[start:end] *= 0.1
        elif i == 1:  # Walking
            t = torch.linspace(0, 4*np.pi, 100)
            X[start:end, 0, 0, :] += torch.sin(t) * 2
            X[start:end, 1, 1, :] += torch.cos(t) * 2
        elif i == 2:  # Running
            t = torch.linspace(0, 8*np.pi, 100)
            X[start:end, 0, :, :] += torch.sin(t).view(1, -1) * 3
            X[start:end, 1, :, :] += torch.cos(t).view(1, -1) * 3
        elif i == 3:  # Jumping
            for j in range(5):
                pos = j * 20
                X[start:end, 2, :, pos:pos+10] += 5.0
        else:
            # Unique pattern per class
            X[start:end, i % 9, :, :] += 2.0
            
        y[start:end] = i
    
    # IMPORTANT: Shuffle the data!
    perm = torch.randperm(n_samples)
    X = X[perm]
    y = y[perm]
    
    # Normalize
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    return X, y

def train_and_evaluate(config, train_loader, val_loader, test_loader, device):
    """Train and evaluate a model."""
    model = AblationModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{str(config)}")
    print(f"Parameters: {num_params:,}")
    
    best_val_acc = 0
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = logits.max(1)
            train_correct += pred.eq(y).sum().item()
            train_total += y.size(0)
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                _, pred = logits.max(1)
                val_correct += pred.eq(y).sum().item()
                val_total += y.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)
        
        if epoch % 5 == 0 or epoch == config.epochs - 1:
            stats = model.get_stats()
            print(f"  Epoch {epoch}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, "
                  f"Codes={stats['codes']}, Perp={stats['perplexity']:.1f}")
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, pred = logits.max(1)
            test_correct += pred.eq(y).sum().item()
            test_total += y.size(0)
    
    test_acc = 100. * test_correct / test_total
    stats = model.get_stats()
    
    return {
        "params": num_params,
        "test_acc": test_acc,
        "best_val": best_val_acc,
        "codes": stats["codes"],
        "perplexity": stats["perplexity"]
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create dataset
    print("Creating dataset...")
    X, y = create_dataset(3000)
    dataset = TensorDataset(X, y)
    
    # PROPER random split
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Configurations
    configs = [
        Config("baseline"),
        Config("fsq", use_fsq=True),
        Config("hdp", use_hdp=True),
        Config("hsmm", use_hsmm=True),
        Config("fsq_hdp", use_fsq=True, use_hdp=True),
        Config("fsq_hsmm", use_fsq=True, use_hsmm=True),
        Config("hdp_hsmm", use_hdp=True, use_hsmm=True),
        Config("all", use_fsq=True, use_hdp=True, use_hsmm=True),
    ]
    
    print("\n" + "="*60)
    print("FSQ ABLATION STUDY")
    print("="*60)
    
    results = {}
    
    for config in configs:
        result = train_and_evaluate(config, train_loader, val_loader, test_loader, device)
        results[str(config)] = result
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Config':<15} {'Params':>8} {'Test':>7} {'Val':>7} {'Codes':>7} {'Perp':>7}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<15} {res['params']:>8,} {res['test_acc']:>6.1f}% "
              f"{res['best_val']:>6.1f}% {res['codes']:>7} {res['perplexity']:>7.1f}")
    
    # Analysis
    baseline = results.get("Baseline", {}).get("test_acc", 0)
    fsq = results.get("FSQ", {}).get("test_acc", 0)
    all_model = results.get("FSQ+HDP+HSMM", {}).get("test_acc", 0)
    
    print(f"\n📊 Analysis:")
    print(f"  Baseline: {baseline:.1f}%")
    print(f"  FSQ only: {fsq:.1f}%")
    print(f"  Full model: {all_model:.1f}%")
    
    best = max(results.items(), key=lambda x: x[1]["test_acc"])
    print(f"\n🏆 Best: {best[0]} with {best[1]['test_acc']:.1f}% accuracy")
    
    # Check FSQ stability
    fsq_configs = [k for k in results if "FSQ" in k]
    if fsq_configs:
        avg_codes = np.mean([results[k]["codes"] for k in fsq_configs])
        avg_perp = np.mean([results[k]["perplexity"] for k in fsq_configs])
        print(f"\n✅ FSQ stability: Avg {avg_codes:.0f} codes, {avg_perp:.1f} perplexity")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"fsq_ablation_final_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to fsq_ablation_final_{timestamp}.json")

if __name__ == "__main__":
    main()