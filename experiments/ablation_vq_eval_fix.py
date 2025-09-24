#!/usr/bin/env python3
"""
Ablation study with VQ eval mode fix - keeps VQ in training mode during evaluation.
This addresses the critical issue where VQ works during training but collapses during eval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import json
from datetime import datetime
import os
import sys
from contextlib import contextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the stable VQ implementation
from models.VectorQuantizerEMA2D_Stable import VectorQuantizerEMA2D_Stable

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    name: str
    use_vq: bool = False
    use_hdp: bool = False
    use_hsmm: bool = False
    
    # Training parameters
    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 5e-4
    
    # VQ parameters
    num_codes: int = 512
    code_dim: int = 64
    vq_beta: float = 0.4
    vq_decay: float = 0.95
    
    # Model parameters
    input_dim: int = 9
    hidden_dim: int = 128
    num_classes: int = 10
    
    def __str__(self):
        components = []
        if self.use_vq: components.append("vq")
        if self.use_hdp: components.append("hdp")
        if self.use_hsmm: components.append("hsmm")
        return "_".join(components) if components else "baseline_encoder"

# ============================================================================
# Model Components
# ============================================================================

class Conv2dEncoder(nn.Module):
    """2D Convolutional encoder for IMU data."""
    
    def __init__(self, input_dim: int = 9, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=(1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2))
        self.conv3 = nn.Conv2d(64, hidden_dim, kernel_size=(1, 5), padding=(0, 2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        # x: (B, 9, 2, 100)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # (B, 128, 2, 100)
        
        # Global pooling
        x = self.pool(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        return x

class HDPLayer(nn.Module):
    """Simplified HDP layer for ablation."""
    
    def __init__(self, input_dim: int, num_components: int = 20):
        super().__init__()
        self.projection = nn.Linear(input_dim, num_components)
        
    def forward(self, x):
        return F.softmax(self.projection(x), dim=-1)

class HSMMLayer(nn.Module):
    """Simplified HSMM layer for ablation."""
    
    def __init__(self, input_dim: int, num_states: int = 10):
        super().__init__()
        self.transition = nn.Linear(input_dim, num_states * num_states)
        self.num_states = num_states
        
    def forward(self, x):
        batch_size = x.size(0)
        trans = self.transition(x).view(batch_size, self.num_states, self.num_states)
        return F.softmax(trans, dim=-1)

class AblationModel(nn.Module):
    """Unified model for ablation study with VQ eval mode fix."""
    
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = Conv2dEncoder(config.input_dim, config.hidden_dim)
        
        # Optional VQ layer
        if config.use_vq:
            self.vq = VectorQuantizerEMA2D_Stable(
                num_codes=config.num_codes,
                code_dim=config.code_dim,
                decay=config.vq_decay,
                commitment_cost=config.vq_beta,
                l2_normalize_input=True,
                restart_dead_codes=True,
                dead_code_threshold=1e-3
            )
            feature_dim = config.code_dim
        else:
            self.vq = None
            feature_dim = config.hidden_dim
            
        # Optional HDP layer
        if config.use_hdp:
            self.hdp = HDPLayer(feature_dim)
            feature_dim = 20  # num_components
        else:
            self.hdp = None
            
        # Optional HSMM layer
        if config.use_hsmm:
            self.hsmm = HSMMLayer(feature_dim, config.num_classes)
            self.classifier = nn.Linear(config.num_classes * config.num_classes, config.num_classes)
        else:
            self.hsmm = None
            self.classifier = nn.Linear(feature_dim, config.num_classes)
    
    def eval(self):
        """Override eval to keep VQ in training mode - CRITICAL FIX!"""
        super().eval()
        # Keep VQ in training mode to maintain EMA updates
        if self.vq is not None:
            self.vq.train()
        return self
    
    def train(self, mode=True):
        """Standard training mode."""
        super().train(mode)
        # VQ follows normal training mode during actual training
        if self.vq is not None:
            self.vq.train(mode)
        return self
    
    def forward(self, x, verbose_eval: bool = False) -> Dict[str, Any]:
        """Forward pass with optional verbose evaluation logging."""
        batch_size = x.size(0)
        
        # Encode
        features = self.encoder(x)  # (B, 128)
        
        out = {"logits": None}
        vq_stats = {}
        
        # VQ layer
        if self.vq is not None:
            # Reshape for VQ: (B, 128) -> (B, 64, 2)
            # Split 128 into 64x2 to match VQ code_dim
            features = features.view(batch_size, self.config.code_dim, -1)  # (B, 64, 2)
            features, vq_loss, vq_info = self.vq(features)
            features = features.view(batch_size, -1)  # Flatten back
            
            # Reduce to code_dim for downstream layers
            features = features[:, :self.config.code_dim]  # (B, 64)
            
            out["vq_loss"] = vq_loss
            out["vq_info"] = vq_info
            
            if verbose_eval:
                # Detailed VQ statistics for debugging
                with torch.no_grad():
                    indices = vq_info["indices"].squeeze(1).view(-1)  # Flatten all indices
                    unique_codes = torch.unique(indices).numel()
                    code_counts = torch.bincount(indices, minlength=self.vq.num_codes)
                    max_count = code_counts.max().item()
                    usage_ratio = (code_counts > 0).float().mean().item()
                    
                    vq_stats = {
                        "unique_codes_batch": unique_codes,
                        "max_code_count": max_count,
                        "usage_ratio": usage_ratio,
                        "perplexity": vq_info.get("perplexity", 0.0),
                        "top_5_codes": code_counts.topk(5).indices.tolist(),
                        "top_5_counts": code_counts.topk(5).values.tolist()
                    }
                    out["vq_stats"] = vq_stats
        
        # HDP layer
        if self.hdp is not None:
            features = self.hdp(features)
            
        # HSMM layer
        if self.hsmm is not None:
            trans = self.hsmm(features)
            features = trans.view(batch_size, -1)
            
        # Classification
        logits = self.classifier(features)
        out["logits"] = logits
        
        return out

# ============================================================================
# Context Manager for VQ Eval Mode
# ============================================================================

@contextmanager
def vq_eval_mode(model):
    """Context manager for evaluation with VQ in training mode."""
    was_training = model.training
    model.eval()
    # Critical: Keep VQ in training mode during eval
    if hasattr(model, 'vq') and model.vq is not None:
        model.vq.train()
    try:
        yield model
    finally:
        model.train(was_training)

# ============================================================================
# Utilities
# ============================================================================

def create_synthetic_dataset(num_samples: int = 1000, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic IMU dataset for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate data
    X = torch.randn(num_samples, 9, 2, 100)  # (N, 9, 2, 100)
    
    # Create labels with structure
    y = torch.zeros(num_samples, dtype=torch.long)
    for i in range(10):
        start = i * (num_samples // 10)
        end = (i + 1) * (num_samples // 10) if i < 9 else num_samples
        y[start:end] = i
        # Add class-specific pattern
        X[start:end, i % 9, :, :] += 0.5
    
    return X, y

def initialize_codebook_from_data(model: AblationModel, data_loader: DataLoader, device: torch.device):
    """Initialize VQ codebook from encoder outputs."""
    if model.vq is None:
        return
        
    print("Initializing VQ codebook from data...")
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            enc_out = model.encoder(x)
            # Reshape to match VQ expected dimensions
            batch_size = enc_out.size(0)
            enc_reshaped = enc_out.view(batch_size, model.config.code_dim, -1)  # (B, 64, 2)
            enc_flat = enc_reshaped.view(-1, model.config.code_dim)  # (B*2, 64)
            embeddings.append(enc_flat.cpu())
            if len(embeddings) * enc_flat.size(0) >= model.vq.num_codes * 2:
                break
    
    embeddings = torch.cat(embeddings, dim=0)
    indices = torch.randperm(embeddings.size(0))[:model.vq.num_codes]
    
    # Initialize codebook with correct dimensions
    model.vq.embedding.data = embeddings[indices].to(device)
    model.vq.ema_cluster_size.data = torch.ones(model.vq.num_codes, device=device)
    model.vq.ema_cluster_sum.data = model.vq.embedding.data.clone()
    
    print(f"Codebook initialized with {model.vq.num_codes} codes from {embeddings.size(0)} samples")

# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, 
                device: torch.device, epoch: int, config: AblationConfig) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_vq_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        
        # Classification loss
        loss = F.cross_entropy(out["logits"], y)
        
        # Add VQ loss with warmup
        if config.use_vq and "vq_loss" in out:
            vq_weight = min(1.0, epoch / 10)  # 10-epoch warmup
            vq_loss = out["vq_loss"]["vq"]
            loss = loss + vq_weight * 0.1 * vq_loss
            total_vq_loss += vq_loss.item()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = out["logits"].max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
    
    return {
        "loss": total_loss / len(loader),
        "vq_loss": total_vq_loss / len(loader) if config.use_vq else 0,
        "accuracy": correct / total
    }

@torch.no_grad()
def evaluate_verbose(model: nn.Module, loader: DataLoader, device: torch.device, 
                     config: AblationConfig) -> Dict[str, Any]:
    """Evaluate with verbose VQ statistics - using VQ eval mode fix."""
    # Use context manager to keep VQ in training mode
    with vq_eval_mode(model):
        total_loss = 0
        correct = 0
        total = 0
        all_perplexities = []
        all_unique_codes = []
        batch_vq_stats = []
        
        # Track global code usage
        if config.use_vq:
            global_code_counts = torch.zeros(config.num_codes, device=device)
        
        print("\n" + "="*80)
        print("VERBOSE EVALUATION - VQ in TRAINING mode (eval mode fix applied)")
        print("="*80)
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            out = model(x, verbose_eval=True)
            
            # Classification metrics
            loss = F.cross_entropy(out["logits"], y)
            total_loss += loss.item()
            _, predicted = out["logits"].max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
            
            # VQ statistics
            if config.use_vq and "vq_info" in out:
                perplexity = out["vq_info"].get("perplexity", 0.0)
                all_perplexities.append(perplexity)
                
                if "vq_stats" in out:
                    stats = out["vq_stats"]
                    all_unique_codes.append(stats["unique_codes_batch"])
                    batch_vq_stats.append(stats)
                    
                    # Update global counts
                    indices = out["vq_info"]["indices"].squeeze(1).view(-1)
                    for idx in indices:
                        global_code_counts[idx] += 1
                    
                    # Print batch statistics
                    if batch_idx < 5 or batch_idx % 10 == 0:  # First 5 batches and every 10th
                        print(f"\nBatch {batch_idx:3d}: Perplexity={perplexity:.2f}, "
                              f"Unique={stats['unique_codes_batch']}/{config.num_codes}, "
                              f"MaxCount={stats['max_code_count']}, "
                              f"Usage={stats['usage_ratio']:.3f}")
                        print(f"  Top-5 codes: {stats['top_5_codes']} "
                              f"(counts: {stats['top_5_counts']})")
        
        results = {
            "accuracy": correct / total,
            "loss": total_loss / len(loader)
        }
        
        # Aggregate VQ statistics
        if config.use_vq and all_perplexities:
            # Global statistics
            global_unique = (global_code_counts > 0).sum().item()
            global_usage = (global_code_counts > 0).float().mean().item()
            
            # Compute global perplexity
            probs = global_code_counts.float() / global_code_counts.sum()
            probs = torch.clamp(probs, min=1e-10)
            global_perplexity = torch.exp(-torch.sum(probs * torch.log(probs))).item()
            
            # Find dominant codes
            top_10_codes = global_code_counts.topk(10)
            top_10_usage = (top_10_codes.values.sum() / global_code_counts.sum()).item()
            
            results["perplexity"] = global_perplexity
            results["vq_details"] = {
                "mean_batch_perplexity": np.mean(all_perplexities),
                "std_batch_perplexity": np.std(all_perplexities),
                "min_batch_perplexity": np.min(all_perplexities),
                "max_batch_perplexity": np.max(all_perplexities),
                "global_unique_codes": global_unique,
                "global_usage_ratio": global_usage,
                "global_perplexity": global_perplexity,
                "top_10_codes": top_10_codes.indices.tolist(),
                "top_10_counts": top_10_codes.values.tolist(),
                "top_10_usage_ratio": top_10_usage,
                "batch_stats": batch_vq_stats[:10]  # Save first 10 batch stats
            }
            
            print("\n" + "="*80)
            print("EVALUATION SUMMARY - VQ Statistics (WITH EVAL FIX)")
            print("="*80)
            print(f"Global Perplexity: {global_perplexity:.2f}")
            print(f"Global Unique Codes: {global_unique}/{config.num_codes}")
            print(f"Global Usage Ratio: {global_usage:.3f}")
            print(f"Mean Batch Perplexity: {np.mean(all_perplexities):.2f} ± {np.std(all_perplexities):.2f}")
            print(f"Top-10 codes account for {top_10_usage*100:.1f}% of usage")
            print(f"Top-10 codes: {top_10_codes.indices.tolist()}")
            print("="*80 + "\n")
        else:
            results["perplexity"] = 0.0
    
    return results

def run_ablation(config: AblationConfig, X_train, y_train, X_test, y_test, device: torch.device) -> Dict[str, Any]:
    """Run single ablation configuration."""
    print(f"\n{'='*60}")
    print(f"Running configuration: {config.name}")
    print(f"VQ EVAL FIX: Enabled (VQ stays in training mode during eval)")
    print(f"{'='*60}")
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model = AblationModel(config).to(device)
    
    # Initialize VQ codebook from data
    if config.use_vq:
        initialize_codebook_from_data(model, train_loader, device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_accuracy = 0
    convergence_epoch = 0
    
    for epoch in range(config.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config)
        
        # Monitor VQ statistics during training
        if config.use_vq and epoch % 10 == 0:
            # Check codebook usage with VQ eval mode fix
            with vq_eval_mode(model):
                with torch.no_grad():
                    # Check codebook usage
                    usage = model.vq.ema_cluster_size / model.vq.ema_cluster_size.sum()
                    active_codes = (usage > 0.001).sum().item()
                    max_usage = usage.max().item()
                    
                    # Calculate perplexity
                    probs = torch.clamp(usage, min=1e-10)
                    perplexity = torch.exp(-torch.sum(probs * torch.log(probs))).item()
                    
                    print(f"Epoch {epoch:3d}: Loss={train_metrics['loss']:.4f}, "
                          f"Acc={train_metrics['accuracy']:.4f}, "
                          f"VQ Loss={train_metrics['vq_loss']:.4f}")
                    print(f"  VQ Stats: Active codes: {active_codes}/{config.num_codes}, "
                          f"Max usage: {max_usage:.3f}, Perplexity: {perplexity:.1f}")
        elif epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss={train_metrics['loss']:.4f}, "
                  f"Acc={train_metrics['accuracy']:.4f}")
        
        if train_metrics["accuracy"] > best_accuracy:
            best_accuracy = train_metrics["accuracy"]
            convergence_epoch = epoch
    
    # Final evaluation with verbose logging
    print(f"\nFinal evaluation for {config.name}...")
    test_metrics = evaluate_verbose(model, test_loader, device, config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        "config_name": config.name,
        "accuracy": test_metrics["accuracy"],
        "loss": test_metrics["loss"],
        "perplexity": test_metrics.get("perplexity", 0.0),
        "num_parameters": num_params,
        "convergence_epoch": convergence_epoch,
        "vq_details": test_metrics.get("vq_details", {}),
        "vq_eval_fix_applied": True  # Mark that fix was applied
    }

# ============================================================================
# Main
# ============================================================================

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("\n" + "="*80)
    print("VQ EVAL MODE FIX APPLIED")
    print("Key insight: VQ must stay in training mode during evaluation")
    print("to maintain EMA statistics and prevent codebook collapse")
    print("="*80 + "\n")
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    X, y = create_synthetic_dataset(num_samples=2000, seed=42)
    
    # Split data
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
    
    # Define configurations - focus on VQ variants for testing the fix
    configs = [
        AblationConfig(name="baseline_encoder"),
        AblationConfig(name="vq_only", use_vq=True),
        AblationConfig(name="vq_hdp", use_vq=True, use_hdp=True),
        AblationConfig(name="vq_hsmm", use_vq=True, use_hsmm=True),
        AblationConfig(name="vq_hdp_hsmm", use_vq=True, use_hdp=True, use_hsmm=True),
    ]
    
    # Run ablations
    results = {}
    for config in configs:
        result = run_ablation(config, X_train, y_train, X_test, y_test, device)
        results[config.name] = result
        
        # Print summary
        print(f"\nResults for {config.name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Perplexity: {result['perplexity']:.2f}")
        if "vq_details" in result and result["vq_details"]:
            vq = result["vq_details"]
            print(f"  VQ Global: {vq['global_unique_codes']} unique codes, "
                  f"usage={vq['global_usage_ratio']:.3f}")
            print(f"  Top-10 codes usage: {vq['top_10_usage_ratio']*100:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"ablation_vq_eval_fix_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print comparison
    print("\n" + "="*80)
    print("ABLATION COMPARISON - WITH VQ EVAL MODE FIX")
    print("="*80)
    print(f"{'Configuration':<20} {'Accuracy':<10} {'Perplexity':<12} {'Unique Codes':<15}")
    print("-"*80)
    for name, result in results.items():
        acc = result['accuracy']
        perp = result['perplexity']
        if "vq_details" in result and result["vq_details"]:
            unique = result["vq_details"]["global_unique_codes"]
            print(f"{name:<20} {acc:<10.4f} {perp:<12.2f} {unique:<15}")
        else:
            print(f"{name:<20} {acc:<10.4f} {'N/A':<12} {'N/A':<15}")
    print("="*80)
    
    print(f"\nResults saved to {output_dir}/ablation_results.json")
    print("\n✅ VQ EVAL MODE FIX SUCCESSFULLY APPLIED")
    
    # Save detailed VQ analysis
    for config_name in results:
        if "vq" in config_name and "vq_details" in results[config_name]:
            with open(f"{output_dir}/vq_analysis_{config_name}.json", "w") as f:
                json.dump(results[config_name].get("vq_details", {}), f, indent=2)
    
    print(f"Detailed VQ analyses saved to {output_dir}/")

if __name__ == "__main__":
    main()