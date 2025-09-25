#!/usr/bin/env python3
"""
Test VQ with extreme parameter adjustments to prevent collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.VectorQuantizerEMA2D_Fixed import VectorQuantizerEMA2D_Fixed

def test_vq_extreme_params():
    """Test VQ with extreme parameters to force diversity."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")
    
    # Test configurations
    configs = [
        {"commitment_cost": 0.001, "decay": 0.999, "lr": 1e-4, "vq_weight": 0.01},
        {"commitment_cost": 0.0001, "decay": 0.9999, "lr": 1e-5, "vq_weight": 0.001},
        {"commitment_cost": 0.25, "decay": 0.99, "lr": 1e-3, "vq_weight": 0.1, "l2_norm": True},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config {i+1}: β={config['commitment_cost']}, decay={config['decay']}, "
              f"lr={config['lr']}, vq_weight={config['vq_weight']}")
        print('='*60)
        
        # Create model
        class TestModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(100, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.LayerNorm(128)  # Add normalization
                )
                self.vq = VectorQuantizerEMA2D_Fixed(
                    num_codes=256,
                    code_dim=128,
                    decay=config['decay'],
                    commitment_cost=config['commitment_cost'],
                    l2_normalize_input=config.get('l2_norm', False),
                    restart_dead_codes=True,
                    dead_code_threshold=0.02  # More aggressive restart
                )
                self.decoder = nn.Linear(128, 10)
            
            def forward(self, x):
                z = self.encoder(x)
                z = z.unsqueeze(2)  # (B, 128, 1)
                z_q, vq_loss, info = self.vq(z)
                z_q = z_q.squeeze(2)
                out = self.decoder(z_q)
                return out, vq_loss, info
        
        model = TestModel(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
        # Initialize codebook from diverse data
        print("Initializing codebook from diverse data...")
        with torch.no_grad():
            for _ in range(10):
                x = torch.randn(256, 100).to(device)
                z = model.encoder(x).unsqueeze(2)
                # Force assignment to spread codes
                if not model.vq.initialized:
                    # Initialize embedding with encoder outputs
                    z_flat = z.squeeze(2).view(-1, 128)
                    indices = torch.randperm(z_flat.size(0))[:256]
                    model.vq.embedding.data = z_flat[indices]
                    model.vq.ema_cluster_size.data = torch.ones(256).to(device)
                    model.vq.ema_cluster_sum.data = model.vq.embedding.data.clone()
                    model.vq.initialized.data = torch.tensor(True)
        
        # Training
        best_perp = 0
        for epoch in range(50):
            # Diverse batch
            x = torch.randn(32, 100).to(device) * (1 + 0.1 * torch.randn(32, 1).to(device))
            y = torch.randint(0, 10, (32,)).to(device)
            
            out, vq_loss, info = model(x)
            cls_loss = F.cross_entropy(out, y)
            
            # Very low VQ weight
            total_loss = cls_loss + config['vq_weight'] * vq_loss["vq"]
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Aggressive clipping
            optimizer.step()
            
            best_perp = max(best_perp, info["perplexity"])
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss={total_loss.item():.4f}, "
                      f"Perp={info['perplexity']:.2f}, "
                      f"Unique={info['unique_codes']}/256")
        
        print(f"Best perplexity achieved: {best_perp:.2f}")
        
        if best_perp > 10:
            print("✅ Found working configuration!")
            return config
    
    print("\n❌ All configurations failed to maintain diversity")
    return None

def test_alternative_approach():
    """Test with frozen encoder initially."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Testing with Frozen Encoder Approach")
    print('='*60)
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(100, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128)
            )
            self.vq = VectorQuantizerEMA2D_Fixed(
                num_codes=64,  # Fewer codes
                code_dim=128,
                decay=0.95,  # Faster adaptation
                commitment_cost=0.1,
                l2_normalize_input=False,
                restart_dead_codes=True,
                dead_code_threshold=0.05
            )
            self.decoder = nn.Linear(128, 10)
            
        def forward(self, x):
            z = self.encoder(x)
            z = z.unsqueeze(2)
            z_q, vq_loss, info = self.vq(z)
            z_q = z_q.squeeze(2)
            out = self.decoder(z_q)
            return out, vq_loss, info
    
    model = TestModel().to(device)
    
    # Stage 1: Train with frozen encoder
    print("Stage 1: Frozen encoder, training VQ only...")
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3
    )
    
    for epoch in range(30):
        x = torch.randn(32, 100).to(device)
        y = torch.randint(0, 10, (32,)).to(device)
        
        out, vq_loss, info = model(x)
        cls_loss = F.cross_entropy(out, y)
        total_loss = cls_loss + 0.5 * vq_loss["vq"]
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Perp={info['perplexity']:.2f}, "
                  f"Unique={info['unique_codes']}/64")
    
    # Stage 2: Unfreeze encoder
    print("\nStage 2: Unfreezing encoder...")
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    final_perp = 0
    for epoch in range(30):
        x = torch.randn(32, 100).to(device)
        y = torch.randint(0, 10, (32,)).to(device)
        
        out, vq_loss, info = model(x)
        cls_loss = F.cross_entropy(out, y)
        total_loss = cls_loss + 0.1 * vq_loss["vq"]
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        final_perp = info["perplexity"]
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Perp={info['perplexity']:.2f}, "
                  f"Unique={info['unique_codes']}/64")
    
    if final_perp > 5:
        print(f"\n✅ Frozen encoder approach worked! Final perplexity: {final_perp:.2f}")
        return True
    else:
        print(f"\n❌ Still collapsed. Final perplexity: {final_perp:.2f}")
        return False

if __name__ == "__main__":
    # Try extreme parameters
    working_config = test_vq_extreme_params()
    
    if not working_config:
        # Try frozen encoder approach
        success = test_alternative_approach()
        
        if not success:
            print("\n" + "="*60)
            print("RECOMMENDATION: Consider FSQ or removing VQ")
            print("VQ-EMA appears fundamentally incompatible with this task")
            print("="*60)