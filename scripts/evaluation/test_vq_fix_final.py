#!/usr/bin/env python3
"""
Test the fixed VQ implementation with proper gradient detachment and loss clamping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.VectorQuantizerEMA2D_Fixed import VectorQuantizerEMA2D_Fixed

def test_vq_stability():
    """Test that VQ doesn't explode with the fixes."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")
    
    # Create a simple model with VQ
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(100, 128)
            self.vq = VectorQuantizerEMA2D_Fixed(
                num_codes=256,
                code_dim=128,
                decay=0.99,
                commitment_cost=0.25,  # Back to reasonable value
                l2_normalize_input=False,
                restart_dead_codes=True
            )
            self.decoder = nn.Linear(128, 10)
        
        def forward(self, x):
            # Encode
            z = self.encoder(x)
            # Add temporal dimension for VQ
            z = z.unsqueeze(2)  # (B, 128, 1)
            # Quantize
            z_q, vq_loss, info = self.vq(z)
            # Remove temporal dimension
            z_q = z_q.squeeze(2)
            # Decode
            out = self.decoder(z_q)
            return out, vq_loss, info
    
    model = TestModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n" + "="*60)
    print("Testing VQ Stability with Fixes")
    print("="*60)
    
    # Training loop
    losses = []
    perplexities = []
    
    for epoch in range(100):
        # Generate batch
        x = torch.randn(32, 100).to(device)
        y = torch.randint(0, 10, (32,)).to(device)
        
        # Forward pass
        out, vq_loss, info = model(x)
        
        # Classification loss
        cls_loss = F.cross_entropy(out, y)
        
        # Total loss with VQ
        total_loss = cls_loss + 0.25 * vq_loss["vq"]
        
        # Check for explosion
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"❌ Loss exploded at epoch {epoch}: {total_loss.item()}")
            return False
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        losses.append(total_loss.item())
        perplexities.append(info["perplexity"])
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss={total_loss.item():.4f}, "
                  f"VQ={vq_loss['vq'].item():.4f}, "
                  f"Perplexity={info['perplexity']:.2f}, "
                  f"Usage={info['usage']:.3f}, "
                  f"Unique={info['unique_codes']}/256")
    
    # Check final status
    final_loss = losses[-1]
    final_perp = perplexities[-1]
    avg_perp = sum(perplexities[-10:]) / 10
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Perplexity: {final_perp:.2f}")
    print(f"Avg Last 10 Perplexity: {avg_perp:.2f}")
    print(f"Max Loss During Training: {max(losses):.4f}")
    
    # Success criteria
    success = (
        final_loss < 10.0 and  # Loss didn't explode
        max(losses) < 100.0 and  # No explosion during training
        avg_perp > 5.0  # Maintains some diversity
    )
    
    if success:
        print("\n✅ SUCCESS! VQ is stable with fixes applied.")
        print("   - Loss stayed bounded")
        print("   - No numerical explosion")
        print("   - Maintained codebook diversity")
    else:
        print("\n⚠️ VQ still has issues:")
        if final_loss >= 10.0:
            print("   - Loss is too high")
        if max(losses) >= 100.0:
            print("   - Loss exploded during training")
        if avg_perp <= 5.0:
            print("   - Codebook collapsed (low perplexity)")
    
    return success

if __name__ == "__main__":
    success = test_vq_stability()
    exit(0 if success else 1)