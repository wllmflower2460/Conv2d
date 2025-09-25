#!/usr/bin/env python3
"""
Quick test to verify VQ eval mode fix works.
Tests that VQ maintains perplexity during evaluation when kept in training mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.VectorQuantizerEMA2D_Stable import VectorQuantizerEMA2D_Stable

def test_vq_eval_modes():
    """Test VQ behavior in different modes."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    code_dim = 64
    num_codes = 256
    seq_len = 10
    
    # Create VQ layer
    vq = VectorQuantizerEMA2D_Stable(
        num_codes=num_codes,
        code_dim=code_dim,
        decay=0.95,
        commitment_cost=0.4,
        l2_normalize_input=True,
        restart_dead_codes=True
    ).to(device)
    
    # Generate synthetic data
    torch.manual_seed(42)
    
    print("="*60)
    print("Testing VQ Eval Mode Fix")
    print("="*60)
    
    # Train for a few iterations to establish codebook
    print("\n1. Training Phase - Establishing codebook diversity...")
    vq.train()
    
    for i in range(50):
        # Generate diverse data
        x = torch.randn(batch_size, code_dim, seq_len).to(device)
        x = x + torch.randn(1, code_dim, 1).to(device) * 0.5  # Add variation
        
        z_q, losses, info = vq(x)
        
        if i % 10 == 0:
            print(f"  Step {i:3d}: Perplexity={info['perplexity']:.2f}, Usage={info['usage']:.3f}")
    
    print(f"\nFinal training perplexity: {info['perplexity']:.2f}")
    training_perplexity = info['perplexity']
    
    # Test 1: Standard eval mode (BROKEN - codebook collapses)
    print("\n2. Testing STANDARD eval mode (expect collapse)...")
    vq.eval()  # Standard eval - freezes EMA
    
    eval_perplexities_broken = []
    for i in range(20):
        x = torch.randn(batch_size, code_dim, seq_len).to(device)
        with torch.no_grad():
            z_q, losses, info = vq(x)
        eval_perplexities_broken.append(info['perplexity'])
        if i % 5 == 0:
            print(f"  Step {i:3d}: Perplexity={info['perplexity']:.2f}, Usage={info['usage']:.3f}")
    
    mean_perp_broken = np.mean(eval_perplexities_broken)
    print(f"Mean eval perplexity (BROKEN): {mean_perp_broken:.2f}")
    
    # Test 2: Fixed eval mode (VQ stays in training mode)
    print("\n3. Testing FIXED eval mode (VQ in training mode)...")
    vq.train()  # Keep VQ in training mode during eval!
    
    eval_perplexities_fixed = []
    for i in range(20):
        x = torch.randn(batch_size, code_dim, seq_len).to(device)
        with torch.no_grad():
            z_q, losses, info = vq(x)
        eval_perplexities_fixed.append(info['perplexity'])
        if i % 5 == 0:
            print(f"  Step {i:3d}: Perplexity={info['perplexity']:.2f}, Usage={info['usage']:.3f}")
    
    mean_perp_fixed = np.mean(eval_perplexities_fixed)
    print(f"Mean eval perplexity (FIXED): {mean_perp_fixed:.2f}")
    
    # Results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Training perplexity:        {training_perplexity:.2f}")
    print(f"Eval perplexity (BROKEN):   {mean_perp_broken:.2f} " + 
          ("❌ COLLAPSED!" if mean_perp_broken < 5 else ""))
    print(f"Eval perplexity (FIXED):    {mean_perp_fixed:.2f} " +
          ("✅ MAINTAINED!" if mean_perp_fixed > 20 else "⚠️ Still low"))
    print("="*60)
    
    if mean_perp_fixed > mean_perp_broken * 5:
        print("\n✅ SUCCESS: VQ eval mode fix is working!")
        print("   Keeping VQ in training mode prevents codebook collapse.")
    else:
        print("\n⚠️ WARNING: Fix may not be fully effective.")
        print("   Consider additional measures like frozen encoder warmup.")
    
    return {
        "training_perplexity": training_perplexity,
        "eval_broken": mean_perp_broken,
        "eval_fixed": mean_perp_fixed,
        "improvement_ratio": mean_perp_fixed / max(mean_perp_broken, 0.1)
    }

if __name__ == "__main__":
    results = test_vq_eval_modes()
    print(f"\nImprovement ratio: {results['improvement_ratio']:.1f}x")