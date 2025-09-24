#!/usr/bin/env python3
"""
Test FSQ directly to understand the quantization.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_quantize_pytorch import FSQ

def test_fsq_basic():
    """Test basic FSQ functionality."""
    print("="*60)
    print("Testing FSQ Basic Functionality")
    print("="*60)
    
    # Create FSQ with different levels
    fsq = FSQ(levels=[8, 5, 5, 4])
    
    # Test different input ranges
    batch_size = 4
    seq_len = 1
    dim = 4  # Must match number of levels
    
    print("\n1. Testing with normal randn input:")
    x = torch.randn(batch_size, seq_len, dim)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Input sample: {x[0, 0]}")
    
    quantized, codes = fsq(x)
    print(f"Quantized shape: {quantized.shape}")
    print(f"Quantized range: [{quantized.min():.3f}, {quantized.max():.3f}]")
    print(f"Quantized sample: {quantized[0, 0]}")
    print(f"Codes: {codes.squeeze().tolist()}")
    print(f"Unique codes: {torch.unique(codes).numel()}/{batch_size}")
    
    print("\n2. Testing with smaller input range:")
    x = torch.randn(batch_size, seq_len, dim) * 0.1
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    quantized, codes = fsq(x)
    print(f"Quantized range: [{quantized.min():.3f}, {quantized.max():.3f}]")
    print(f"Unique codes: {torch.unique(codes).numel()}/{batch_size}")
    
    print("\n3. Testing with larger input range:")
    x = torch.randn(batch_size, seq_len, dim) * 10
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    quantized, codes = fsq(x)
    print(f"Quantized range: [{quantized.min():.3f}, {quantized.max():.3f}]")
    print(f"Unique codes: {torch.unique(codes).numel()}/{batch_size}")
    
    print("\n4. Testing gradient flow:")
    x = torch.randn(2, 1, 4, requires_grad=True)
    quantized, codes = fsq(x)
    loss = quantized.sum()
    loss.backward()
    print(f"Input grad exists: {x.grad is not None}")
    print(f"Grad norm: {x.grad.norm().item():.3f}")

def test_fsq_with_projection():
    """Test FSQ with learned projection."""
    print("\n" + "="*60)
    print("Testing FSQ with Projection")
    print("="*60)
    
    # Simulate encoder output -> projection -> FSQ
    encoder_dim = 128
    fsq_levels = [8, 5, 5, 4]
    fsq_dim = len(fsq_levels)
    
    # Create layers
    projection = nn.Linear(encoder_dim, fsq_dim)
    fsq = FSQ(levels=fsq_levels)
    
    # Initialize projection with reasonable values
    nn.init.xavier_uniform_(projection.weight)
    nn.init.zeros_(projection.bias)
    
    batch_size = 4
    
    print("\n1. Before projection:")
    encoder_output = torch.randn(batch_size, encoder_dim)
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Encoder output range: [{encoder_output.min():.3f}, {encoder_output.max():.3f}]")
    
    print("\n2. After projection:")
    projected = projection(encoder_output)
    print(f"Projected shape: {projected.shape}")
    print(f"Projected range: [{projected.min():.3f}, {projected.max():.3f}]")
    print(f"Projected sample: {projected[0]}")
    
    print("\n3. After FSQ:")
    projected_3d = projected.unsqueeze(1)
    quantized, codes = fsq(projected_3d)
    quantized = quantized.squeeze(1)
    
    print(f"Quantized shape: {quantized.shape}")
    print(f"Quantized range: [{quantized.min():.3f}, {quantized.max():.3f}]")
    print(f"Quantized sample: {quantized[0]}")
    print(f"Unique codes: {torch.unique(codes).numel()}/{batch_size}")
    
    # Check if all outputs are the same
    if torch.allclose(quantized[0], quantized[1]) and torch.allclose(quantized[0], quantized[2]):
        print("\n⚠️ WARNING: All quantized outputs are the same!")
        print("This explains the 0% accuracy - all inputs map to same code")
    else:
        print("\n✅ Quantized outputs are different")

def test_fsq_levels():
    """Test different FSQ level configurations."""
    print("\n" + "="*60)
    print("Testing Different FSQ Levels")
    print("="*60)
    
    configs = [
        [8, 8, 8, 8],      # More levels per dimension
        [4, 4, 4, 4],      # Fewer levels
        [16, 16],          # 2D with more levels
        [3, 3, 3, 3, 3],   # 5D with few levels
    ]
    
    batch_size = 100
    
    for levels in configs:
        fsq = FSQ(levels=levels)
        dim = len(levels)
        
        # Random input
        x = torch.randn(batch_size, 1, dim)
        quantized, codes = fsq(x)
        
        unique_codes = torch.unique(codes).numel()
        max_codes = 1
        for l in levels:
            max_codes *= l
            
        print(f"\nLevels {levels}: {unique_codes}/{batch_size} unique codes (max {max_codes})")
        print(f"  Quantized range: [{quantized.min():.3f}, {quantized.max():.3f}]")
        
        # Test with projection from 128
        proj = nn.Linear(128, dim)
        encoder_out = torch.randn(batch_size, 128)
        projected = proj(encoder_out).unsqueeze(1)
        quantized, codes = fsq(projected)
        unique_codes = torch.unique(codes).numel()
        print(f"  With projection: {unique_codes}/{batch_size} unique codes")

def main():
    test_fsq_basic()
    test_fsq_with_projection()
    test_fsq_levels()
    
    print("\n" + "="*60)
    print("Diagnosis Complete")
    print("="*60)
    print("\nThe issue is likely that projection to 4D is too restrictive.")
    print("FSQ with [8,5,5,4] expects 4D input, but projecting from 128D to 4D")
    print("loses too much information. Solutions:")
    print("1. Use more FSQ dimensions (e.g., [8]*8 for 8D)")
    print("2. Keep higher dimensional features before FSQ")
    print("3. Use different FSQ levels configuration")

if __name__ == "__main__":
    main()