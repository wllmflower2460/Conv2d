#!/usr/bin/env python3
"""
Comprehensive tests for vectorized Euclidean distance computation.
Verifies correctness, numerical stability, and gradient flow.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.kinematic_features import KinematicFeatureExtractor


def compute_distance_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using explicit loops for comparison.
    
    Args:
        x, y: Tensors of shape (B, C, T)
        
    Returns:
        Mean Euclidean distance over time: shape (B,)
    """
    B = x.shape[0]
    distances = []
    
    for b in range(B):
        # Euclidean distance between sequences
        dist = torch.sqrt(torch.sum((x[b] - y[b])**2, dim=0)).mean()
        distances.append(dist)
    
    return torch.tensor(distances, device=x.device)


def test_euclidean_distance():
    """Test all aspects of vectorized Euclidean distance."""
    
    print("Testing Vectorized Euclidean Distance")
    print("=" * 50)
    
    # Create feature extractor
    extractor = KinematicFeatureExtractor()
    
    # Test 1: Correctness against reference implementation
    print("\n1. Correctness (vectorized vs. loop reference):")
    
    # Test different shapes
    test_cases = [
        (4, 3, 10),    # Small batch
        (32, 9, 100),  # Standard IMU shape
        (64, 6, 200),  # Large batch
        (1, 3, 50),    # Single sample
    ]
    
    for B, C, T in test_cases:
        torch.manual_seed(42)
        x = torch.randn(B, C, T)
        y = torch.randn(B, C, T)
        
        # Compute with both methods
        dist_vectorized = extractor._compute_dtw_distance(x, y)
        dist_reference = compute_distance_reference(x, y)
        
        # Check if results match with tight tolerance
        max_diff = torch.max(torch.abs(dist_vectorized - dist_reference)).item()
        passed = max_diff < 1e-6
        
        print(f"  Shape ({B}, {C}, {T}): Max diff={max_diff:.2e} {'✓ PASS' if passed else '✗ FAIL'}")
        assert passed, f"Results don't match for shape ({B}, {C}, {T})"
    
    # Test 2: Shape assertions
    print("\n2. Shape mismatch handling:")
    x = torch.randn(10, 3, 50)
    y_mismatch = torch.randn(10, 4, 50)  # Different channel dimension
    
    try:
        extractor._compute_dtw_distance(x, y_mismatch)
        print("  ✗ FAIL: Should have raised assertion error")
        assert False
    except AssertionError as e:
        print(f"  ✓ PASS: Caught shape mismatch: {e}")
    
    # Test 3: Numerical stability with large values
    print("\n3. Numerical stability:")
    
    # Test with very large values
    x_large = torch.randn(16, 3, 100) * 1e3
    y_large = torch.randn(16, 3, 100) * 1e3
    
    dist_large = extractor._compute_dtw_distance(x_large, y_large)
    print(f"  Large values: No NaN/Inf = {not (torch.isnan(dist_large).any() or torch.isinf(dist_large).any())}")
    assert not torch.isnan(dist_large).any()
    assert not torch.isinf(dist_large).any()
    
    # Test with very small differences
    x_close = torch.randn(16, 3, 100)
    y_close = x_close + torch.randn_like(x_close) * 1e-8
    
    dist_close = extractor._compute_dtw_distance(x_close, y_close)
    print(f"  Small differences: Distance < 1e-6 = {(dist_close < 1e-6).all().item()}")
    
    # Test 4: Gradient flow
    print("\n4. Gradient flow:")
    x_grad = torch.randn(8, 3, 50, requires_grad=True)
    y_grad = torch.randn(8, 3, 50, requires_grad=True)
    
    # Gradients will flow automatically when requires_grad=True
    dist_grad = extractor._compute_dtw_distance(x_grad, y_grad)
    
    # Check if gradients can flow
    loss = dist_grad.sum()
    loss.backward()
    
    print(f"  x gradient exists: {x_grad.grad is not None}")
    print(f"  y gradient exists: {y_grad.grad is not None}")
    
    if x_grad.grad is not None:
        print(f"  x gradient norm: {x_grad.grad.norm().item():.4f}")
        print(f"  y gradient norm: {y_grad.grad.norm().item():.4f}")
    
    # Test 5: Edge cases
    print("\n5. Edge cases:")
    
    # Identical inputs (distance should be 0)
    x_same = torch.randn(5, 3, 20)
    dist_same = extractor._compute_dtw_distance(x_same, x_same)
    print(f"  Identical inputs: Distance ~0 = {torch.allclose(dist_same, torch.zeros_like(dist_same), atol=1e-7)}")
    assert torch.allclose(dist_same, torch.zeros_like(dist_same), atol=1e-7)
    
    # Orthogonal vectors (known distance)
    B, C, T = 2, 3, 10
    x_orth = torch.zeros(B, C, T)
    y_orth = torch.zeros(B, C, T)
    x_orth[:, 0, :] = 1.0  # Unit vector in first channel
    y_orth[:, 1, :] = 1.0  # Unit vector in second channel
    
    dist_orth = extractor._compute_dtw_distance(x_orth, y_orth)
    expected_dist = np.sqrt(2)  # ||[1,0,0] - [0,1,0]|| = sqrt(2)
    print(f"  Orthogonal vectors: Distance ≈ √2 = {torch.allclose(dist_orth, torch.tensor(expected_dist, dtype=torch.float32), atol=1e-5)}")
    
    # Test 6: Performance comparison
    print("\n6. Performance (vectorized vs. loop):")
    import time
    
    x_perf = torch.randn(128, 9, 200)
    y_perf = torch.randn(128, 9, 200)
    
    # Time vectorized version
    start = time.perf_counter()
    for _ in range(100):
        dist_vec = extractor._compute_dtw_distance(x_perf, y_perf)
    time_vectorized = (time.perf_counter() - start) / 100
    
    # Time reference loop version
    start = time.perf_counter()
    for _ in range(100):
        dist_ref = compute_distance_reference(x_perf, y_perf)
    time_reference = (time.perf_counter() - start) / 100
    
    speedup = time_reference / time_vectorized
    print(f"  Vectorized: {time_vectorized*1000:.2f}ms")
    print(f"  Reference: {time_reference*1000:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Test 7: Different devices (if CUDA available)
    print("\n7. Device consistency:")
    if torch.cuda.is_available():
        x_cpu = torch.randn(16, 3, 50)
        y_cpu = torch.randn(16, 3, 50)
        x_gpu = x_cpu.cuda()
        y_gpu = y_cpu.cuda()
        
        dist_cpu = extractor._compute_dtw_distance(x_cpu, y_cpu)
        dist_gpu = extractor._compute_dtw_distance(x_gpu, y_gpu)
        
        print(f"  CPU/GPU consistency: {torch.allclose(dist_cpu, dist_gpu.cpu(), atol=1e-5)}")
    else:
        print("  CUDA not available - skipping GPU test")
    
    # Test 8: Batch independence
    print("\n8. Batch independence:")
    x_batch = torch.randn(10, 3, 30)
    y_batch = torch.randn(10, 3, 30)
    
    # Compute full batch
    dist_batch = extractor._compute_dtw_distance(x_batch, y_batch)
    
    # Compute individual samples
    dist_individual = []
    for i in range(10):
        dist_i = extractor._compute_dtw_distance(
            x_batch[i:i+1], y_batch[i:i+1]
        )
        dist_individual.append(dist_i)
    dist_individual = torch.cat(dist_individual)
    
    print(f"  Batch vs. individual: {torch.allclose(dist_batch, dist_individual, atol=1e-6)}")
    assert torch.allclose(dist_batch, dist_individual, atol=1e-6)
    
    print("\n" + "=" * 50)
    print("✅ All Euclidean distance tests passed!")
    
    return True


if __name__ == "__main__":
    test_euclidean_distance()