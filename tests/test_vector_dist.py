#!/usr/bin/env python3
"""
Fast test comparing vectorized vs reference loop distance computation.
Fixed seed for reproducibility.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.kinematic_features import KinematicFeatureExtractor


def compute_distance_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference implementation using explicit loops."""
    B = x.shape[0]
    distances = []
    
    for b in range(B):
        # Euclidean distance at each timestep
        dist_per_time = torch.sqrt(torch.sum((x[b] - y[b])**2, dim=0))
        # Mean over time
        distances.append(dist_per_time.mean())
    
    return torch.stack(distances)


def test_vectorized_vs_reference():
    """Test vectorized matches reference with fixed seed."""
    # Fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Standard test case: B=7, C=9, T=100
    B, C, T = 7, 9, 100
    x = torch.randn(B, C, T)
    y = torch.randn(B, C, T)
    
    # Create extractor
    extractor = KinematicFeatureExtractor()
    
    # Compute with both methods
    dist_vectorized = extractor._compute_dtw_distance(x, y)
    dist_reference = compute_distance_reference(x, y)
    
    # Check shapes
    assert dist_vectorized.shape == (B,), f"Wrong shape: {dist_vectorized.shape}"
    assert dist_reference.shape == (B,), f"Wrong shape: {dist_reference.shape}"
    
    # Compare values
    max_diff = torch.max(torch.abs(dist_vectorized - dist_reference)).item()
    assert max_diff < 1e-6, f"Max diff too large: {max_diff}"
    
    # Print some values for manual inspection
    print(f"First 3 distances (vectorized): {dist_vectorized[:3].tolist()}")
    print(f"First 3 distances (reference):  {dist_reference[:3].tolist()}")
    print(f"Max difference: {max_diff:.2e}")
    
    print("✓ Vectorized matches reference")


def test_shape_mismatch():
    """Test that shape mismatch is caught."""
    extractor = KinematicFeatureExtractor()
    
    x = torch.randn(5, 3, 50)
    y = torch.randn(5, 4, 50)  # Wrong channel dimension
    
    try:
        extractor._compute_dtw_distance(x, y)
        assert False, "Should have raised assertion error"
    except AssertionError as e:
        assert "Shape mismatch" in str(e) or "shape" in str(e).lower()
        print("✓ Shape mismatch caught correctly")


def test_identical_signals():
    """Test distance is 0 for identical signals."""
    extractor = KinematicFeatureExtractor()
    
    x = torch.randn(3, 6, 80)
    dist = extractor._compute_dtw_distance(x, x)
    
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-7), \
        f"Distance not ~0 for identical: {dist}"
    
    print("✓ Identical signals have distance ~0")


def test_known_distance():
    """Test with known distance case."""
    extractor = KinematicFeatureExtractor()
    
    # Orthogonal unit vectors
    B, C, T = 2, 3, 50
    x = torch.zeros(B, C, T)
    y = torch.zeros(B, C, T)
    
    x[:, 0, :] = 1.0  # Unit in first channel
    y[:, 1, :] = 1.0  # Unit in second channel
    
    dist = extractor._compute_dtw_distance(x, y)
    
    # Distance should be sqrt(2) for orthogonal unit vectors
    expected = np.sqrt(2)
    assert torch.allclose(dist, torch.tensor(expected, dtype=dist.dtype).expand_as(dist), atol=1e-5), \
        f"Expected {expected}, got {dist[0].item()}"
    
    print("✓ Known distance case correct")


def test_performance():
    """Quick performance comparison."""
    import time
    
    torch.manual_seed(42)
    x = torch.randn(32, 9, 200)
    y = torch.randn(32, 9, 200)
    
    extractor = KinematicFeatureExtractor()
    
    # Time vectorized
    start = time.perf_counter()
    for _ in range(100):
        dist_vec = extractor._compute_dtw_distance(x, y)
    time_vec = (time.perf_counter() - start) / 100
    
    # Time reference
    start = time.perf_counter()
    for _ in range(100):
        dist_ref = compute_distance_reference(x, y)
    time_ref = (time.perf_counter() - start) / 100
    
    speedup = time_ref / time_vec
    print(f"Vectorized: {time_vec*1000:.2f}ms")
    print(f"Reference:  {time_ref*1000:.2f}ms")
    print(f"Speedup: {speedup:.1f}x")
    print("✓ Performance test complete")


if __name__ == "__main__":
    print("Testing Vectorized Distance (Fast)")
    print("=" * 40)
    
    test_vectorized_vs_reference()
    test_shape_mismatch()
    test_identical_signals()
    test_known_distance()
    test_performance()
    
    print("=" * 40)
    print("✅ All fast distance tests passed!")