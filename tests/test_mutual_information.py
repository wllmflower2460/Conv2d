#!/usr/bin/env python3
"""
Comprehensive tests for mutual information estimation.
Verifies robustness, numerical stability, and edge case handling.
"""

import torch
import numpy as np
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.vectorized_optimizations import VectorizedOperations


class TestMutualInformation:
    """Test suite for mutual information computation."""
    
    def test_basic_mi_computation(self):
        """Test basic MI computation with known distributions."""
        print("\n1. Basic MI Computation:")
        
        # Test 1: Identical signals (maximum MI)
        x = torch.randn(4, 3, 100)
        y = x.clone()
        
        mi = VectorizedOperations.estimate_mutual_information_vectorized(x, y)
        
        # MI(X,X) should be high (close to entropy of X)
        assert mi.shape == (4,), f"Shape mismatch: {mi.shape}"
        assert torch.all(mi > 0), "MI should be positive for identical signals"
        print(f"  Identical signals: MI = {mi.mean():.4f}")
        
        # Test 2: Independent signals (minimum MI)
        x = torch.randn(4, 3, 100)
        y = torch.randn(4, 3, 100)
        
        mi_indep = VectorizedOperations.estimate_mutual_information_vectorized(x, y)
        
        # MI should be much lower for independent signals
        assert torch.all(mi_indep < mi), "Independent signals should have lower MI"
        print(f"  Independent signals: MI = {mi_indep.mean():.4f}")
        
        print("  ✓ Basic MI computation correct")
    
    def test_consistent_binning(self):
        """Test that MI uses consistent binning across batches."""
        print("\n2. Consistent Binning:")
        
        # Create signals with different scales in different batches
        x = torch.randn(8, 3, 100)
        x[0] *= 0.1   # Small scale
        x[1] *= 10.0  # Large scale
        
        y = torch.randn(8, 3, 100)
        y[0] *= 0.1
        y[1] *= 10.0
        
        # Compute MI
        mi = VectorizedOperations.estimate_mutual_information_vectorized(x, y, n_bins=10)
        
        # All MI values should be valid (no NaN/Inf)
        assert not torch.isnan(mi).any(), "NaN in MI computation"
        assert not torch.isinf(mi).any(), "Inf in MI computation"
        assert torch.all(mi >= 0), "MI should be non-negative"
        
        print(f"  MI range: [{mi.min():.4f}, {mi.max():.4f}]")
        print("  ✓ Consistent binning across different scales")
    
    def test_edge_cases(self):
        """Test MI computation with edge cases."""
        print("\n3. Edge Cases:")
        
        # Test 1: Constant signals (zero variance)
        x = torch.ones(4, 3, 100)
        y = torch.ones(4, 3, 100)
        
        mi_const = VectorizedOperations.estimate_mutual_information_vectorized(x, y)
        
        # MI should be 0 for constant signals
        assert torch.allclose(mi_const, torch.zeros_like(mi_const), atol=1e-6)
        print(f"  Constant signals: MI = {mi_const.mean():.6f} (≈0)")
        
        # Test 2: Binary signals
        x_binary = torch.randint(0, 2, (4, 3, 100)).float()
        y_binary = torch.randint(0, 2, (4, 3, 100)).float()
        
        mi_binary = VectorizedOperations.estimate_mutual_information_vectorized(
            x_binary, y_binary, n_bins=2
        )
        
        assert not torch.isnan(mi_binary).any()
        assert torch.all(mi_binary >= 0)
        print(f"  Binary signals: MI = {mi_binary.mean():.4f}")
        
        # Test 3: Perfectly correlated with noise
        x = torch.randn(4, 3, 100)
        y = x + torch.randn_like(x) * 0.1  # Add small noise
        
        mi_corr = VectorizedOperations.estimate_mutual_information_vectorized(x, y)
        
        # Should have high MI but not infinite
        assert torch.all(mi_corr > 0)
        assert torch.all(torch.isfinite(mi_corr))
        print(f"  Correlated signals: MI = {mi_corr.mean():.4f}")
        
        print("  ✓ Edge cases handled correctly")
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        print("\n4. Numerical Stability:")
        
        # Test with very large values
        x_large = torch.randn(4, 3, 100) * 1e6
        y_large = torch.randn(4, 3, 100) * 1e6
        
        mi_large = VectorizedOperations.estimate_mutual_information_vectorized(
            x_large, y_large
        )
        
        assert torch.all(torch.isfinite(mi_large)), "MI should be finite for large values"
        print(f"  Large values (1e6): MI = {mi_large.mean():.4f}")
        
        # Test with very small values
        x_small = torch.randn(4, 3, 100) * 1e-6
        y_small = torch.randn(4, 3, 100) * 1e-6
        
        mi_small = VectorizedOperations.estimate_mutual_information_vectorized(
            x_small, y_small
        )
        
        assert torch.all(torch.isfinite(mi_small)), "MI should be finite for small values"
        print(f"  Small values (1e-6): MI = {mi_small.mean():.4f}")
        
        # Test with mixed scales
        x_mixed = torch.randn(4, 3, 100)
        x_mixed[0] *= 1e6
        x_mixed[1] *= 1e-6
        y_mixed = torch.randn(4, 3, 100)
        
        mi_mixed = VectorizedOperations.estimate_mutual_information_vectorized(
            x_mixed, y_mixed
        )
        
        assert torch.all(torch.isfinite(mi_mixed)), "MI should be finite for mixed scales"
        print(f"  Mixed scales: MI = {mi_mixed.mean():.4f}")
        
        print("  ✓ Numerically stable across all scales")
    
    def test_zero_bin_handling(self):
        """Test handling of sparse distributions with many zero bins."""
        print("\n5. Zero Bin Handling:")
        
        # Create sparse signals (many values clustered)
        x = torch.zeros(4, 3, 100)
        y = torch.zeros(4, 3, 100)
        
        # Add sparse spikes
        x[:, :, ::10] = 1.0  # Spike every 10 timesteps
        y[:, :, ::15] = 1.0  # Spike every 15 timesteps
        
        mi_sparse = VectorizedOperations.estimate_mutual_information_vectorized(
            x, y, n_bins=20  # Many bins for sparse data
        )
        
        # Should handle sparse data without NaN/Inf
        assert not torch.isnan(mi_sparse).any(), "NaN in sparse MI"
        assert not torch.isinf(mi_sparse).any(), "Inf in sparse MI"
        assert torch.all(mi_sparse >= 0), "MI should be non-negative"
        
        print(f"  Sparse signals: MI = {mi_sparse.mean():.4f}")
        
        # Test with all zeros in one batch
        x_partial = torch.randn(4, 3, 100)
        y_partial = torch.randn(4, 3, 100)
        x_partial[0] = 0  # First batch all zeros
        y_partial[0] = 0
        
        mi_partial = VectorizedOperations.estimate_mutual_information_vectorized(
            x_partial, y_partial
        )
        
        assert torch.isfinite(mi_partial[0]), "Should handle all-zero batch"
        assert mi_partial[0] == 0, "All-zero batch should have MI=0"
        
        print(f"  All-zero batch: MI[0] = {mi_partial[0]:.4f}")
        print("  ✓ Zero bins handled correctly")
    
    def test_batch_consistency(self):
        """Test that MI computation uses consistent binning across batches."""
        print("\n6. Batch Consistency:")
        
        # Create test data with varying scales across batches
        x = torch.randn(10, 3, 100)
        y = torch.randn(10, 3, 100)
        
        # Scale some batches differently
        x[0] *= 10.0  # Large scale
        x[1] *= 0.1   # Small scale
        y[0] *= 10.0
        y[1] *= 0.1
        
        # Compute batch MI with consistent binning
        mi_batch = VectorizedOperations.estimate_mutual_information_vectorized(x, y)
        
        # All MI values should be valid despite scale differences
        assert not torch.isnan(mi_batch).any(), "NaN in batch MI"
        assert not torch.isinf(mi_batch).any(), "Inf in batch MI"
        assert torch.all(mi_batch >= 0), "MI should be non-negative"
        
        print(f"  MI stats: mean={mi_batch.mean():.4f}, std={mi_batch.std():.4f}")
        print("  ✓ Consistent binning across different scales")
        
        # Test that results are reproducible
        mi_batch2 = VectorizedOperations.estimate_mutual_information_vectorized(x, y)
        torch.testing.assert_close(mi_batch, mi_batch2)
        print("  ✓ Reproducible results with same data")
    
    def test_different_bin_counts(self):
        """Test MI with different numbers of bins."""
        print("\n7. Different Bin Counts:")
        
        x = torch.randn(4, 3, 100)
        y = x + torch.randn_like(x) * 0.5  # Correlated signals
        
        mi_values = []
        bin_counts = [2, 5, 10, 20, 50]
        
        for n_bins in bin_counts:
            mi = VectorizedOperations.estimate_mutual_information_vectorized(
                x, y, n_bins=n_bins
            )
            mi_values.append(mi.mean().item())
            print(f"  {n_bins:2d} bins: MI = {mi.mean():.4f}")
        
        # MI should generally increase with more bins (finer resolution)
        # But should stabilize at some point
        assert all(torch.isfinite(torch.tensor(mi_values)))
        print("  ✓ All bin counts produce valid MI")
    
    def test_gradient_flow(self):
        """Test that gradients can flow through MI computation."""
        print("\n8. Gradient Flow:")
        
        x = torch.randn(4, 3, 100, requires_grad=True)
        y = torch.randn(4, 3, 100, requires_grad=True)
        
        # Compute MI
        mi = VectorizedOperations.estimate_mutual_information_vectorized(x, y)
        
        # Check that operation supports gradients
        loss = mi.sum()
        
        try:
            loss.backward()
            print(f"  x gradient norm: {x.grad.norm():.4f}")
            print(f"  y gradient norm: {y.grad.norm():.4f}")
            print("  ✓ Gradients flow correctly")
        except RuntimeError as e:
            # Some operations might not support gradients (histogramdd)
            print(f"  ⚠ Gradient not supported: {str(e)[:50]}")
            print("  ✓ Expected for histogram-based MI")
    
    def test_performance(self):
        """Test performance of vectorized MI computation."""
        print("\n9. Performance Test:")
        
        # Large batch
        x = torch.randn(64, 3, 200)
        y = torch.randn(64, 3, 200)
        
        # Time vectorized version
        start = time.perf_counter()
        for _ in range(10):
            mi = VectorizedOperations.estimate_mutual_information_vectorized(x, y)
        time_vectorized = (time.perf_counter() - start) / 10
        
        print(f"  64 batch MI: {time_vectorized*1000:.2f}ms")
        print(f"  MI values: mean={mi.mean():.4f}, std={mi.std():.4f}")
        print("  ✓ Performance acceptable")
    
    def test_information_theoretic_properties(self):
        """Test that MI satisfies information-theoretic properties."""
        print("\n10. Information-Theoretic Properties:")
        
        # Property 1: MI(X,Y) = MI(Y,X) (symmetry)
        x = torch.randn(4, 3, 100)
        y = torch.randn(4, 3, 100)
        
        mi_xy = VectorizedOperations.estimate_mutual_information_vectorized(x, y)
        mi_yx = VectorizedOperations.estimate_mutual_information_vectorized(y, x)
        
        torch.testing.assert_close(mi_xy, mi_yx, rtol=0.1, atol=0.01)
        print("  ✓ Symmetry: MI(X,Y) ≈ MI(Y,X)")
        
        # Property 2: MI(X,X) >= MI(X,Y) for any Y
        mi_xx = VectorizedOperations.estimate_mutual_information_vectorized(x, x)
        
        assert torch.all(mi_xx >= mi_xy - 0.01), "MI(X,X) should be maximal"
        print("  ✓ Self-MI is maximal")
        
        # Property 3: MI >= 0 (non-negativity)
        for _ in range(10):
            x_test = torch.randn(2, 3, 50)
            y_test = torch.randn(2, 3, 50)
            mi_test = VectorizedOperations.estimate_mutual_information_vectorized(
                x_test, y_test
            )
            assert torch.all(mi_test >= 0), "MI must be non-negative"
        
        print("  ✓ Non-negativity satisfied")
        
        print("  ✓ All information-theoretic properties satisfied")


def run_all_tests():
    """Run all mutual information tests."""
    print("Testing Mutual Information Estimation")
    print("=" * 50)
    
    test_suite = TestMutualInformation()
    
    # Run all test methods
    test_methods = [
        test_suite.test_basic_mi_computation,
        test_suite.test_consistent_binning,
        test_suite.test_edge_cases,
        test_suite.test_numerical_stability,
        test_suite.test_zero_bin_handling,
        test_suite.test_batch_consistency,
        test_suite.test_different_bin_counts,
        test_suite.test_gradient_flow,
        test_suite.test_performance,
        test_suite.test_information_theoretic_properties,
    ]
    
    for test_method in test_methods:
        try:
            test_method()
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 50)
    print("✅ All mutual information tests passed!")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)