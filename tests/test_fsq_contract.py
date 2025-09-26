#!/usr/bin/env python3
"""Comprehensive tests for FSQ encoding contract.

Tests verify:
1. Determinism guarantee (same input → same codes)
2. Shape contracts are enforced
3. Type contracts (int32/float32) are maintained
4. Invariant checks work correctly
5. Statistics tracking is accurate
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from conv2d.features.fsq_contract import (
    CodesAndFeatures,
    FSQEncoder,
    encode_fsq,
    verify_fsq_invariants,
)


class TestFSQContract:
    """Test FSQ encoding contract guarantees."""
    
    def test_determinism(self):
        """Test deterministic encoding: same input + same seed → identical codes."""
        # Create test input
        torch.manual_seed(42)
        x = torch.randn(32, 9, 2, 100, dtype=torch.float32)
        
        # Encode multiple times
        result1 = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        result2 = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        result3 = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        
        # Verify identical codes
        assert torch.equal(result1.codes, result2.codes), "Codes not deterministic"
        assert torch.equal(result2.codes, result3.codes), "Codes not deterministic"
        
        # Verify identical features
        assert torch.allclose(result1.quantized, result2.quantized, atol=1e-7), "Features not deterministic"
        assert torch.allclose(result2.quantized, result3.quantized, atol=1e-7), "Features not deterministic"
        
    def test_shape_contract(self):
        """Test shape contracts: (B,9,2,100) → (B,T) codes, (B,D,T) features."""
        batch_sizes = [1, 8, 32]
        
        for B in batch_sizes:
            x = torch.randn(B, 9, 2, 100, dtype=torch.float32)
            result = encode_fsq(x, embedding_dim=64)
            
            # Check code shape
            assert result.codes.shape == (B, 100), f"Code shape wrong for B={B}"
            
            # Check feature shape
            assert result.quantized.shape == (B, 64, 100), f"Feature shape wrong for B={B}"
            
    def test_dtype_contract(self):
        """Test dtype contracts: int32 codes, float32 features."""
        x = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        result = encode_fsq(x)
        
        # Check dtypes
        assert result.codes.dtype == torch.int32, f"Codes dtype {result.codes.dtype} != int32"
        assert result.quantized.dtype == torch.float32, f"Features dtype {result.quantized.dtype} != float32"
        
    def test_wrong_input_shape(self):
        """Test that wrong input shapes are rejected."""
        # Wrong number of channels
        try:
            x = torch.randn(16, 8, 2, 100)  # 8 channels instead of 9
            encode_fsq(x)
            assert False, "Should have raised error for wrong channels"
        except AssertionError as e:
            assert "Expected 9 channels" in str(e)
            
        # Wrong number of sensors
        try:
            x = torch.randn(16, 9, 3, 100)  # 3 sensors instead of 2
            encode_fsq(x)
            assert False, "Should have raised error for wrong sensors"
        except AssertionError as e:
            assert "Expected 2 sensors" in str(e)
            
        # Wrong number of timesteps
        try:
            x = torch.randn(16, 9, 2, 50)  # 50 timesteps instead of 100
            encode_fsq(x)
            assert False, "Should have raised error for wrong timesteps"
        except AssertionError as e:
            assert "Expected 100 timesteps" in str(e)
            
    def test_code_range(self):
        """Test that codes are in valid range [0, codebook_size)."""
        x = torch.randn(32, 9, 2, 100, dtype=torch.float32)
        levels = [8, 6, 5]
        codebook_size = 8 * 6 * 5  # 240
        
        result = encode_fsq(x, levels=levels)
        
        # Check code range
        assert result.codes.min() >= 0, f"Negative code found: {result.codes.min()}"
        assert result.codes.max() < codebook_size, f"Code {result.codes.max()} >= {codebook_size}"
        assert result.codebook_size == codebook_size, f"Codebook size mismatch"
        
    def test_perplexity_tracking(self):
        """Test that perplexity tracking works correctly."""
        x = torch.randn(100, 9, 2, 100, dtype=torch.float32)
        
        # First encoding
        result1 = encode_fsq(x, reset_stats=True)
        perplexity1 = result1.perplexity
        
        # Second encoding (accumulates stats)
        result2 = encode_fsq(x, reset_stats=False)
        perplexity2 = result2.perplexity
        
        # Perplexity should be positive
        assert perplexity1 > 0, "Perplexity should be positive"
        assert perplexity2 > 0, "Perplexity should be positive"
        
        # With reset, should get same perplexity
        result3 = encode_fsq(x, reset_stats=True)
        assert np.isclose(result3.perplexity, perplexity1, rtol=0.01), "Reset stats not working"
        
    def test_usage_histogram(self):
        """Test code usage histogram."""
        x = torch.randn(64, 9, 2, 100, dtype=torch.float32)
        result = encode_fsq(x)
        
        # Check histogram
        assert result.usage_histogram is not None, "No usage histogram"
        assert len(result.usage_histogram) == result.codebook_size, "Histogram size mismatch"
        assert np.allclose(result.usage_histogram.sum(), 1.0, rtol=0.01), "Histogram not normalized"
        assert np.all(result.usage_histogram >= 0), "Negative histogram values"
        
    def test_invariant_verification(self):
        """Test that invariant verification works."""
        # Create encoder and input
        encoder = FSQEncoder(levels=[8, 6, 5], seed=42)
        x = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        
        # Verify invariants
        is_valid = verify_fsq_invariants(encoder, x)
        assert is_valid, "Invariants not satisfied"
        
    def test_different_levels(self):
        """Test encoding with different quantization levels."""
        x = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        
        # Test different level configurations
        level_configs = [
            [8, 6, 5],  # Default: 240 codes
            [4, 4, 4],  # Small: 64 codes
            [16, 12, 10],  # Large: 1920 codes
            [8, 8],  # 2D: 64 codes
            [8, 6, 5, 4],  # 4D: 960 codes
        ]
        
        for levels in level_configs:
            result = encode_fsq(x, levels=levels)
            expected_size = np.prod(levels)
            assert result.codebook_size == expected_size, f"Wrong codebook size for {levels}"
            assert result.codes.max() < expected_size, f"Code out of range for {levels}"
            
    def test_embedding_dimensions(self):
        """Test different embedding dimensions."""
        x = torch.randn(8, 9, 2, 100, dtype=torch.float32)
        
        for embedding_dim in [32, 64, 128, 256]:
            result = encode_fsq(x, embedding_dim=embedding_dim)
            assert result.quantized.shape[1] == embedding_dim, f"Wrong embedding dim {embedding_dim}"
            
    def test_gradient_flow(self):
        """Test that gradients flow through straight-through estimator."""
        # Create encoder with gradient tracking
        encoder = FSQEncoder(levels=[8, 6, 5], seed=42)
        encoder.train()
        
        # Create input with gradients
        x = torch.randn(8, 9, 2, 100, dtype=torch.float32, requires_grad=True)
        
        # Forward pass
        result = encoder(x)
        
        # Compute loss and backward
        loss = result.quantized.mean()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None, "No gradients on input"
        assert not torch.all(x.grad == 0), "Zero gradients"
        
    def test_output_validation(self):
        """Test that CodesAndFeatures validates its outputs."""
        # Create valid output
        codes = torch.randint(0, 240, (16, 100), dtype=torch.int32)
        quantized = torch.randn(16, 64, 100, dtype=torch.float32)
        histogram = np.random.rand(240)
        histogram /= histogram.sum()
        
        # This should work
        output = CodesAndFeatures(
            codes=codes,
            quantized=quantized,
            perplexity=50.0,
            usage_histogram=histogram,
            codebook_size=240,
        )
        
        # Test wrong dtype for codes
        try:
            CodesAndFeatures(
                codes=codes.float(),  # Wrong dtype
                quantized=quantized,
                perplexity=50.0,
                usage_histogram=histogram,
                codebook_size=240,
            )
            assert False, "Should have raised error for wrong dtype"
        except AssertionError as e:
            assert "codes must be int32" in str(e)
            
        # Test shape mismatch
        try:
            CodesAndFeatures(
                codes=codes[:8],  # Different batch size
                quantized=quantized,
                perplexity=50.0,
                usage_histogram=histogram,
                codebook_size=240,
            )
            assert False, "Should have raised error for shape mismatch"
        except AssertionError as e:
            assert "Batch size mismatch" in str(e)
            
        # Test code out of range
        bad_codes = codes.clone()
        bad_codes[0, 0] = 240  # Out of range
        try:
            CodesAndFeatures(
                codes=bad_codes,
                quantized=quantized,
                perplexity=50.0,
                usage_histogram=histogram,
                codebook_size=240,
            )
            assert False, "Should have raised error for code out of range"
        except AssertionError as e:
            assert "240 >= codebook size" in str(e)


class TestFSQStability:
    """Test stability guarantees of FSQ encoding."""
    
    def test_batch_independence(self):
        """Test that batch samples are encoded independently."""
        torch.manual_seed(42)
        
        # Create test data
        x1 = torch.randn(1, 9, 2, 100, dtype=torch.float32)
        x2 = torch.randn(1, 9, 2, 100, dtype=torch.float32)
        x_batch = torch.cat([x1, x2], dim=0)
        
        # Encode separately
        result1 = encode_fsq(x1, reset_stats=True)
        result2 = encode_fsq(x2, reset_stats=True)
        
        # Encode as batch
        result_batch = encode_fsq(x_batch, reset_stats=True)
        
        # Check codes match
        assert torch.equal(result1.codes, result_batch.codes[0:1]), "Batch encoding differs"
        assert torch.equal(result2.codes, result_batch.codes[1:2]), "Batch encoding differs"
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        x_small = torch.randn(8, 9, 2, 100) * 1e-8
        result_small = encode_fsq(x_small.float())
        assert not torch.isnan(result_small.quantized).any(), "NaN in small value encoding"
        assert not torch.isinf(result_small.quantized).any(), "Inf in small value encoding"
        
        # Test with very large values
        x_large = torch.randn(8, 9, 2, 100) * 1e8
        result_large = encode_fsq(x_large.float())
        assert not torch.isnan(result_large.quantized).any(), "NaN in large value encoding"
        assert not torch.isinf(result_large.quantized).any(), "Inf in large value encoding"
        
        # Test with mixed scales
        x_mixed = torch.randn(8, 9, 2, 100)
        x_mixed[0] *= 1e-6
        x_mixed[1] *= 1e6
        result_mixed = encode_fsq(x_mixed.float())
        assert not torch.isnan(result_mixed.quantized).any(), "NaN in mixed scale encoding"
        
    def test_zero_input(self):
        """Test encoding of zero input."""
        x_zero = torch.zeros(16, 9, 2, 100, dtype=torch.float32)
        result = encode_fsq(x_zero)
        
        # Should produce valid codes
        assert result.codes.min() >= 0, "Invalid codes for zero input"
        assert result.codes.max() < result.codebook_size, "Invalid codes for zero input"
        
        # Should be deterministic
        result2 = encode_fsq(x_zero)
        assert torch.equal(result.codes, result2.codes), "Zero encoding not deterministic"
        
    def test_constant_input(self):
        """Test encoding of constant input."""
        x_const = torch.ones(16, 9, 2, 100, dtype=torch.float32) * 3.14
        result = encode_fsq(x_const)
        
        # Should produce valid codes
        assert result.codes.min() >= 0, "Invalid codes for constant input"
        assert result.codes.max() < result.codebook_size, "Invalid codes for constant input"
        
        # All samples should encode similarly (but not necessarily identically due to encoder)
        # Check that perplexity is low (few codes used)
        assert result.perplexity < 50, "High perplexity for constant input"


class TestFSQPerformance:
    """Performance and efficiency tests."""
    
    def test_memory_efficiency(self):
        """Test that encoding doesn't create unnecessary copies."""
        x = torch.randn(32, 9, 2, 100, dtype=torch.float32)
        
        # Skip if no CUDA
        if not torch.cuda.is_available():
            print("(skipped - no CUDA)")
            return
            
        # Get initial memory
        torch.cuda.reset_peak_memory_stats()
        x = x.cuda()
        initial_memory = torch.cuda.memory_allocated()
        
        # Encode
        result = encode_fsq(x)
        
        # Check memory usage is reasonable
        peak_memory = torch.cuda.max_memory_allocated()
        memory_increase = peak_memory - initial_memory
        
        # Should not use more than 2x the input size
        input_bytes = x.numel() * 4  # float32
        assert memory_increase < 2 * input_bytes, "Excessive memory usage"
            
    def test_large_batch(self):
        """Test encoding large batches."""
        # Test with large batch
        x_large = torch.randn(256, 9, 2, 100, dtype=torch.float32)
        
        # Should complete without error
        result = encode_fsq(x_large)
        assert result.codes.shape == (256, 100), "Large batch shape wrong"
        
    def test_encode_speed(self):
        """Test that encoding is reasonably fast."""
        import time
        
        x = torch.randn(64, 9, 2, 100, dtype=torch.float32)
        
        # Warm-up
        _ = encode_fsq(x, reset_stats=True)
        
        # Time encoding
        start = time.time()
        for _ in range(10):
            _ = encode_fsq(x, reset_stats=False)
        elapsed = time.time() - start
        
        # Should be fast (< 100ms per call on CPU)
        ms_per_call = (elapsed / 10) * 1000
        print(f"Encoding speed: {ms_per_call:.2f}ms per call")
        assert ms_per_call < 100, f"Encoding too slow: {ms_per_call:.2f}ms"


def test_fsq_integration():
    """Test FSQ integration with data pipeline."""
    from conv2d.data import Compose, Standardize, ToTensor
    
    # Create pipeline (no windowing for this test) 
    pipeline = Compose([
        Standardize(),
        ToTensor(),
    ])
    
    # Create data with shape (B, C, T) where T=100 for FSQ
    X = np.random.randn(50, 9, 100).astype(np.float32)
    
    # Process through pipeline
    X_processed = pipeline.fit_transform(X)
    
    # Add sensor dimension for FSQ (B, C, T) -> (B, C, S=2, T)
    B, C, T = X_processed.shape
    X_reshaped = X_processed.unsqueeze(2)  # Add sensor dimension
    X_reshaped = X_reshaped.expand(B, C, 2, T)  # Duplicate for 2 sensors
    
    # Encode with FSQ
    result = encode_fsq(X_reshaped)
    
    # Check output
    assert result.codes.shape == (B, T)
    assert result.quantized.shape == (B, 64, T)
    assert result.perplexity > 0


def run_tests():
    """Run all tests and report results."""
    import traceback
    
    test_classes = [TestFSQContract, TestFSQStability, TestFSQPerformance]
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print(f"{'='*60}")
        
        test_instance = test_class()
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                total_tests += 1
                test_method = getattr(test_instance, method_name)
                
                try:
                    print(f"\n  {method_name}...", end=" ")
                    test_method()
                    print("PASSED ✓")
                    passed_tests += 1
                except Exception as e:
                    print(f"FAILED ✗")
                    print(f"    Error: {e}")
                    failed_tests.append((method_name, traceback.format_exc()))
    
    # Test integration
    print(f"\n{'='*60}")
    print("Running Integration Tests")
    print(f"{'='*60}")
    
    try:
        print(f"\n  test_fsq_integration...", end=" ")
        test_fsq_integration()
        print("PASSED ✓")
        passed_tests += 1
        total_tests += 1
    except Exception as e:
        print(f"FAILED ✗")
        print(f"    Error: {e}")
        failed_tests.append(("test_fsq_integration", traceback.format_exc()))
        total_tests += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for test_name, _ in failed_tests:
            print(f"  - {test_name}")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)