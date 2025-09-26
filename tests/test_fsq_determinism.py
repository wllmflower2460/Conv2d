#!/usr/bin/env python3
"""FSQ determinism tests - same input + levels + seed ⇒ identical codes."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conv2d.features.fsq_contract import encode_fsq


class TestFSQDeterminism:
    """Test FSQ deterministic behavior - critical for production reliability."""
    
    def test_identical_inputs_produce_identical_codes(self):
        """Same input tensor must produce identical codes every time."""
        
        # Create deterministic input
        torch.manual_seed(42)
        x = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        
        # Multiple runs with same input
        result1 = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        result2 = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        result3 = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        
        # Codes must be bit-for-bit identical
        assert torch.equal(result1.codes, result2.codes), \
            "FSQ codes not deterministic between runs"
        assert torch.equal(result2.codes, result3.codes), \
            "FSQ codes not deterministic across multiple runs"
        
        # Features must be numerically identical
        assert torch.allclose(result1.features, result2.features, atol=1e-8), \
            "FSQ features not deterministic"
        assert torch.allclose(result2.features, result3.features, atol=1e-8), \
            "FSQ features not deterministic across runs"
    
    def test_different_seeds_produce_different_codes(self):
        """Different random seeds should produce different results (sanity check)."""
        
        # Same input, different random states
        torch.manual_seed(42)
        x1 = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        result1 = encode_fsq(x1, levels=[8, 6, 5], reset_stats=True)
        
        torch.manual_seed(123)  # Different seed
        x2 = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        result2 = encode_fsq(x2, levels=[8, 6, 5], reset_stats=True)
        
        # Results should be different (with high probability)
        assert not torch.equal(result1.codes, result2.codes), \
            "Different inputs produced identical codes (very unlikely)"
    
    def test_levels_consistency(self):
        """Different quantization levels produce consistent behavior."""
        
        torch.manual_seed(42)
        x = torch.randn(8, 9, 2, 100, dtype=torch.float32)
        
        # Test different level configurations
        level_configs = [
            [8, 6, 5],      # 240 codes
            [4, 4, 4, 4],   # 256 codes  
            [8, 3, 5],      # 120 codes
        ]
        
        for levels in level_configs:
            result1 = encode_fsq(x, levels=levels, reset_stats=True)
            result2 = encode_fsq(x, levels=levels, reset_stats=True)
            
            # Must be deterministic for each level config
            assert torch.equal(result1.codes, result2.codes), \
                f"FSQ not deterministic for levels {levels}"
            
            # Check code range is valid
            expected_max = np.prod(levels) - 1
            actual_max = result1.codes.max().item()
            assert actual_max <= expected_max, \
                f"Code {actual_max} exceeds max {expected_max} for levels {levels}"
    
    def test_non_zero_usage_for_most_bins(self):
        """Most quantization bins should have non-zero usage."""
        
        # Use diverse input to encourage code usage
        torch.manual_seed(42)
        x = torch.randn(100, 9, 2, 100, dtype=torch.float32) * 2.0  # Larger variance
        
        levels = [8, 6, 5]  # 240 total codes
        result = encode_fsq(x, levels=levels, reset_stats=True)
        
        # Count unique codes used
        unique_codes = torch.unique(result.codes)
        total_possible_codes = np.prod(levels)
        usage_rate = len(unique_codes) / total_possible_codes
        
        # Should use at least 30% of available codes with diverse input
        assert usage_rate >= 0.30, \
            f"Low code usage: {usage_rate:.1%} ({len(unique_codes)}/{total_possible_codes})"
        
        # Perplexity should be reasonable
        assert result.perplexity > 10, \
            f"Very low perplexity: {result.perplexity:.1f} indicates poor utilization"
    
    def test_batch_size_independence(self):
        """Results should be independent of batch size."""
        
        torch.manual_seed(42)
        x_large = torch.randn(32, 9, 2, 100, dtype=torch.float32)
        
        # Split into smaller batches
        x_batch1 = x_large[:16]
        x_batch2 = x_large[16:]
        
        # Process as single large batch
        result_large = encode_fsq(x_large, levels=[8, 6, 5], reset_stats=True)
        
        # Process as two smaller batches
        result_small1 = encode_fsq(x_batch1, levels=[8, 6, 5], reset_stats=True)
        result_small2 = encode_fsq(x_batch2, levels=[8, 6, 5], reset_stats=True)
        
        # Concatenate small batch results
        codes_concat = torch.cat([result_small1.codes, result_small2.codes], dim=0)
        features_concat = torch.cat([result_small1.features, result_small2.features], dim=0)
        
        # Should match large batch results
        assert torch.equal(result_large.codes, codes_concat), \
            "FSQ codes depend on batch size (not deterministic)"
        assert torch.allclose(result_large.features, features_concat, atol=1e-6), \
            "FSQ features depend on batch size"
    
    def test_floating_point_precision_stability(self):
        """Small floating point differences shouldn't affect codes."""
        
        torch.manual_seed(42)
        x_base = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        
        # Add tiny numerical noise
        epsilon = 1e-7  # Very small noise
        x_noise = x_base + torch.randn_like(x_base) * epsilon
        
        result_base = encode_fsq(x_base, levels=[8, 6, 5], reset_stats=True)
        result_noise = encode_fsq(x_noise, levels=[8, 6, 5], reset_stats=True)
        
        # Codes should be mostly the same despite tiny noise
        agreement = (result_base.codes == result_noise.codes).float().mean()
        assert agreement > 0.95, \
            f"FSQ too sensitive to tiny numerical differences: {agreement:.1%} agreement"
    
    def test_edge_case_inputs(self):
        """FSQ should handle edge cases deterministically."""
        
        # Test zero input
        x_zero = torch.zeros(4, 9, 2, 100, dtype=torch.float32)
        result_zero1 = encode_fsq(x_zero, levels=[8, 6, 5], reset_stats=True)
        result_zero2 = encode_fsq(x_zero, levels=[8, 6, 5], reset_stats=True)
        assert torch.equal(result_zero1.codes, result_zero2.codes), \
            "Zero input not deterministic"
        
        # Test constant input
        x_const = torch.ones(4, 9, 2, 100, dtype=torch.float32) * 1.5
        result_const1 = encode_fsq(x_const, levels=[8, 6, 5], reset_stats=True)
        result_const2 = encode_fsq(x_const, levels=[8, 6, 5], reset_stats=True)
        assert torch.equal(result_const1.codes, result_const2.codes), \
            "Constant input not deterministic"
        
        # Test very large values
        x_large = torch.ones(4, 9, 2, 100, dtype=torch.float32) * 100.0
        result_large1 = encode_fsq(x_large, levels=[8, 6, 5], reset_stats=True)
        result_large2 = encode_fsq(x_large, levels=[8, 6, 5], reset_stats=True)
        assert torch.equal(result_large1.codes, result_large2.codes), \
            "Large value input not deterministic"
    
    def test_reset_stats_behavior(self):
        """reset_stats=True should ensure fresh state."""
        
        torch.manual_seed(42)
        x = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        
        # First run without reset
        result1 = encode_fsq(x, levels=[8, 6, 5], reset_stats=False)
        
        # Second run without reset (might have accumulated state)
        result2 = encode_fsq(x, levels=[8, 6, 5], reset_stats=False)
        
        # Third run with reset (should be fresh)
        result3 = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        
        # Fourth run with reset (should match third)
        result4 = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        
        # Reset runs should be identical
        assert torch.equal(result3.codes, result4.codes), \
            "reset_stats=True not providing identical results"
    
    def test_memory_consistency(self):
        """Multiple runs shouldn't accumulate memory artifacts."""
        
        torch.manual_seed(42)
        x = torch.randn(8, 9, 2, 100, dtype=torch.float32)
        
        # Run many times to check for memory consistency
        results = []
        for i in range(10):
            result = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
            results.append(result.codes.clone())
        
        # All results should be identical
        reference = results[0]
        for i, result in enumerate(results[1:], 1):
            assert torch.equal(reference, result), \
                f"Run {i} differs from reference - memory inconsistency detected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])