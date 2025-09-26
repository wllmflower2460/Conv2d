#!/usr/bin/env python3
"""
Fast tests for NaN interpolation.
Cases: edge-NaNs, mid-NaNs, all-NaNs, dtype preservation.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.data_quality_handler import DataQualityHandler


def test_edge_nans():
    """Test edge NaNs (beginning and end)."""
    handler = DataQualityHandler(default_nan_strategy='interpolate')
    
    # Hand-crafted example: 2 batches, 3 channels, 10 timesteps
    data = np.ones((2, 3, 10), dtype=np.float32) * 5.0
    
    # Add edge NaNs
    data[0, 0, :3] = np.nan    # First 3 timesteps
    data[0, 1, -2:] = np.nan   # Last 2 timesteps
    data[1, 2, 0] = np.nan     # Single edge point
    
    result = handler._interpolate_nan(data.copy())
    
    # Check no NaNs remain
    assert not np.any(np.isnan(result)), "Edge NaNs not handled"
    
    # Check dtype preserved
    assert result.dtype == np.float32, f"Dtype not preserved: {result.dtype}"
    
    # Value checks - should be filled with constant (5.0)
    assert np.allclose(result[0, 0, :3], 5.0), "Beginning edge not filled correctly"
    assert np.allclose(result[0, 1, -2:], 5.0), "End edge not filled correctly"
    
    print("✓ Edge NaNs test passed")


def test_mid_nans():
    """Test middle NaNs (interior interpolation)."""
    handler = DataQualityHandler(default_nan_strategy='interpolate')
    
    # Create linear ramp for easy verification
    data = np.zeros((1, 2, 10), dtype=np.float32)
    data[0, 0, :] = np.arange(10)  # 0,1,2,3,4,5,6,7,8,9
    data[0, 1, :] = np.arange(10) * 2  # 0,2,4,6,8,10,12,14,16,18
    
    # Add interior NaNs
    data[0, 0, 3:6] = np.nan  # Gap from index 3-5
    data[0, 1, 5] = np.nan    # Single point
    
    result = handler._interpolate_nan(data.copy())
    
    # Check interpolated values
    # Channel 0: linear between 2 and 6
    expected_3 = 3.0  # Linear interp
    expected_4 = 4.0
    expected_5 = 5.0
    
    assert np.isclose(result[0, 0, 3], expected_3, atol=0.1), f"Got {result[0, 0, 3]}"
    assert np.isclose(result[0, 0, 4], expected_4, atol=0.1), f"Got {result[0, 0, 4]}"
    assert np.isclose(result[0, 0, 5], expected_5, atol=0.1), f"Got {result[0, 0, 5]}"
    
    # Channel 1: single point should be interpolated
    assert np.isclose(result[0, 1, 5], 10.0, atol=0.1), f"Single NaN: got {result[0, 1, 5]}"
    
    print("✓ Mid NaNs test passed")


def test_all_nans():
    """Test all-NaN rows with fallback."""
    handler = DataQualityHandler(default_nan_strategy='interpolate')
    
    # Create data with known values
    data = np.ones((2, 3, 8), dtype=np.float32)
    data[0, :, :] = 10.0
    data[1, :, :] = 20.0
    
    # Make one row all NaN
    data[0, 1, :] = np.nan
    
    # Test with zero fallback
    result = handler._interpolate_nan(data.copy(), nan_fallback='zero')
    assert np.all(result[0, 1, :] == 0), "All-NaN row not zeroed"
    
    # Test with mean fallback
    data[0, 1, :] = np.nan  # Reset
    result = handler._interpolate_nan(data.copy(), nan_fallback='mean')
    # Mean of channel 1 across batches (excluding the NaN row)
    expected_mean = 20.0  # From batch 1
    assert np.allclose(result[0, 1, :], expected_mean, atol=0.1), \
        f"Mean fallback failed: got {result[0, 1, 0]}"
    
    print("✓ All NaNs test passed")


def test_dtype_preservation():
    """Test that float32 is preserved."""
    handler = DataQualityHandler(default_nan_strategy='interpolate')
    
    for input_dtype in [np.float16, np.float32, np.float64]:
        data = np.ones((1, 2, 5), dtype=input_dtype) * 3.14
        data[0, 0, 2] = np.nan
        
        result = handler._interpolate_nan(data.copy())
        
        # Should preserve float32 or convert to it
        if input_dtype in [np.float16, np.float32]:
            expected = input_dtype
        else:
            expected = np.float32
            
        assert result.dtype == expected, \
            f"Dtype {input_dtype} -> {result.dtype}, expected {expected}"
    
    print("✓ Dtype preservation test passed")


def test_small_handcrafted():
    """Test with small hand-crafted examples."""
    handler = DataQualityHandler(default_nan_strategy='interpolate')
    
    # Example 1: Simple increasing sequence
    data = np.array([[[1., 2., np.nan, 4., 5.]]], dtype=np.float32)
    result = handler._interpolate_nan(data.copy())
    assert np.isclose(result[0, 0, 2], 3.0), "Linear interp failed"
    
    # Example 2: Two NaNs in a row
    data = np.array([[[10., np.nan, np.nan, 40., 50.]]], dtype=np.float32)
    result = handler._interpolate_nan(data.copy())
    # Should interpolate linearly
    assert np.isclose(result[0, 0, 1], 20.0, atol=1.0), "Two-NaN interp failed"
    assert np.isclose(result[0, 0, 2], 30.0, atol=1.0), "Two-NaN interp failed"
    
    print("✓ Hand-crafted examples test passed")


if __name__ == "__main__":
    print("Testing NaN Interpolation (Fast)")
    print("=" * 40)
    
    test_edge_nans()
    test_mid_nans()
    test_all_nans()
    test_dtype_preservation()
    test_small_handcrafted()
    
    print("=" * 40)
    print("✅ All fast NaN interpolation tests passed!")