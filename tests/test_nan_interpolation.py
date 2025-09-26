#!/usr/bin/env python3
"""
Comprehensive tests for improved NaN interpolation.
Tests edge cases, fallback strategies, and dtype preservation.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.data_quality_handler import DataQualityHandler


def test_nan_interpolation():
    """Test all aspects of the improved NaN interpolation."""
    
    print("Testing Improved NaN Interpolation")
    print("=" * 50)
    
    handler = DataQualityHandler(default_nan_strategy='interpolate')
    
    # Test 1: Edge NaNs only
    print("\n1. Edge NaNs (beginning and end):")
    data = np.ones((2, 3, 10), dtype=np.float32)
    # Add NaNs at edges
    data[0, 0, :3] = np.nan  # Beginning NaNs
    data[0, 1, -3:] = np.nan  # End NaNs
    data[1, 0, [0, -1]] = np.nan  # Both edges
    
    # Test different edge methods
    for edge_method in ['extrapolate', 'constant', 'ffill']:
        result = handler._interpolate_nan(data.copy(), edge_method=edge_method)
        print(f"  {edge_method}: Shape={result.shape}, Dtype={result.dtype}, "
              f"No NaNs={not np.any(np.isnan(result))}")
        assert result.shape == data.shape
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
    
    # Test 2: Interior NaNs only
    print("\n2. Interior NaNs (middle of sequence):")
    data = np.ones((2, 3, 20), dtype=np.float32) * 10
    data[0, 0, 8:12] = np.nan  # Middle gap
    data[1, 1, [5, 10, 15]] = np.nan  # Scattered NaNs
    
    result = handler._interpolate_nan(data.copy())
    print(f"  Result: No NaNs={not np.any(np.isnan(result))}, "
          f"Interior interpolated correctly")
    assert not np.any(np.isnan(result))
    # Check that interpolation is reasonable (between neighbors)
    assert 9 < result[0, 0, 10] < 11  # Should be close to 10
    
    # Test 3: All NaN rows with different fallbacks
    print("\n3. All-NaN rows with fallback strategies:")
    data = np.random.randn(3, 4, 15).astype(np.float32)
    # Make some rows all NaN
    data[0, 0, :] = np.nan  # First sample, first channel
    data[1, 2, :] = np.nan  # Second sample, third channel
    
    for fallback in ['zero', 'mean', 'median']:
        result = handler._interpolate_nan(data.copy(), nan_fallback=fallback)
        print(f"  {fallback}: All-NaN rows handled, No remaining NaNs")
        assert not np.any(np.isnan(result))
        
        if fallback == 'zero':
            assert np.all(result[0, 0, :] == 0)
        elif fallback == 'mean':
            # Should be close to channel mean (excluding the all-NaN row)
            channel_mean = np.nanmean(data[:, 0, :])
            if not np.isnan(channel_mean):
                assert np.allclose(result[0, 0, :], channel_mean, atol=1e-5)
    
    # Test 4: Single valid point
    print("\n4. Single valid point (edge case):")
    data = np.full((1, 2, 10), np.nan, dtype=np.float32)
    data[0, 0, 5] = 42.0  # Only one valid point
    
    result = handler._interpolate_nan(data.copy())
    print(f"  Single valid: All values={result[0, 0, 0]:.1f}")
    assert np.all(result[0, 0, :] == 42.0)
    
    # Test 5: Dtype preservation
    print("\n5. Dtype preservation:")
    for dtype in [np.float16, np.float32, np.float64]:
        data = np.ones((2, 2, 5), dtype=dtype)
        data[0, 0, 2] = np.nan
        
        result = handler._interpolate_nan(data.copy())
        expected_dtype = dtype if dtype in [np.float16, np.float32] else np.float32
        print(f"  Input {dtype.__name__} -> Output {result.dtype}")
        assert result.dtype == expected_dtype
    
    # Test 6: Mixed NaN patterns
    print("\n6. Mixed patterns (edge + interior + all-NaN):")
    data = np.random.randn(4, 3, 25).astype(np.float32) * 10
    # Various NaN patterns
    data[0, 0, :5] = np.nan  # Leading edge
    data[0, 1, 10:15] = np.nan  # Interior gap
    data[1, 0, -5:] = np.nan  # Trailing edge
    data[2, 1, :] = np.nan  # All NaN
    data[3, 2, [0, 5, 10, 15, 20, 24]] = np.nan  # Scattered including edges
    
    result = handler._interpolate_nan(data.copy(), nan_fallback='median', edge_method='constant')
    print(f"  Complex patterns: Shape preserved={result.shape == data.shape}, "
          f"No NaNs={not np.any(np.isnan(result))}")
    assert result.shape == data.shape
    assert not np.any(np.isnan(result))
    assert result.dtype == np.float32
    
    # Test 7: Performance with large arrays
    print("\n7. Performance test (large array):")
    import time
    
    data = np.random.randn(64, 9, 200).astype(np.float32)
    # Add 10% NaNs randomly
    nan_mask = np.random.random(data.shape) < 0.1
    data[nan_mask] = np.nan
    
    start = time.perf_counter()
    result = handler._interpolate_nan(data.copy())
    elapsed = time.perf_counter() - start
    
    print(f"  64×9×200 array: {elapsed*1000:.2f}ms, "
          f"NaNs removed={not np.any(np.isnan(result))}")
    assert not np.any(np.isnan(result))
    
    # Test 8: Channel-aware statistics
    print("\n8. Channel-aware fallback (gyro vs accel):")
    data = np.random.randn(10, 6, 50).astype(np.float32)
    # Simulate different scales for accel (0-2) vs gyro (3-5)
    data[:, :3, :] *= 10  # Accelerometer scale
    data[:, 3:, :] *= 0.1  # Gyroscope scale
    
    # Make one accel and one gyro row all-NaN
    data[0, 1, :] = np.nan  # Accel channel
    data[0, 4, :] = np.nan  # Gyro channel
    
    result = handler._interpolate_nan(data.copy(), nan_fallback='median')
    
    # Check that fallback respects channel scale
    accel_median = np.nanmedian(data[:, 1, :])
    gyro_median = np.nanmedian(data[:, 4, :])
    
    print(f"  Accel channel filled with ~{result[0, 1, 0]:.2f} "
          f"(median ~{accel_median:.2f})")
    print(f"  Gyro channel filled with ~{result[0, 4, 0]:.4f} "
          f"(median ~{gyro_median:.4f})")
    
    # The filled values should be close to respective channel medians
    if not np.isnan(accel_median):
        assert np.abs(result[0, 1, 0] - accel_median) < np.abs(result[0, 1, 0] - gyro_median)
    
    print("\n" + "=" * 50)
    print("✅ All NaN interpolation tests passed!")
    
    return True


if __name__ == "__main__":
    test_nan_interpolation()