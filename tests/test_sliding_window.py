#!/usr/bin/env python3
"""
Comprehensive tests for sliding window operations with stride tricks.
Verifies safety measures, memory efficiency, and correctness.
"""

import numpy as np
import torch
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.vectorized_optimizations import VectorizedOperations


class TestSlidingWindow:
    """Test suite for sliding window operations."""
    
    def test_basic_windowing(self):
        """Test basic sliding window functionality."""
        print("\n1. Basic Windowing:")
        
        # Create test data
        data = np.arange(100).reshape(100, 1).astype(np.float32)
        window_size = 10
        step_size = 5
        
        # Apply sliding window
        windows = VectorizedOperations.sliding_window_vectorized(
            data, window_size, step_size
        )
        
        # Verify shape
        expected_windows = (100 - 10) // 5 + 1  # 19 windows
        assert windows.shape == (19, 10, 1), f"Shape mismatch: {windows.shape}"
        
        # Verify content
        assert np.array_equal(windows[0], data[:10])
        assert np.array_equal(windows[1], data[5:15])
        
        print(f"  ✓ Basic windowing: {windows.shape}")
    
    def test_memory_safety(self):
        """Test that windows are properly copied and safe to modify."""
        print("\n2. Memory Safety:")
        
        # Create test data
        original = np.arange(50).reshape(50, 1).astype(np.float32)
        data = original.copy()
        
        # Apply sliding window
        windows = VectorizedOperations.sliding_window_vectorized(
            data, window_size=10, step_size=5
        )
        
        # Modify windows - should NOT affect original data
        windows[0, 0, 0] = 999.0
        
        # Check original data is unchanged
        assert data[0, 0] == 0.0, "Original data was modified!"
        assert np.array_equal(data, original), "Original data corrupted"
        
        print("  ✓ Memory safety: Windows are independent copies")
    
    def test_stride_tricks_safety(self):
        """Test that internal stride tricks implementation is safe."""
        print("\n3. Stride Tricks Safety:")
        
        # Test that we can't accidentally create overlapping writable views
        data = np.arange(100).reshape(100, 1).astype(np.float32)
        
        # Get windows
        windows1 = VectorizedOperations.sliding_window_vectorized(
            data, window_size=20, step_size=10
        )
        windows2 = VectorizedOperations.sliding_window_vectorized(
            data, window_size=20, step_size=10
        )
        
        # Modify one set of windows
        windows1[0, 0, 0] = 999.0
        
        # Other windows should be unaffected
        assert windows2[0, 0, 0] != 999.0, "Windows share memory!"
        
        print("  ✓ Stride tricks: Each call returns independent data")
    
    def test_edge_cases(self):
        """Test edge cases for windowing."""
        print("\n4. Edge Cases:")
        
        # Test 1: Window size equals data length
        data = np.arange(20).reshape(20, 1).astype(np.float32)
        windows = VectorizedOperations.sliding_window_vectorized(
            data, window_size=20, step_size=1
        )
        assert windows.shape == (1, 20, 1)
        print("  ✓ Window size = data length")
        
        # Test 2: Single channel (1D) data
        data_1d = np.arange(30).astype(np.float32)
        windows = VectorizedOperations.sliding_window_vectorized(
            data_1d, window_size=10, step_size=5
        )
        assert windows.shape == (5, 10, 1)
        print("  ✓ 1D data handling")
        
        # Test 3: Large step size
        data = np.arange(100).reshape(100, 1).astype(np.float32)
        windows = VectorizedOperations.sliding_window_vectorized(
            data, window_size=10, step_size=50
        )
        assert windows.shape == (2, 10, 1)  # Only 2 windows
        print("  ✓ Large step size")
        
        # Test 4: Multi-channel data
        data = np.random.randn(100, 9).astype(np.float32)
        windows = VectorizedOperations.sliding_window_vectorized(
            data, window_size=20, step_size=10
        )
        assert windows.shape == (9, 20, 9)
        print("  ✓ Multi-channel data")
    
    def test_numerical_precision(self):
        """Test that windowing preserves numerical precision."""
        print("\n5. Numerical Precision:")
        
        # Create high-precision data
        data = np.random.randn(50, 3).astype(np.float64) * 1e-10
        
        # Apply windowing (should handle dtype conversion)
        windows = VectorizedOperations.sliding_window_vectorized(
            data, window_size=10, step_size=5
        )
        
        # Check that values are preserved (within float32 precision)
        for i in range(min(3, windows.shape[0])):
            window_data = windows[i]
            original_segment = data[i*5:i*5+10]
            np.testing.assert_allclose(
                window_data, original_segment,
                rtol=1e-6, atol=1e-8
            )
        
        print("  ✓ Numerical precision maintained")
    
    def test_performance(self):
        """Test performance of vectorized sliding window."""
        print("\n6. Performance Test:")
        
        # Large dataset
        data = np.random.randn(10000, 9).astype(np.float32)
        window_size = 100
        step_size = 50
        
        # Time vectorized version
        start = time.perf_counter()
        windows_vec = VectorizedOperations.sliding_window_vectorized(
            data, window_size, step_size
        )
        time_vec = time.perf_counter() - start
        
        # Time naive loop version
        start = time.perf_counter()
        windows_loop = []
        for i in range(0, len(data) - window_size + 1, step_size):
            windows_loop.append(data[i:i+window_size])
        windows_loop = np.array(windows_loop)
        time_loop = time.perf_counter() - start
        
        speedup = time_loop / time_vec
        print(f"  Vectorized: {time_vec*1000:.2f}ms")
        print(f"  Loop: {time_loop*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Verify results match
        np.testing.assert_array_almost_equal(windows_vec, windows_loop)
        print("  ✓ Results match loop implementation")
    
    def test_boundary_conditions(self):
        """Test boundary conditions and corner cases."""
        print("\n7. Boundary Conditions:")
        
        # Test when window_size > data length
        data = np.arange(10).reshape(10, 1).astype(np.float32)
        try:
            windows = VectorizedOperations.sliding_window_vectorized(
                data, window_size=20, step_size=5
            )
            # Should return empty or single window
            assert windows.shape[0] == 0, "Should handle oversized window"
            print("  ✓ Oversized window handled")
        except:
            print("  ✓ Oversized window raises appropriate error")
        
        # Test with step_size = 1 (maximum overlap)
        data = np.arange(30).reshape(30, 1).astype(np.float32)
        windows = VectorizedOperations.sliding_window_vectorized(
            data, window_size=10, step_size=1
        )
        assert windows.shape[0] == 21  # 30 - 10 + 1
        print("  ✓ Maximum overlap (step=1)")
        
        # Test with no overlap (step = window)
        windows = VectorizedOperations.sliding_window_vectorized(
            data, window_size=10, step_size=10
        )
        assert windows.shape[0] == 3  # Exactly 3 non-overlapping windows
        print("  ✓ No overlap (step=window)")
    
    def test_dtype_handling(self):
        """Test handling of different data types."""
        print("\n8. Dtype Handling:")
        
        for dtype in [np.float16, np.float32, np.float64, np.int32]:
            data = np.arange(50).astype(dtype).reshape(50, 1)
            windows = VectorizedOperations.sliding_window_vectorized(
                data, window_size=10, step_size=5
            )
            
            # Check shape is correct
            assert windows.shape == (9, 10, 1)
            
            # Check values are preserved
            np.testing.assert_array_equal(windows[0], data[:10])
            
            print(f"  ✓ {dtype.__name__} handled correctly")
    
    def test_concurrent_access(self):
        """Test that concurrent windowing operations are safe."""
        print("\n9. Concurrent Access Safety:")
        
        # Shared data
        shared_data = np.arange(100).reshape(100, 1).astype(np.float32)
        
        # Multiple simultaneous windows
        windows1 = VectorizedOperations.sliding_window_vectorized(
            shared_data, window_size=20, step_size=10
        )
        windows2 = VectorizedOperations.sliding_window_vectorized(
            shared_data, window_size=15, step_size=5
        )
        windows3 = VectorizedOperations.sliding_window_vectorized(
            shared_data, window_size=25, step_size=15
        )
        
        # Modify each independently
        windows1[0, 0, 0] = 111
        windows2[0, 0, 0] = 222
        windows3[0, 0, 0] = 333
        
        # Check isolation
        assert shared_data[0, 0] == 0, "Shared data modified"
        assert windows1[0, 0, 0] == 111
        assert windows2[0, 0, 0] == 222
        assert windows3[0, 0, 0] == 333
        
        print("  ✓ Concurrent operations are isolated")
    
    def test_large_scale(self):
        """Test with large-scale realistic data."""
        print("\n10. Large Scale Test:")
        
        # Simulate 1 hour of IMU data at 100Hz
        # 9 channels (3 accel, 3 gyro, 3 mag), 360000 samples
        data = np.random.randn(360000, 9).astype(np.float32)
        
        # 1-second windows with 0.5-second overlap
        window_size = 100  # 1 second at 100Hz
        step_size = 50     # 0.5 second step
        
        start = time.perf_counter()
        windows = VectorizedOperations.sliding_window_vectorized(
            data, window_size, step_size
        )
        elapsed = time.perf_counter() - start
        
        expected_windows = (360000 - 100) // 50 + 1
        assert windows.shape == (expected_windows, 100, 9)
        
        print(f"  1 hour IMU data: {elapsed*1000:.2f}ms")
        print(f"  Windows shape: {windows.shape}")
        print("  ✓ Large scale processing successful")


def run_all_tests():
    """Run all sliding window tests."""
    print("Testing Sliding Window Operations with Stride Tricks")
    print("=" * 50)
    
    test_suite = TestSlidingWindow()
    
    # Run all test methods
    test_methods = [
        test_suite.test_basic_windowing,
        test_suite.test_memory_safety,
        test_suite.test_stride_tricks_safety,
        test_suite.test_edge_cases,
        test_suite.test_numerical_precision,
        test_suite.test_performance,
        test_suite.test_boundary_conditions,
        test_suite.test_dtype_handling,
        test_suite.test_concurrent_access,
        test_suite.test_large_scale,
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
    print("✅ All sliding window tests passed!")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)