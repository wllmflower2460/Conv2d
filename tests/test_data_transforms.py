#!/usr/bin/env python3
"""Unit tests for data transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conv2d.data.transforms import (
    BandpassFilter,
    Clip,
    Compose,
    InterpolateNaN,
    QuantizeFSQ,
    Standardize,
    ToTensor,
    Window,
)
from conv2d.data.utils import ensure_fp32


class TestStandardize:
    """Test Standardize transform."""
    
    def test_fit_transform(self):
        """Test fit and transform."""
        X = np.random.randn(10, 5, 100).astype(np.float32)
        
        transform = Standardize()
        X_transformed = transform.fit_transform(X)
        
        # Check mean and std
        assert np.allclose(X_transformed.mean(), 0, atol=1e-6)
        assert np.allclose(X_transformed.std(), 1, atol=1e-6)
        
        # Check dtype preserved
        assert X_transformed.dtype == np.float32
        
    def test_inverse_transform(self):
        """Test inverse transform."""
        X = np.random.randn(10, 5, 100).astype(np.float32)
        
        transform = Standardize()
        X_transformed = transform.fit_transform(X)
        X_recovered = transform.inverse_transform(X_transformed)
        
        # Check recovery
        np.testing.assert_allclose(X, X_recovered, rtol=1e-5)
        
    def test_no_mutation(self):
        """Test that input is not mutated."""
        X = np.random.randn(10, 5, 100).astype(np.float32)
        X_copy = X.copy()
        
        transform = Standardize()
        _ = transform.fit_transform(X)
        
        # Original should be unchanged
        np.testing.assert_array_equal(X, X_copy)
        
    def test_torch_tensor(self):
        """Test with PyTorch tensors."""
        X = torch.randn(10, 5, 100)
        
        transform = Standardize()
        X_transformed = transform.fit_transform(X)
        
        assert isinstance(X_transformed, torch.Tensor)
        assert torch.allclose(X_transformed.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(X_transformed.std(), torch.tensor(1.0), atol=1e-6)


class TestWindow:
    """Test Window transform."""
    
    def test_basic_windowing(self):
        """Test basic sliding window."""
        X = np.arange(100).reshape(1, 1, 100).astype(np.float32)
        
        transform = Window(window_size=10, step_size=5)
        X_windowed = transform.transform(X)
        
        # Check shape: (1, 1, n_windows, window_size)
        expected_windows = (100 - 10) // 5 + 1
        assert X_windowed.shape == (1, 1, expected_windows, 10)
        
        # Check first window
        np.testing.assert_array_equal(X_windowed[0, 0, 0], np.arange(10))
        
        # Check second window (step=5)
        np.testing.assert_array_equal(X_windowed[0, 0, 1], np.arange(5, 15))
        
    def test_no_mutation(self):
        """Test that windowing doesn't mutate input."""
        X = np.random.randn(5, 3, 100).astype(np.float32)
        X_copy = X.copy()
        
        transform = Window(window_size=20, step_size=10)
        _ = transform.transform(X)
        
        np.testing.assert_array_equal(X, X_copy)
        
    def test_torch_unfold(self):
        """Test windowing with torch tensors (safe unfold)."""
        X = torch.randn(5, 3, 100)
        
        transform = Window(window_size=20, step_size=10)
        X_windowed = transform.transform(X)
        
        assert isinstance(X_windowed, torch.Tensor)
        assert X_windowed.is_contiguous()  # Should be contiguous (safe)
        
        # Modify windowed tensor shouldn't affect original
        X_copy = X.clone()
        X_windowed[0, 0, 0, 0] = 999
        assert torch.equal(X, X_copy)


class TestInterpolateNaN:
    """Test NaN interpolation."""
    
    def test_linear_interpolation(self):
        """Test linear interpolation of NaN values."""
        X = np.array([[[1, 2, np.nan, 4, 5]]]).astype(np.float32)
        
        transform = InterpolateNaN(method="linear")
        X_clean = transform.transform(X)
        
        # Should interpolate to 3
        assert np.isclose(X_clean[0, 0, 2], 3.0)
        assert not np.any(np.isnan(X_clean))
        
    def test_zero_fill(self):
        """Test zero filling of NaN values."""
        X = np.array([[[1, 2, np.nan, 4, 5]]]).astype(np.float32)
        
        transform = InterpolateNaN(method="zero")
        X_clean = transform.transform(X)
        
        assert X_clean[0, 0, 2] == 0.0
        assert not np.any(np.isnan(X_clean))
        
    def test_no_mutation(self):
        """Test that NaN interpolation doesn't mutate input."""
        X = np.array([[[1, 2, np.nan, 4, 5]]]).astype(np.float32)
        X_copy = X.copy()
        
        transform = InterpolateNaN(method="linear")
        _ = transform.transform(X)
        
        np.testing.assert_array_equal(X, X_copy)
        assert np.isnan(X[0, 0, 2])  # Original still has NaN


class TestClip:
    """Test Clip transform."""
    
    def test_clipping(self):
        """Test value clipping."""
        X = np.array([[-2, -1, 0, 1, 2]]).astype(np.float32)
        
        transform = Clip(min_val=-1, max_val=1)
        X_clipped = transform.transform(X)
        
        np.testing.assert_array_equal(X_clipped, [[-1, -1, 0, 1, 1]])
        
    def test_no_mutation(self):
        """Test that clipping doesn't mutate input."""
        X = np.array([[-2, -1, 0, 1, 2]]).astype(np.float32)
        X_copy = X.copy()
        
        transform = Clip(min_val=-1, max_val=1)
        _ = transform.transform(X)
        
        np.testing.assert_array_equal(X, X_copy)


class TestQuantizeFSQ:
    """Test FSQ quantization."""
    
    def test_uniform_quantization(self):
        """Test uniform quantization."""
        X = np.random.randn(10, 3).astype(np.float32)
        
        transform = QuantizeFSQ(levels=[4, 4, 4], method="uniform")
        X_quantized = transform.fit_transform(X)
        
        # Check that we have at most 4 unique values per dimension
        for dim in range(3):
            unique_vals = np.unique(X_quantized[:, dim])
            assert len(unique_vals) <= 4
            
        # Check dtype
        assert X_quantized.dtype == np.float32
        
    def test_no_mutation(self):
        """Test that quantization doesn't mutate input."""
        X = np.random.randn(10, 3).astype(np.float32)
        X_copy = X.copy()
        
        transform = QuantizeFSQ(levels=[4, 4, 4])
        _ = transform.fit_transform(X)
        
        np.testing.assert_array_equal(X, X_copy)


class TestCompose:
    """Test transform composition."""
    
    def test_pipeline(self):
        """Test composing multiple transforms."""
        X = np.random.randn(10, 5, 100).astype(np.float64)  # Wrong dtype
        
        # Add some NaNs
        X[0, 0, 10:15] = np.nan
        
        pipeline = Compose([
            InterpolateNaN(method="linear"),
            Standardize(),
            Clip(min_val=-3, max_val=3),
        ])
        
        X_transformed = pipeline.fit_transform(X)
        
        # Check no NaNs
        assert not np.any(np.isnan(X_transformed))
        
        # Check standardized
        assert np.allclose(X_transformed.mean(), 0, atol=0.1)
        assert np.allclose(X_transformed.std(), 1, atol=0.1)
        
        # Check clipped
        assert X_transformed.min() >= -3
        assert X_transformed.max() <= 3
        
        # Check dtype (should be float32)
        assert X_transformed.dtype == np.float32


class TestToTensor:
    """Test tensor conversion."""
    
    def test_numpy_to_tensor(self):
        """Test converting numpy to tensor."""
        X = np.random.randn(10, 5, 100).astype(np.float32)
        
        transform = ToTensor()
        X_tensor = transform.transform(X)
        
        assert isinstance(X_tensor, torch.Tensor)
        assert X_tensor.dtype == torch.float32
        np.testing.assert_array_almost_equal(X, X_tensor.numpy())
        
    def test_device_placement(self):
        """Test tensor device placement."""
        X = np.random.randn(10, 5, 100).astype(np.float32)
        
        if torch.cuda.is_available():
            transform = ToTensor(device="cuda")
            X_tensor = transform.transform(X)
            assert X_tensor.device.type == "cuda"
        else:
            transform = ToTensor(device="cpu")
            X_tensor = transform.transform(X)
            assert X_tensor.device.type == "cpu"


class TestDtypeEnforcement:
    """Test dtype enforcement at boundaries."""
    
    def test_ensure_fp32(self):
        """Test ensure_fp32 utility."""
        # Test numpy
        X_f64 = np.random.randn(10, 5).astype(np.float64)
        X_f32 = ensure_fp32(X_f64)
        assert X_f32.dtype == np.float32
        
        # Test tensor
        X_tensor = torch.randn(10, 5, dtype=torch.float64)
        X_tensor_f32 = ensure_fp32(X_tensor)
        assert X_tensor_f32.dtype == torch.float32
        
        # Test no-op when already float32
        X_already = np.random.randn(10, 5).astype(np.float32)
        X_same = ensure_fp32(X_already)
        assert X_same is X_already  # Should be same object
        
        # Test copy flag
        X_copy = ensure_fp32(X_already, copy=True)
        assert X_copy is not X_already  # Should be different object
        np.testing.assert_array_equal(X_copy, X_already)


class TestBandpassFilter:
    """Test bandpass filter."""
    
    def test_filter_application(self):
        """Test that bandpass filter works."""
        # Create signal with known frequencies
        fs = 100.0  # Sampling rate
        t = np.arange(0, 2, 1/fs)
        
        # 5 Hz + 25 Hz signal
        X = (np.sin(2 * np.pi * 5 * t) + 
             np.sin(2 * np.pi * 25 * t)).reshape(1, 1, -1).astype(np.float32)
        
        # Filter to keep only 5 Hz (bandpass 2-10 Hz)
        transform = BandpassFilter(2.0, 10.0, fs)
        X_filtered = transform.transform(X)
        
        # Check that high frequency is attenuated
        fft_orig = np.abs(np.fft.rfft(X[0, 0]))
        fft_filt = np.abs(np.fft.rfft(X_filtered[0, 0]))
        freqs = np.fft.rfftfreq(len(t), 1/fs)
        
        # 25 Hz should be attenuated
        idx_25hz = np.argmin(np.abs(freqs - 25))
        assert fft_filt[idx_25hz] < fft_orig[idx_25hz] * 0.1
        
    def test_no_mutation(self):
        """Test that filtering doesn't mutate input."""
        X = np.random.randn(5, 3, 100).astype(np.float32)
        X_copy = X.copy()
        
        transform = BandpassFilter(1.0, 20.0, 100.0)
        _ = transform.transform(X)
        
        np.testing.assert_array_equal(X, X_copy)


def test_functional_transforms_are_stateless():
    """Test that functional transforms don't require fitting."""
    transforms = [
        Window(10, 5),
        InterpolateNaN(),
        Clip(-1, 1),
        ToTensor(),
    ]
    
    X = np.random.randn(5, 3, 100).astype(np.float32)
    
    for transform in transforms:
        # Should work without fitting
        _ = transform.transform(X)
        assert transform.is_fitted


def test_stateful_transforms_require_fitting():
    """Test that stateful transforms require fitting."""
    transforms = [
        Standardize(),
        QuantizeFSQ([4, 4, 4]),
    ]
    
    X = np.random.randn(10, 3).astype(np.float32)
    
    for transform in transforms:
        # Should fail without fitting
        with pytest.raises(RuntimeError, match="must be fitted"):
            _ = transform.transform(X)
            
        # Should work after fitting
        transform.fit(X)
        _ = transform.transform(X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])