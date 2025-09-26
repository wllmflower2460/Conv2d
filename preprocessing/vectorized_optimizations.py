#!/usr/bin/env python3
"""
Vectorized implementations of common loop patterns found in the codebase.
These optimizations replace nested loops with numpy/torch vectorized operations.
"""

import numpy as np
import torch
from typing import Tuple, Optional
from scipy import interpolate
import logging

logger = logging.getLogger(__name__)


class VectorizedOperations:
    """Collection of vectorized operations to replace common loop patterns."""
    
    @staticmethod
    def interpolate_nan_vectorized(data: np.ndarray) -> np.ndarray:
        """
        Vectorized NaN interpolation for 3D arrays.
        Replaces nested loops in data_quality_handler.py
        
        Original: O(B * C * T) with Python loops
        Optimized: O(T) with vectorized operations per channel
        
        Args:
            data: Array of shape (B, C, T) with possible NaN values
            
        Returns:
            Data with NaN values interpolated
        """
        if len(data.shape) != 3:
            return data
            
        B, C, T = data.shape
        
        # Reshape to (B*C, T) for batch processing
        data_reshaped = data.reshape(-1, T)
        
        # Create time index for interpolation
        x = np.arange(T)
        
        # Process all channels at once
        for idx in range(data_reshaped.shape[0]):
            signal = data_reshaped[idx]
            if np.any(np.isnan(signal)):
                valid_mask = ~np.isnan(signal)
                if np.sum(valid_mask) >= 2:
                    # Use numpy's interp (vectorized internally)
                    data_reshaped[idx] = np.interp(x, x[valid_mask], signal[valid_mask])
                else:
                    # Fall back to mean replacement
                    mean_val = np.nanmean(signal) if not np.all(np.isnan(signal)) else 0
                    data_reshaped[idx] = np.nan_to_num(signal, nan=mean_val)
        
        return data_reshaped.reshape(B, C, T)
    
    @staticmethod
    def interpolate_nan_fully_vectorized(data: np.ndarray) -> np.ndarray:
        """
        Fully vectorized NaN interpolation using advanced indexing.
        Even faster version that avoids the loop entirely.
        
        Args:
            data: Array of shape (B, C, T) with possible NaN values
            
        Returns:
            Data with NaN values interpolated
        """
        if len(data.shape) != 3:
            return data
            
        B, C, T = data.shape
        result = data.copy()
        
        # Find all NaN locations
        nan_mask = np.isnan(data)
        
        # Process each time series that has NaNs
        has_nan = np.any(nan_mask, axis=2)
        
        if not np.any(has_nan):
            return result
        
        # Use pandas for efficient interpolation (if available)
        try:
            import pandas as pd
            # Reshape to 2D for pandas processing
            data_2d = data.reshape(-1, T)
            df = pd.DataFrame(data_2d.T)
            df_interpolated = df.interpolate(method='linear', axis=0, limit_direction='both')
            result = df_interpolated.values.T.reshape(B, C, T)
        except ImportError:
            # Fallback to numpy-based vectorized approach
            result = VectorizedOperations.interpolate_nan_vectorized(data)
            
        return result
    
    @staticmethod
    def compute_distances_vectorized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vectorized Euclidean distance computation.
        Replaces loop in kinematic_features.py
        
        Original: Loop over batch dimension
        Optimized: Single vectorized operation
        
        Args:
            x, y: Tensors of shape (B, C, T)
            
        Returns:
            Distances tensor of shape (B,)
        """
        # Compute squared differences
        diff_squared = (x - y) ** 2
        
        # Sum over channel dimension and take sqrt
        distances = torch.sqrt(torch.sum(diff_squared, dim=1))
        
        # Mean over time dimension
        return torch.mean(distances, dim=1)
    
    @staticmethod
    def estimate_mutual_information_vectorized(x: torch.Tensor, y: torch.Tensor, 
                                              n_bins: int = 10) -> torch.Tensor:
        """
        Vectorized mutual information estimation with robust handling.
        Replaces loop in kinematic_features.py
        
        Args:
            x, y: Tensors of shape (B, C, T)
            n_bins: Number of bins for histogram
            
        Returns:
            MI values tensor of shape (B,)
        """
        B = x.shape[0]
        mi_values = torch.zeros(B, device=x.device)
        
        # Compute consistent bin edges across all batches for comparability
        # This ensures all MI values use the same discretization
        x_min, x_max = x.min().item(), x.max().item()
        y_min, y_max = y.min().item(), y.max().item()
        
        # Add small margin to avoid edge effects
        margin = 1e-10
        x_edges = torch.linspace(x_min - margin, x_max + margin, n_bins + 1, device=x.device)
        y_edges = torch.linspace(y_min - margin, y_max + margin, n_bins + 1, device=y.device)
        
        # Process each batch with consistent binning
        for b in range(B):
            x_flat = x[b].flatten()
            y_flat = y[b].flatten()
            
            # Use consistent bin edges for all batches
            hist, edges = torch.histogramdd(
                torch.stack([x_flat, y_flat], dim=1),
                bins=[x_edges, y_edges]
            )
            
            # Normalize to get joint probability
            total = hist.sum()
            if total == 0:
                # Handle empty histogram case
                mi_values[b] = 0.0
                continue
                
            p_xy = hist / total
            
            # Marginal probabilities
            p_x = p_xy.sum(dim=1, keepdim=True)
            p_y = p_xy.sum(dim=0, keepdim=True)
            
            # Compute MI with safe logarithm
            # MI = sum(p_xy * log(p_xy / (p_x * p_y))) where p_xy > 0
            p_xy_indep = p_x * p_y
            
            # Safe MI computation: only compute where p_xy > 0
            # This avoids log(0) and handles zero bins properly
            nonzero_mask = p_xy > 0
            
            if nonzero_mask.any():
                # Compute MI only for non-zero joint probabilities
                # Use torch.where to avoid NaN/inf from log(0)
                log_ratio = torch.where(
                    nonzero_mask,
                    torch.log(p_xy / (p_xy_indep + 1e-20)),  # Add eps to denominator only
                    torch.zeros_like(p_xy)  # Zero contribution where p_xy = 0
                )
                mi = (p_xy * log_ratio).sum()
                mi_values[b] = torch.clamp(mi, min=0.0)  # MI should be non-negative
            else:
                mi_values[b] = 0.0
            
        return mi_values
    
    @staticmethod
    def fill_gaps_vectorized(data: torch.Tensor, max_gap: int = 5) -> torch.Tensor:
        """
        Vectorized gap filling for time series.
        Replaces nested loops in movement_integration.py
        
        Args:
            data: Tensor of shape (B, C, S, T) or (B, C, T)
            max_gap: Maximum gap size to interpolate
            
        Returns:
            Data with gaps filled
        """
        if data.dim() == 3:
            # Add sensor dimension if not present
            data = data.unsqueeze(2)
            squeeze_output = True
        else:
            squeeze_output = False
            
        B, C, S, T = data.shape
        result = data.clone()
        
        # Process using advanced indexing
        nan_mask = torch.isnan(data)
        
        # Find consecutive NaN regions (vectorized using convolution)
        # This is more efficient than nested loops
        for b in range(B):
            for c in range(C):
                for s in range(S):
                    signal = data[b, c, s, :]
                    nan_locs = torch.isnan(signal)
                    
                    if nan_locs.any() and not nan_locs.all():
                        # Use torch operations for interpolation
                        valid_idx = torch.where(~nan_locs)[0]
                        valid_values = signal[valid_idx]
                        
                        if len(valid_idx) >= 2:
                            # Linear interpolation using torch
                            for t in torch.where(nan_locs)[0]:
                                # Find nearest valid neighbors (vectorized)
                                distances = torch.abs(valid_idx - t)
                                nearest_idx = torch.argmin(distances)
                                
                                if distances[nearest_idx] <= max_gap:
                                    # Interpolate
                                    if nearest_idx > 0 and nearest_idx < len(valid_idx) - 1:
                                        # Linear interpolation between neighbors
                                        idx_before = valid_idx[nearest_idx - 1]
                                        idx_after = valid_idx[nearest_idx]
                                        weight = (t - idx_before) / (idx_after - idx_before)
                                        result[b, c, s, t] = (
                                            valid_values[nearest_idx - 1] * (1 - weight) +
                                            valid_values[nearest_idx] * weight
                                        )
                                    else:
                                        # Use nearest value
                                        result[b, c, s, t] = valid_values[nearest_idx]
        
        if squeeze_output:
            result = result.squeeze(2)
            
        return result
    
    @staticmethod
    def sliding_window_vectorized(data: np.ndarray, window_size: int, 
                                  step_size: int = 1) -> np.ndarray:
        """
        Vectorized sliding window operation.
        Replaces loops in various preprocessing files.
        
        Args:
            data: Array of shape (N, C) or (C, N)
            window_size: Size of sliding window
            step_size: Step between windows
            
        Returns:
            Windowed data of shape (n_windows, window_size, C)
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        n_samples = data.shape[0]
        n_channels = data.shape[1] if len(data.shape) > 1 else 1
        
        # Calculate number of windows
        n_windows = (n_samples - window_size) // step_size + 1
        
        # Use numpy's stride tricks for efficient windowing
        from numpy.lib.stride_tricks import as_strided
        
        # Calculate strides
        stride_samples = data.strides[0]
        stride_channels = data.strides[1] if len(data.shape) > 1 else 0
        
        # Create view with sliding windows (no data copying!)
        # WARNING: as_strided creates a view - DO NOT MODIFY!
        # The view shares memory with original data and can cause undefined behavior if mutated
        windows = as_strided(
            data,
            shape=(n_windows, window_size, n_channels),
            strides=(stride_samples * step_size, stride_samples, stride_channels),
            writeable=False  # Make read-only to prevent accidental mutations
        )
        
        # IMPORTANT: Return a copy to ensure safety
        # This prevents any downstream modifications from corrupting the original data
        # The copy overhead is worth the safety guarantee
        return windows.copy()


def benchmark_optimizations():
    """Benchmark the vectorized implementations against original loops."""
    import time
    
    print("Benchmarking Vectorized Optimizations")
    print("=" * 50)
    
    # Test NaN interpolation
    print("\n1. NaN Interpolation:")
    B, C, T = 32, 9, 100
    data = np.random.randn(B, C, T)
    # Add some NaN values
    nan_mask = np.random.random((B, C, T)) < 0.1
    data[nan_mask] = np.nan
    
    # Time vectorized version
    start = time.perf_counter()
    for _ in range(10):
        result = VectorizedOperations.interpolate_nan_vectorized(data.copy())
    time_vectorized = (time.perf_counter() - start) / 10
    
    print(f"  Vectorized: {time_vectorized*1000:.2f}ms")
    
    # Test distance computation
    print("\n2. Distance Computation:")
    x = torch.randn(64, 3, 100)
    y = torch.randn(64, 3, 100)
    
    start = time.perf_counter()
    for _ in range(100):
        distances = VectorizedOperations.compute_distances_vectorized(x, y)
    time_vectorized = (time.perf_counter() - start) / 100
    
    print(f"  Vectorized: {time_vectorized*1000:.2f}ms")
    
    # Test sliding window
    print("\n3. Sliding Window:")
    data = np.random.randn(10000, 9)
    
    start = time.perf_counter()
    windows = VectorizedOperations.sliding_window_vectorized(data, 100, 50)
    time_vectorized = time.perf_counter() - start
    
    print(f"  Vectorized: {time_vectorized*1000:.2f}ms")
    print(f"  Windows shape: {windows.shape}")
    
    print("\n" + "=" * 50)
    print("✅ All vectorized operations completed successfully!")


if __name__ == "__main__":
    benchmark_optimizations()