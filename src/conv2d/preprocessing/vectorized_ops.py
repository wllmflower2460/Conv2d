"""Vectorized operations with full type hints."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor


class VectorizedOperations:
    """Collection of vectorized operations with type safety."""
    
    @staticmethod
    def interpolate_nan_vectorized(
        data: NDArray[np.floating],
        method: str = "linear",
    ) -> NDArray[np.floating]:
        """Vectorized NaN interpolation for 3D arrays.
        
        Args:
            data: Array of shape (B, C, T) with possible NaN values
            method: Interpolation method ('linear', 'nearest', 'zero')
            
        Returns:
            Data with NaN values interpolated
        """
        if len(data.shape) != 3:
            return data
            
        B, C, T = data.shape
        
        # Reshape to (B*C, T) for batch processing
        data_reshaped = data.reshape(-1, T)
        
        # Create time index for interpolation
        x = np.arange(T, dtype=np.float32)
        
        # Process all channels at once
        for idx in range(data_reshaped.shape[0]):
            signal = data_reshaped[idx]
            if np.any(np.isnan(signal)):
                valid_mask = ~np.isnan(signal)
                if np.sum(valid_mask) >= 2:
                    # Use numpy's interp (vectorized internally)
                    data_reshaped[idx] = np.interp(
                        x, x[valid_mask], signal[valid_mask]
                    )
                else:
                    # Fall back to mean replacement
                    mean_val = (
                        np.nanmean(signal) 
                        if not np.all(np.isnan(signal)) 
                        else 0.0
                    )
                    data_reshaped[idx] = np.nan_to_num(signal, nan=mean_val)
        
        return data_reshaped.reshape(B, C, T)
    
    @staticmethod
    def compute_distances_vectorized(
        x: Tensor, 
        y: Tensor,
    ) -> Tensor:
        """Vectorized Euclidean distance computation.
        
        Args:
            x: Tensor of shape (B, C, T)
            y: Tensor of shape (B, C, T)
            
        Returns:
            Distances tensor of shape (B,)
            
        Raises:
            ValueError: If shapes don't match
        """
        if x.shape != y.shape:
            raise ValueError(
                f"Shape mismatch: x={x.shape}, y={y.shape}"
            )
        
        # Compute squared differences
        diff_squared = (x - y) ** 2
        
        # Sum over channel dimension and take sqrt
        distances = torch.sqrt(torch.sum(diff_squared, dim=1))
        
        # Mean over time dimension
        return torch.mean(distances, dim=1)
    
    @staticmethod
    def estimate_mutual_information_vectorized(
        x: Tensor,
        y: Tensor,
        n_bins: int = 10,
    ) -> Tensor:
        """Vectorized mutual information estimation with robust handling.
        
        Args:
            x: Tensor of shape (B, C, T)
            y: Tensor of shape (B, C, T)
            n_bins: Number of bins for histogram
            
        Returns:
            MI values tensor of shape (B,)
        """
        B = x.shape[0]
        device = x.device
        mi_values = torch.zeros(B, device=device)
        
        # Compute consistent bin edges across all batches
        x_min, x_max = x.min().item(), x.max().item()
        y_min, y_max = y.min().item(), y.max().item()
        
        # Add small margin to avoid edge effects
        margin = 1e-10
        x_edges = torch.linspace(
            x_min - margin, x_max + margin, n_bins + 1, device=device
        )
        y_edges = torch.linspace(
            y_min - margin, y_max + margin, n_bins + 1, device=device
        )
        
        # Process each batch with consistent binning
        for b in range(B):
            x_flat = x[b].flatten()
            y_flat = y[b].flatten()
            
            # Use consistent bin edges for all batches
            hist, _ = torch.histogramdd(
                torch.stack([x_flat, y_flat], dim=1),
                bins=[x_edges, y_edges],
            )
            
            # Normalize to get joint probability
            total = hist.sum()
            if total == 0:
                mi_values[b] = 0.0
                continue
                
            p_xy = hist / total
            
            # Marginal probabilities
            p_x = p_xy.sum(dim=1, keepdim=True)
            p_y = p_xy.sum(dim=0, keepdim=True)
            
            # Compute MI with safe logarithm
            p_xy_indep = p_x * p_y
            
            # Safe MI computation
            nonzero_mask = p_xy > 0
            
            if nonzero_mask.any():
                log_ratio = torch.where(
                    nonzero_mask,
                    torch.log(p_xy / (p_xy_indep + 1e-20)),
                    torch.zeros_like(p_xy),
                )
                mi = (p_xy * log_ratio).sum()
                mi_values[b] = torch.clamp(mi, min=0.0)
            else:
                mi_values[b] = 0.0
            
        return mi_values
    
    @staticmethod
    def sliding_window_vectorized(
        data: NDArray[np.floating],
        window_size: int,
        step_size: int = 1,
    ) -> NDArray[np.floating]:
        """Vectorized sliding window operation.
        
        Args:
            data: Array of shape (N, C) or (C, N)
            window_size: Size of sliding window
            step_size: Step between windows
            
        Returns:
            Windowed data of shape (n_windows, window_size, C)
            
        Warning:
            Returns a copy to ensure safety. Original data is never modified.
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        n_samples = data.shape[0]
        n_channels = data.shape[1] if len(data.shape) > 1 else 1
        
        # Calculate number of windows
        n_windows = (n_samples - window_size) // step_size + 1
        
        if n_windows <= 0:
            return np.array([]).reshape(0, window_size, n_channels)
        
        # Use numpy's stride tricks for efficient windowing
        from numpy.lib.stride_tricks import as_strided
        
        # Calculate strides
        stride_samples = data.strides[0]
        stride_channels = data.strides[1] if len(data.shape) > 1 else 0
        
        # Create view with sliding windows (no data copying!)
        # WARNING: as_strided creates a view - DO NOT MODIFY!
        windows = as_strided(
            data,
            shape=(n_windows, window_size, n_channels),
            strides=(
                stride_samples * step_size,
                stride_samples,
                stride_channels,
            ),
            writeable=False,  # Make read-only to prevent accidental mutations
        )
        
        # IMPORTANT: Return a copy to ensure safety
        return windows.copy()
    
    @staticmethod
    def fill_gaps_vectorized(
        data: Tensor,
        max_gap: int = 5,
    ) -> Tensor:
        """Vectorized gap filling for time series.
        
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
                                # Find nearest valid neighbors
                                distances = torch.abs(valid_idx - t)
                                nearest_idx = torch.argmin(distances)
                                
                                if distances[nearest_idx] <= max_gap:
                                    # Interpolate or use nearest
                                    if (
                                        nearest_idx > 0 
                                        and nearest_idx < len(valid_idx) - 1
                                    ):
                                        # Linear interpolation between neighbors
                                        idx_before = valid_idx[nearest_idx - 1]
                                        idx_after = valid_idx[nearest_idx]
                                        weight = float(t - idx_before) / float(
                                            idx_after - idx_before
                                        )
                                        result[b, c, s, t] = (
                                            valid_values[nearest_idx - 1] * (1 - weight)
                                            + valid_values[nearest_idx] * weight
                                        )
                                    else:
                                        # Use nearest value
                                        result[b, c, s, t] = valid_values[nearest_idx]
        
        if squeeze_output:
            result = result.squeeze(2)
            
        return result