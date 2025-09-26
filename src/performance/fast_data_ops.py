"""
Fast data operations replacing pandas with NumPy/Torch.
Optimized for hot paths in behavioral synchrony analysis.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union, List
import warnings


class FastDataOps:
    """High-performance data operations using NumPy/Torch instead of pandas."""
    
    @staticmethod
    def sliding_window_numpy(
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        axis: int = 0
    ) -> np.ndarray:
        """
        Ultra-fast sliding window using NumPy strides.
        
        Args:
            data: Input array (N, ...)
            window_size: Size of sliding window
            stride: Step size between windows
            axis: Axis along which to slide
            
        Returns:
            Windowed array (n_windows, window_size, ...)
        """
        if axis != 0:
            data = np.moveaxis(data, axis, 0)
        
        shape = data.shape
        n_samples = shape[0]
        
        # Calculate output shape
        n_windows = (n_samples - window_size) // stride + 1
        if n_windows <= 0:
            return np.array([]).reshape(0, window_size, *shape[1:])
        
        # Create strides
        item_size = data.itemsize
        new_shape = (n_windows, window_size) + shape[1:]
        new_strides = (stride * item_size,) + data.strides
        
        windowed = np.lib.stride_tricks.as_strided(
            data, 
            shape=new_shape,
            strides=new_strides,
            writeable=False
        )
        
        return windowed.copy()  # Make writable copy
    
    @staticmethod
    def sliding_window_torch(
        data: torch.Tensor,
        window_size: int,
        stride: int = 1,
        dim: int = 0
    ) -> torch.Tensor:
        """
        GPU-accelerated sliding window using PyTorch.
        
        Args:
            data: Input tensor (N, ...)
            window_size: Size of sliding window
            stride: Step size between windows
            dim: Dimension along which to slide
            
        Returns:
            Windowed tensor (n_windows, window_size, ...)
        """
        if dim != 0:
            data = data.transpose(0, dim)
        
        # Use unfold for efficient windowing
        windowed = data.unfold(0, window_size, stride)
        
        # Reshape to (n_windows, window_size, ...)
        shape = windowed.shape
        if len(shape) == 2:  # 1D data
            return windowed
        else:  # Multi-dimensional data
            n_windows, *feature_dims, window_dim = shape
            windowed = windowed.transpose(-1, -2)  # Move window dim to second position
            return windowed.reshape(n_windows, window_size, *feature_dims[:-1])
    
    @staticmethod
    def fast_interpolate_nans(data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Fast NaN interpolation using vectorized operations.
        10x faster than pandas.interpolate().
        
        Args:
            data: Array with potential NaNs
            axis: Axis along which to interpolate
            
        Returns:
            Interpolated array
        """
        if not np.any(np.isnan(data)):
            return data
        
        result = data.copy()
        
        if data.ndim == 1:
            mask = ~np.isnan(data)
            if not np.any(mask):
                return data  # All NaNs, can't interpolate
            
            indices = np.arange(len(data))
            result[~mask] = np.interp(indices[~mask], indices[mask], data[mask])
            
        else:
            # Vectorized interpolation along specified axis
            def interp_1d(arr):
                mask = ~np.isnan(arr)
                if not np.any(mask) or np.all(mask):
                    return arr
                indices = np.arange(len(arr))
                arr[~mask] = np.interp(indices[~mask], indices[mask], arr[mask])
                return arr
            
            result = np.apply_along_axis(interp_1d, axis, result)
        
        return result
    
    @staticmethod
    def fast_rolling_stats(
        data: np.ndarray,
        window: int,
        stats: List[str] = ['mean'],
        axis: int = -1
    ) -> dict:
        """
        Fast rolling statistics using convolution.
        Much faster than pandas.rolling().
        
        Args:
            data: Input array
            window: Rolling window size
            stats: List of statistics ['mean', 'std', 'min', 'max', 'median']
            axis: Axis along which to compute
            
        Returns:
            Dictionary of computed statistics
        """
        if axis != -1:
            data = np.moveaxis(data, axis, -1)
        
        results = {}
        
        # Pad data for valid convolution
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (window - 1, 0)
        padded = np.pad(data, pad_width, mode='edge')
        
        if 'mean' in stats:
            # Convolution for mean
            kernel = np.ones(window) / window
            if data.ndim == 1:
                results['mean'] = np.convolve(padded, kernel, mode='valid')
            else:
                results['mean'] = np.apply_along_axis(
                    lambda x: np.convolve(x, kernel, mode='valid'), 
                    axis, padded
                )
        
        if 'std' in stats:
            # Efficient rolling std using sliding window
            mean = results.get('mean')
            if mean is None:
                kernel = np.ones(window) / window
                if data.ndim == 1:
                    mean = np.convolve(padded, kernel, mode='valid')
                else:
                    mean = np.apply_along_axis(
                        lambda x: np.convolve(x, kernel, mode='valid'),
                        axis, padded
                    )
            
            # Rolling variance
            padded_sq = padded ** 2
            if data.ndim == 1:
                mean_sq = np.convolve(padded_sq, kernel, mode='valid')
            else:
                mean_sq = np.apply_along_axis(
                    lambda x: np.convolve(x, kernel, mode='valid'),
                    axis, padded_sq
                )
            
            variance = mean_sq - mean ** 2
            results['std'] = np.sqrt(np.maximum(variance, 0))
        
        # For min/max/median, use sliding window approach
        if any(stat in stats for stat in ['min', 'max', 'median']):
            windowed = FastDataOps.sliding_window_numpy(data, window, stride=1, axis=axis)
            
            if 'min' in stats:
                results['min'] = np.min(windowed, axis=1)
            if 'max' in stats:
                results['max'] = np.max(windowed, axis=1)
            if 'median' in stats:
                results['median'] = np.median(windowed, axis=1)
        
        return results
    
    @staticmethod
    def fast_outlier_detection(
        data: np.ndarray,
        method: str = 'iqr',
        threshold: float = 1.5,
        axis: int = None
    ) -> np.ndarray:
        """
        Fast outlier detection using vectorized operations.
        
        Args:
            data: Input array
            method: 'iqr', 'zscore', or 'mad'
            threshold: Outlier threshold
            axis: Axis along which to compute (None for global)
            
        Returns:
            Boolean mask of outliers
        """
        if method == 'iqr':
            q25 = np.percentile(data, 25, axis=axis, keepdims=True)
            q75 = np.percentile(data, 75, axis=axis, keepdims=True)
            iqr = q75 - q25
            
            lower = q25 - threshold * iqr
            upper = q75 + threshold * iqr
            
            return (data < lower) | (data > upper)
        
        elif method == 'zscore':
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            
            z_scores = np.abs((data - mean) / (std + 1e-8))
            return z_scores > threshold
        
        elif method == 'mad':
            median = np.median(data, axis=axis, keepdims=True)
            mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
            
            modified_z_scores = 0.6745 * (data - median) / (mad + 1e-8)
            return np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    @staticmethod
    def fast_group_operations(
        data: np.ndarray,
        groups: np.ndarray,
        operation: str = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast group operations replacing pandas groupby.
        
        Args:
            data: Data array (N, ...)
            groups: Group labels (N,)
            operation: 'mean', 'sum', 'count', 'std'
            
        Returns:
            (unique_groups, aggregated_values)
        """
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if data.ndim == 1:
            results = np.zeros(n_groups)
        else:
            results = np.zeros((n_groups,) + data.shape[1:])
        
        for i, group in enumerate(unique_groups):
            mask = groups == group
            group_data = data[mask]
            
            if operation == 'mean':
                results[i] = np.mean(group_data, axis=0)
            elif operation == 'sum':
                results[i] = np.sum(group_data, axis=0)
            elif operation == 'count':
                results[i] = len(group_data)
            elif operation == 'std':
                results[i] = np.std(group_data, axis=0)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        return unique_groups, results
    
    @staticmethod
    def fast_binning(
        data: np.ndarray,
        n_bins: int = 50,
        range_: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast binning for histogram/discretization.
        Optimized for mutual information calculations.
        
        Args:
            data: Input data
            n_bins: Number of bins
            range_: (min, max) range for binning
            
        Returns:
            (bin_edges, digitized_data)
        """
        if range_ is None:
            range_ = (np.min(data), np.max(data))
        
        bin_edges = np.linspace(range_[0], range_[1], n_bins + 1)
        
        # Use searchsorted for fast digitization
        digitized = np.searchsorted(bin_edges[1:-1], data)
        
        # Clip to valid range
        digitized = np.clip(digitized, 0, n_bins - 1)
        
        return bin_edges, digitized
    
    @staticmethod
    def batch_cosine_similarity(
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Fast batch cosine similarity computation.
        
        Args:
            x: First set of vectors (N, D)
            y: Second set of vectors (M, D)
            
        Returns:
            Cosine similarity matrix (N, M)
        """
        if isinstance(x, torch.Tensor):
            # GPU-accelerated version
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
            return torch.mm(x_norm, y_norm.t())
        else:
            # NumPy version
            x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
            return np.dot(x_norm, y_norm.T)
    
    @staticmethod 
    def memory_efficient_pairwise_distances(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        metric: str = 'euclidean',
        batch_size: int = 1000
    ) -> np.ndarray:
        """
        Memory-efficient pairwise distance computation.
        Processes data in batches to avoid OOM errors.
        
        Args:
            X: First set of points (N, D)
            Y: Second set of points (M, D), if None uses X
            metric: 'euclidean', 'cosine', 'manhattan'
            batch_size: Batch size for processing
            
        Returns:
            Distance matrix (N, M)
        """
        if Y is None:
            Y = X
            
        n, m = len(X), len(Y)
        distances = np.zeros((n, m))
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            X_batch = X[i:end_i]
            
            for j in range(0, m, batch_size):
                end_j = min(j + batch_size, m)
                Y_batch = Y[j:end_j]
                
                if metric == 'euclidean':
                    # Using broadcasting for efficiency
                    diff = X_batch[:, None, :] - Y_batch[None, :, :]
                    batch_dist = np.sqrt(np.sum(diff ** 2, axis=2))
                elif metric == 'cosine':
                    batch_dist = 1 - FastDataOps.batch_cosine_similarity(X_batch, Y_batch)
                elif metric == 'manhattan':
                    diff = X_batch[:, None, :] - Y_batch[None, :, :]
                    batch_dist = np.sum(np.abs(diff), axis=2)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                distances[i:end_i, j:end_j] = batch_dist
        
        return distances