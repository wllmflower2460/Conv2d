"""Integration with Movement library for advanced preprocessing and diagnostics.

This module provides a bridge between the Movement library's filtering/kinematics
capabilities and our Conv2d-VQ-HDP-HSMM pipeline. It handles IMU data preprocessing
with advanced filtering, interpolation, and kinematic feature extraction.

Key Features:
    - Gap filling via temporal interpolation 
    - Rolling window smoothing (median/mean)
    - Savitzky-Golay polynomial smoothing
    - Velocity and acceleration computation
    - Diagnostic reporting for data quality
"""

import sys
import numpy as np
import torch
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Literal, Union
import warnings

# Add Movement library to path if needed
movement_path = Path('/home/wllmflower/Development/movement')
if movement_path.exists() and str(movement_path) not in sys.path:
    sys.path.insert(0, str(movement_path))

try:
    from movement.filtering import (
        interpolate_over_time,
        rolling_filter, 
        savgol_filter,
        filter_by_confidence
    )
    from movement.kinematics.kinematics import (
        compute_time_derivative,
        compute_velocity,
        compute_acceleration,
        compute_displacement
    )
    from movement.utils.reports import report_nan_values
    MOVEMENT_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Movement library not available: {e}. Some features will be disabled.")
    MOVEMENT_AVAILABLE = False


class MovementPreprocessor:
    """Advanced preprocessing using Movement library capabilities.
    
    Designed for IMU data with shape (B, 9, 2, T) where:
    - B: batch size
    - 9: IMU channels (3 accel + 3 gyro + 3 mag)
    - 2: dual sensor configuration
    - T: time steps (default 100)
    """
    
    def __init__(self, sampling_rate: float = 100.0, verbose: bool = False):
        """Initialize Movement preprocessor.
        
        Args:
            sampling_rate: Data sampling rate in Hz
            verbose: Whether to print diagnostic reports
        """
        if not MOVEMENT_AVAILABLE:
            raise RuntimeError("Movement library is required but not available")
            
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.verbose = verbose
        
    def tensor_to_xarray(self, data: torch.Tensor, 
                        sensor_names: Optional[list] = None) -> xr.DataArray:
        """Convert PyTorch tensor to xarray DataArray for Movement processing.
        
        Args:
            data: Input tensor with shape (B, 9, 2, T) or (9, 2, T)
            sensor_names: Optional names for sensors
            
        Returns:
            xarray.DataArray with proper dimensions and coordinates
        """
        if data.dim() == 4:  # Batched
            B, channels, sensors, timesteps = data.shape
            np_data = data.cpu().numpy() if data.is_cuda else data.numpy()
            
            coords = {
                'batch': np.arange(B),
                'channel': ['accel_x', 'accel_y', 'accel_z',
                           'gyro_x', 'gyro_y', 'gyro_z',
                           'mag_x', 'mag_y', 'mag_z'],
                'sensor': sensor_names or ['sensor_1', 'sensor_2'],
                'time': np.arange(timesteps) * self.dt
            }
            
            return xr.DataArray(
                np_data,
                dims=['batch', 'channel', 'sensor', 'time'],
                coords=coords,
                name='imu_data'
            )
        
        elif data.dim() == 3:  # Single sample
            channels, sensors, timesteps = data.shape
            np_data = data.cpu().numpy() if data.is_cuda else data.numpy()
            
            coords = {
                'channel': ['accel_x', 'accel_y', 'accel_z',
                           'gyro_x', 'gyro_y', 'gyro_z',
                           'mag_x', 'mag_y', 'mag_z'],
                'sensor': sensor_names or ['sensor_1', 'sensor_2'],
                'time': np.arange(timesteps) * self.dt
            }
            
            return xr.DataArray(
                np_data,
                dims=['channel', 'sensor', 'time'],
                coords=coords,
                name='imu_data'
            )
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {data.shape}")
    
    def xarray_to_tensor(self, data: xr.DataArray, 
                        device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert xarray DataArray back to PyTorch tensor.
        
        Args:
            data: xarray DataArray to convert
            device: Target device for tensor
            
        Returns:
            PyTorch tensor with original shape
        """
        np_data = data.values
        tensor = torch.from_numpy(np_data).float()
        
        if device is not None:
            tensor = tensor.to(device)
            
        return tensor
    
    def interpolate_gaps(self, data: torch.Tensor,
                        method: str = 'linear',
                        max_gap: Optional[int] = 5) -> torch.Tensor:
        """Fill missing values using temporal interpolation.
        
        Args:
            data: Input tensor with potential NaN values
            method: Interpolation method ('linear', 'quadratic', 'cubic')
            max_gap: Maximum consecutive NaNs to fill
            
        Returns:
            Interpolated tensor
        """
        device = data.device
        
        # Try xarray interpolation, fall back to torch if it fails
        try:
            xr_data = self.tensor_to_xarray(data)
            
            # Apply interpolation per batch/channel/sensor
            if 'batch' in xr_data.dims:
                interpolated = []
                for b in range(xr_data.shape[0]):
                    batch_data = xr_data.isel(batch=b)
                    batch_interp = interpolate_over_time(
                        batch_data,
                        method=method,
                        max_gap=max_gap,
                        print_report=self.verbose
                    )
                    interpolated.append(batch_interp)
                result = xr.concat(interpolated, dim='batch')
            else:
                result = interpolate_over_time(
                    xr_data,
                    method=method,
                    max_gap=max_gap,
                    print_report=self.verbose
                )
            
            return self.xarray_to_tensor(result, device)
            
        except (ImportError, ModuleNotFoundError):
            # Fallback to simple torch interpolation
            warnings.warn("xarray interpolation failed, using simple torch interpolation")
            return self._torch_interpolate_gaps(data, max_gap)
    
    def _torch_interpolate_gaps(self, data: torch.Tensor, max_gap: Optional[int] = None) -> torch.Tensor:
        """Simple torch-based gap interpolation."""
        result = data.clone()
        
        # Process each batch, channel, sensor
        if data.dim() == 4:
            B, C, S, T = data.shape
            for b in range(B):
                for c in range(C):
                    for s in range(S):
                        signal = data[b, c, s, :]
                        nan_mask = torch.isnan(signal)
                        
                        if nan_mask.any() and not nan_mask.all():
                            # Find valid indices
                            valid_idx = torch.where(~nan_mask)[0]
                            valid_values = signal[valid_idx]
                            
                            # Interpolate
                            for i in range(T):
                                if nan_mask[i]:
                                    # Find nearest valid neighbors
                                    if len(valid_idx) > 0:
                                        distances = torch.abs(valid_idx - i)
                                        
                                        # Check max_gap constraint
                                        min_dist = distances.min()
                                        if max_gap is None or min_dist <= max_gap:
                                            # Linear interpolation between nearest points
                                            if i < valid_idx[0]:
                                                result[b, c, s, i] = valid_values[0]
                                            elif i > valid_idx[-1]:
                                                result[b, c, s, i] = valid_values[-1]
                                            else:
                                                # Find surrounding valid points
                                                left_idx = valid_idx[valid_idx < i].max()
                                                right_idx = valid_idx[valid_idx > i].min()
                                                left_val = signal[left_idx]
                                                right_val = signal[right_idx]
                                                
                                                # Linear interpolation
                                                alpha = (i - left_idx) / (right_idx - left_idx)
                                                result[b, c, s, i] = left_val * (1 - alpha) + right_val * alpha
        
        return result
    
    def smooth_rolling(self, data: torch.Tensor,
                      window: int = 5,
                      statistic: Literal["median", "mean", "max", "min"] = "median",
                      min_periods: Optional[int] = None) -> torch.Tensor:
        """Apply rolling window smoothing.
        
        Args:
            data: Input tensor to smooth
            window: Size of rolling window
            statistic: Statistic to compute ('median', 'mean', 'max', 'min')
            min_periods: Minimum valid observations in window
            
        Returns:
            Smoothed tensor
        """
        device = data.device
        xr_data = self.tensor_to_xarray(data)
        
        # Apply rolling filter
        if 'batch' in xr_data.dims:
            smoothed = []
            for b in range(xr_data.shape[0]):
                batch_data = xr_data.isel(batch=b)
                batch_smooth = rolling_filter(
                    batch_data,
                    window=window,
                    statistic=statistic,
                    min_periods=min_periods,
                    print_report=self.verbose
                )
                smoothed.append(batch_smooth)
            result = xr.concat(smoothed, dim='batch')
        else:
            result = rolling_filter(
                xr_data,
                window=window,
                statistic=statistic,
                min_periods=min_periods,
                print_report=self.verbose
            )
        
        return self.xarray_to_tensor(result, device)
    
    def smooth_savgol(self, data: torch.Tensor,
                     window: int = 7,
                     polyorder: int = 2) -> torch.Tensor:
        """Apply Savitzky-Golay polynomial smoothing.
        
        Args:
            data: Input tensor to smooth
            window: Size of smoothing window (must be odd)
            polyorder: Polynomial order for fitting
            
        Returns:
            Smoothed tensor preserving peaks
        """
        if window % 2 == 0:
            window += 1  # Ensure odd window
            
        device = data.device
        xr_data = self.tensor_to_xarray(data)
        
        # Apply Savitzky-Golay filter
        if 'batch' in xr_data.dims:
            smoothed = []
            for b in range(xr_data.shape[0]):
                batch_data = xr_data.isel(batch=b)
                batch_smooth = savgol_filter(
                    batch_data,
                    window=window,
                    polyorder=polyorder,
                    print_report=self.verbose
                )
                smoothed.append(batch_smooth)
            result = xr.concat(smoothed, dim='batch')
        else:
            result = savgol_filter(
                xr_data,
                window=window,
                polyorder=polyorder,
                print_report=self.verbose
            )
        
        return self.xarray_to_tensor(result, device)
    
    def compute_kinematics(self, data: torch.Tensor,
                          compute_vel: bool = True,
                          compute_acc: bool = True) -> Dict[str, torch.Tensor]:
        """Compute velocity and acceleration from position data.
        
        Args:
            data: Input position/orientation data
            compute_vel: Whether to compute velocity
            compute_acc: Whether to compute acceleration
            
        Returns:
            Dictionary with 'position', 'velocity', 'acceleration' tensors
        """
        device = data.device
        xr_data = self.tensor_to_xarray(data)
        
        results = {'position': data}
        
        # For IMU data, we need to handle channels differently
        # Accelerometer already gives acceleration, gyro gives angular velocity
        # So we compute derivatives for diagnostic purposes
        
        if compute_vel:
            # First derivative (rate of change)
            if 'batch' in xr_data.dims:
                vel_list = []
                for b in range(xr_data.shape[0]):
                    batch_data = xr_data.isel(batch=b)
                    batch_vel = compute_time_derivative(batch_data, order=1)
                    vel_list.append(batch_vel)
                velocity = xr.concat(vel_list, dim='batch')
            else:
                velocity = compute_time_derivative(xr_data, order=1)
            
            results['velocity_derivative'] = self.xarray_to_tensor(velocity, device)
        
        if compute_acc:
            # Second derivative
            if 'batch' in xr_data.dims:
                acc_list = []
                for b in range(xr_data.shape[0]):
                    batch_data = xr_data.isel(batch=b)
                    batch_acc = compute_time_derivative(batch_data, order=2)
                    acc_list.append(batch_acc)
                acceleration = xr.concat(acc_list, dim='batch')
            else:
                acceleration = compute_time_derivative(xr_data, order=2)
            
            results['acceleration_derivative'] = self.xarray_to_tensor(acceleration, device)
        
        return results
    
    def preprocess_pipeline(self, data: torch.Tensor,
                           interpolate: bool = True,
                           smooth_method: Optional[str] = 'median',
                           smooth_window: int = 5,
                           compute_derivatives: bool = False) -> Dict[str, torch.Tensor]:
        """Complete preprocessing pipeline.
        
        Args:
            data: Raw IMU tensor (B, 9, 2, T)
            interpolate: Whether to fill gaps
            smooth_method: 'median', 'mean', 'savgol', or None
            smooth_window: Window size for smoothing
            compute_derivatives: Whether to compute kinematic derivatives
            
        Returns:
            Dictionary with processed tensors
        """
        results = {'raw': data.clone()}
        
        # Step 1: Interpolate gaps
        if interpolate:
            data = self.interpolate_gaps(data)
            results['interpolated'] = data.clone()
        
        # Step 2: Smooth
        if smooth_method == 'median':
            data = self.smooth_rolling(data, window=smooth_window, statistic='median')
        elif smooth_method == 'mean':
            data = self.smooth_rolling(data, window=smooth_window, statistic='mean')
        elif smooth_method == 'savgol':
            data = self.smooth_savgol(data, window=smooth_window)
        
        if smooth_method is not None:
            results['smoothed'] = data.clone()
        
        # Step 3: Compute derivatives if requested
        if compute_derivatives:
            kin_results = self.compute_kinematics(data)
            results.update(kin_results)
        
        results['processed'] = data
        
        return results
    
    def generate_diagnostic_report(self, data: torch.Tensor) -> str:
        """Generate diagnostic report for data quality.
        
        Args:
            data: Input tensor to analyze
            
        Returns:
            String report with data quality metrics
        """
        xr_data = self.tensor_to_xarray(data)
        
        report_lines = ["=== IMU Data Diagnostic Report ===\n"]
        
        # Basic statistics
        report_lines.append(f"Shape: {data.shape}")
        report_lines.append(f"Device: {data.device}")
        report_lines.append(f"Dtype: {data.dtype}")
        report_lines.append(f"Memory: {data.element_size() * data.nelement() / 1024:.2f} KB\n")
        
        # NaN analysis
        nan_count = torch.isnan(data).sum().item()
        nan_pct = 100 * nan_count / data.numel()
        report_lines.append(f"NaN values: {nan_count} ({nan_pct:.2f}%)")
        
        # Channel-wise statistics
        if data.dim() == 4:
            for ch_idx, ch_name in enumerate(xr_data.coords['channel'].values):
                ch_data = data[:, ch_idx, :, :]
                ch_mean = torch.mean(ch_data[~torch.isnan(ch_data)]).item()
                ch_std = torch.std(ch_data[~torch.isnan(ch_data)]).item()
                ch_min = torch.min(ch_data[~torch.isnan(ch_data)]).item()
                ch_max = torch.max(ch_data[~torch.isnan(ch_data)]).item()
                
                report_lines.append(
                    f"\n{ch_name}: mean={ch_mean:.3f}, std={ch_std:.3f}, "
                    f"range=[{ch_min:.3f}, {ch_max:.3f}]"
                )
        
        # Gap analysis (consecutive NaNs)
        if nan_count > 0:
            report_lines.append("\n=== Gap Analysis ===")
            # Simplified gap detection
            nan_mask = torch.isnan(data)
            if data.dim() == 4:
                # Check time axis for gaps
                time_axis_nans = nan_mask.any(dim=(0,1,2))
                gaps = self._find_consecutive_gaps(time_axis_nans)
                if gaps:
                    report_lines.append(f"Time gaps found: {len(gaps)}")
                    max_gap = max(g[1] - g[0] for g in gaps)
                    report_lines.append(f"Longest gap: {max_gap} timesteps")
        
        return '\n'.join(report_lines)
    
    @staticmethod
    def _find_consecutive_gaps(mask: torch.Tensor) -> list:
        """Find consecutive True values in 1D boolean tensor."""
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, val in enumerate(mask):
            if val and not in_gap:
                gap_start = i
                in_gap = True
            elif not val and in_gap:
                gaps.append((gap_start, i))
                in_gap = False
        
        if in_gap:
            gaps.append((gap_start, len(mask)))
        
        return gaps


class MovementDiagnostics:
    """Diagnostic utilities using Movement library capabilities."""
    
    def __init__(self, preprocessor: Optional[MovementPreprocessor] = None):
        """Initialize diagnostics.
        
        Args:
            preprocessor: Optional preprocessor instance to use
        """
        self.preprocessor = preprocessor or MovementPreprocessor()
    
    def analyze_data_quality(self, data: torch.Tensor, 
                            confidence: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Comprehensive data quality analysis.
        
        Args:
            data: Input data tensor
            confidence: Optional confidence scores
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['shape'] = data.shape
        metrics['nan_count'] = torch.isnan(data).sum().item()
        metrics['nan_percentage'] = 100 * metrics['nan_count'] / data.numel()
        metrics['inf_count'] = torch.isinf(data).sum().item()
        
        # Value ranges
        valid_data = data[~torch.isnan(data) & ~torch.isinf(data)]
        if valid_data.numel() > 0:
            metrics['min'] = valid_data.min().item()
            metrics['max'] = valid_data.max().item()
            metrics['mean'] = valid_data.mean().item()
            metrics['std'] = valid_data.std().item()
        
        # Signal quality metrics
        if data.dim() >= 3:  # Has time dimension
            # Compute signal-to-noise ratio estimate
            time_dim = -1
            signal_power = torch.var(data, dim=time_dim, keepdim=True)
            noise_estimate = torch.var(torch.diff(data, dim=time_dim), dim=time_dim, keepdim=True) / 2
            
            snr = signal_power / (noise_estimate + 1e-10)
            metrics['mean_snr_db'] = (10 * torch.log10(snr.mean())).item()
        
        # Confidence analysis if provided
        if confidence is not None:
            metrics['confidence_mean'] = confidence.mean().item()
            metrics['confidence_std'] = confidence.std().item()
            metrics['low_confidence_ratio'] = (confidence < 0.5).float().mean().item()
        
        return metrics
    
    def compare_preprocessing_methods(self, data: torch.Tensor) -> Dict[str, Dict]:
        """Compare different preprocessing approaches.
        
        Args:
            data: Raw input data
            
        Returns:
            Dictionary comparing different methods
        """
        comparisons = {}
        
        # Original data metrics
        comparisons['original'] = self.analyze_data_quality(data)
        
        # Interpolated only
        interp_data = self.preprocessor.interpolate_gaps(data)
        comparisons['interpolated'] = self.analyze_data_quality(interp_data)
        
        # Median smoothed
        median_data = self.preprocessor.smooth_rolling(data, statistic='median')
        comparisons['median_smoothed'] = self.analyze_data_quality(median_data)
        
        # Savgol smoothed
        savgol_data = self.preprocessor.smooth_savgol(data)
        comparisons['savgol_smoothed'] = self.analyze_data_quality(savgol_data)
        
        # Full pipeline
        full_pipeline = self.preprocessor.preprocess_pipeline(
            data, 
            interpolate=True,
            smooth_method='median'
        )
        comparisons['full_pipeline'] = self.analyze_data_quality(full_pipeline['processed'])
        
        return comparisons


def create_movement_preprocessor(config: Optional[Dict] = None) -> MovementPreprocessor:
    """Factory function to create configured preprocessor.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured MovementPreprocessor instance
    """
    default_config = {
        'sampling_rate': 100.0,
        'verbose': False
    }
    
    if config:
        default_config.update(config)
    
    return MovementPreprocessor(**default_config)


if __name__ == "__main__":
    # Test with synthetic IMU data
    print("Testing Movement integration with synthetic IMU data...")
    
    # Create synthetic data (B=4, 9 channels, 2 sensors, 100 timesteps)
    B, C, S, T = 4, 9, 2, 100
    test_data = torch.randn(B, C, S, T)
    
    # Add some NaN values to simulate dropouts
    dropout_mask = torch.rand_like(test_data) < 0.05  # 5% dropout
    test_data[dropout_mask] = float('nan')
    
    # Initialize preprocessor
    preprocessor = MovementPreprocessor(sampling_rate=100.0, verbose=True)
    
    # Test preprocessing pipeline
    print("\nRunning preprocessing pipeline...")
    results = preprocessor.preprocess_pipeline(
        test_data,
        interpolate=True,
        smooth_method='median',
        smooth_window=5,
        compute_derivatives=True
    )
    
    print(f"\nPipeline results keys: {results.keys()}")
    
    # Generate diagnostic report
    print("\nGenerating diagnostic report...")
    report = preprocessor.generate_diagnostic_report(results['processed'])
    print(report)
    
    # Test diagnostics
    diagnostics = MovementDiagnostics(preprocessor)
    quality_metrics = diagnostics.analyze_data_quality(test_data)
    
    print("\n=== Data Quality Metrics ===")
    for key, value in quality_metrics.items():
        print(f"{key}: {value}")
    
    # Compare methods
    print("\n=== Method Comparison ===")
    comparisons = diagnostics.compare_preprocessing_methods(test_data)
    for method, metrics in comparisons.items():
        print(f"\n{method}:")
        print(f"  NaN%: {metrics.get('nan_percentage', 0):.2f}%")
        print(f"  Mean: {metrics.get('mean', 0):.3f}")
        print(f"  Std: {metrics.get('std', 0):.3f}")
        if 'mean_snr_db' in metrics:
            print(f"  SNR: {metrics['mean_snr_db']:.2f} dB")
    
    print("\n✅ Movement integration test complete!")