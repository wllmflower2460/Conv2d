"""Preprocessing command implementation."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import time
from rich.progress import Progress

from ..qa_gates import QAResult


def validate_data(input_dir: Path, config: Dict[str, Any]) -> QAResult:
    """
    Validate input data quality.
    
    Checks:
    - File existence and format
    - NaN detection
    - Outlier detection (MAD-based)
    - Signal continuity
    """
    issues = []
    metrics = {}
    
    # Check for data files
    data_files = list(input_dir.glob("*.npy")) + list(input_dir.glob("*.npz"))
    if not data_files:
        issues.append(f"No data files found in {input_dir}")
        return QAResult(passed=False, issues=issues, metrics=metrics)
    
    # Load sample data for validation
    sample_data = np.load(data_files[0])
    if isinstance(sample_data, np.lib.npyio.NpzFile):
        sample_data = sample_data['data']
    
    # Check for NaNs
    nan_count = np.isnan(sample_data).sum()
    if nan_count > 0:
        nan_percent = 100 * nan_count / sample_data.size
        metrics['nan_percent'] = nan_percent
        if nan_percent > 5:
            issues.append(f"High NaN rate: {nan_percent:.1f}% (threshold: 5%)")
    
    # Check for outliers using MAD
    if len(sample_data.shape) >= 2:
        flat_data = sample_data.flatten()
        median = np.median(flat_data[~np.isnan(flat_data)])
        mad = np.median(np.abs(flat_data[~np.isnan(flat_data)] - median))
        
        if mad > 0:
            z_scores = np.abs((flat_data - median) / (1.4826 * mad))
            outlier_count = (z_scores > 5).sum()
            outlier_percent = 100 * outlier_count / len(flat_data)
            metrics['outlier_percent'] = outlier_percent
            
            if outlier_percent > 10:
                issues.append(f"High outlier rate: {outlier_percent:.1f}% (threshold: 10%)")
    
    # Check signal continuity
    if len(sample_data.shape) >= 2:
        diff = np.diff(sample_data, axis=-1)
        jump_threshold = 10 * np.std(diff[~np.isnan(diff)])
        large_jumps = (np.abs(diff) > jump_threshold).sum()
        
        if large_jumps > 100:
            issues.append(f"Signal discontinuities detected: {large_jumps} large jumps")
    
    metrics['total_samples'] = sample_data.shape[0] if len(sample_data.shape) > 0 else 1
    metrics['data_files'] = len(data_files)
    
    return QAResult(
        passed=len(issues) == 0,
        issues=issues,
        metrics=metrics
    )


def process_data(
    input_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    progress: Optional[Progress] = None
) -> Dict[str, Any]:
    """
    Process raw data with sliding windows.
    
    Returns statistics about processing.
    """
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    window_size = config.get('window_size', 100)
    stride = config.get('stride', 50)
    
    stats = {
        'total_windows': 0,
        'valid_windows': 0,
        'nan_events': 0,
        'outliers': 0,
        'time': 0
    }
    
    # Process each data file
    data_files = list(input_dir.glob("*.npy")) + list(input_dir.glob("*.npz"))
    
    all_windows = []
    all_labels = []
    
    for file_path in data_files:
        # Load data
        data = np.load(file_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if 'data' in data:
                signals = data['data']
                labels = data.get('labels', None)
            else:
                signals = data['signals']
                labels = data.get('labels', None)
        else:
            signals = data
            labels = None
        
        # Extract windows
        n_samples = signals.shape[0] if len(signals.shape) > 1 else len(signals)
        
        for i in range(0, n_samples - window_size + 1, stride):
            window = signals[i:i + window_size]
            
            # Quality checks
            if np.isnan(window).any():
                stats['nan_events'] += 1
                # Interpolate NaNs
                window = interpolate_nans(window)
            
            # Check for outliers
            outliers = detect_outliers(window)
            if outliers.any():
                stats['outliers'] += outliers.sum()
                # Clip outliers
                window = clip_outliers(window)
            
            all_windows.append(window)
            stats['total_windows'] += 1
            stats['valid_windows'] += 1
            
            if labels is not None:
                if len(labels.shape) > 1:
                    all_labels.append(labels[i:i + window_size])
                else:
                    all_labels.append(labels[i])
    
    # Save processed data
    output_file = output_dir / "processed_data.npz"
    save_data = {'windows': np.array(all_windows)}
    if all_labels:
        save_data['labels'] = np.array(all_labels)
    
    np.savez_compressed(output_file, **save_data)
    
    stats['time'] = time.time() - start_time
    return stats


def interpolate_nans(data: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values."""
    result = data.copy()
    
    if len(data.shape) == 1:
        nans = np.isnan(data)
        if nans.any():
            x = np.arange(len(data))
            result[nans] = np.interp(x[nans], x[~nans], data[~nans])
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1] if len(data.shape) > 1 else 1):
                if len(data.shape) > 1:
                    signal = data[i, j]
                else:
                    signal = data[i]
                
                nans = np.isnan(signal)
                if nans.any() and (~nans).any():
                    x = np.arange(len(signal))
                    if len(data.shape) > 1:
                        result[i, j, nans] = np.interp(
                            x[nans], x[~nans], signal[~nans]
                        )
                    else:
                        result[i][nans] = np.interp(
                            x[nans], x[~nans], signal[~nans]
                        )
    
    return result


def detect_outliers(data: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Detect outliers using MAD-based z-scores."""
    flat_data = data.flatten()
    median = np.median(flat_data[~np.isnan(flat_data)])
    mad = np.median(np.abs(flat_data[~np.isnan(flat_data)] - median))
    
    if mad > 0:
        z_scores = np.abs((flat_data - median) / (1.4826 * mad))
        outliers = z_scores > threshold
    else:
        outliers = np.zeros_like(flat_data, dtype=bool)
    
    return outliers.reshape(data.shape)


def clip_outliers(data: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Clip outliers to threshold values."""
    result = data.copy()
    outliers = detect_outliers(data, threshold)
    
    if outliers.any():
        flat_data = result.flatten()
        median = np.median(flat_data[~np.isnan(flat_data)])
        mad = np.median(np.abs(flat_data[~np.isnan(flat_data)] - median))
        
        if mad > 0:
            lower = median - threshold * 1.4826 * mad
            upper = median + threshold * 1.4826 * mad
            flat_data[outliers.flatten()] = np.clip(
                flat_data[outliers.flatten()], lower, upper
            )
            result = flat_data.reshape(data.shape)
    
    return result