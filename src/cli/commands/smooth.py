"""Temporal smoothing command implementation."""

from pathlib import Path
from typing import Dict, Any
import pickle
import numpy as np
from scipy.signal import medfilt


def apply_smoothing(
    clusters_dir: Path,
    window: int,
    min_duration: int
) -> Dict[str, Any]:
    """Apply temporal smoothing to behavioral sequences."""
    # Load cluster labels
    labels_path = clusters_dir / "labels.npy"
    labels = np.load(labels_path)
    
    # Count transitions before
    transitions_before = np.sum(np.diff(labels) != 0)
    
    # Apply median filter
    smoothed = medfilt(labels, kernel_size=window)
    
    # Enforce minimum duration
    smoothed = enforce_min_duration(smoothed, min_duration)
    
    # Count transitions after
    transitions_after = np.sum(np.diff(smoothed) != 0)
    
    # Calculate durations
    duration_before = calculate_mean_duration(labels)
    duration_after = calculate_mean_duration(smoothed)
    
    # Estimate noise events
    noise_before = count_noise_events(labels, min_duration)
    noise_after = count_noise_events(smoothed, min_duration)
    
    return {
        'labels_original': labels,
        'labels_smoothed': smoothed,
        'transitions_before': int(transitions_before),
        'transitions_after': int(transitions_after),
        'duration_before': duration_before,
        'duration_after': duration_after,
        'noise_before': noise_before,
        'noise_after': noise_after,
        'reduction': (transitions_before - transitions_after) / transitions_before
    }


def enforce_min_duration(labels: np.ndarray, min_duration: int) -> np.ndarray:
    """Enforce minimum duration for each motif."""
    result = labels.copy()
    
    # Find segments
    changes = np.where(np.diff(labels) != 0)[0] + 1
    segments = np.split(np.arange(len(labels)), changes)
    
    for segment in segments:
        if len(segment) < min_duration and len(segment) > 0:
            # Replace short segments with neighboring label
            if segment[0] > 0:
                result[segment] = result[segment[0] - 1]
            elif segment[-1] < len(result) - 1:
                result[segment] = result[segment[-1] + 1]
    
    return result


def calculate_mean_duration(labels: np.ndarray) -> float:
    """Calculate mean duration of motifs."""
    changes = np.where(np.diff(labels) != 0)[0] + 1
    segments = np.split(np.arange(len(labels)), changes)
    durations = [len(s) for s in segments if len(s) > 0]
    return np.mean(durations) if durations else 0


def count_noise_events(labels: np.ndarray, min_duration: int) -> int:
    """Count noisy short-duration events."""
    changes = np.where(np.diff(labels) != 0)[0] + 1
    segments = np.split(np.arange(len(labels)), changes)
    return sum(1 for s in segments if 0 < len(s) < min_duration)


def save_smoothed(output_dir: Path, results: Dict[str, Any]):
    """Save smoothed results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "labels_smoothed.npy", results['labels_smoothed'])
    
    with open(output_dir / "smoothing_stats.pkl", 'wb') as f:
        stats = {k: v for k, v in results.items() 
                if k not in ['labels_original', 'labels_smoothed']}
        pickle.dump(stats, f)