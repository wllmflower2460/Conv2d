"""Median filter with hysteresis for temporal smoothing.

Implements a sliding median filter with configurable hysteresis
thresholds and minimum dwell time enforcement.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.stats import mode

from conv2d.temporal.interface import TemporalConfig, TemporalPolicy

logger = logging.getLogger(__name__)


class MedianHysteresisPolicy(TemporalPolicy):
    """Median filtering with hysteresis for stable motifs.
    
    Combines:
    - Sliding median/mode filter for local consensus
    - Hysteresis thresholds for transition control
    - Minimum dwell time enforcement
    """
    
    def __init__(self, config: TemporalConfig):
        """Initialize median hysteresis policy.
        
        Args:
            config: Temporal configuration
        """
        super().__init__(config)
        
        # Validate median-specific params
        if config.window_size % 2 == 0:
            logger.warning(
                f"Even window size {config.window_size} -> {config.window_size + 1} (odd is better)"
            )
            self.window_size = config.window_size + 1
        else:
            self.window_size = config.window_size
    
    def smooth_sequence(
        self, 
        motifs: NDArray[np.int32],
        confidences: Optional[NDArray[np.float32]] = None,
    ) -> NDArray[np.int32]:
        """Apply median filtering with hysteresis.
        
        Args:
            motifs: Raw motif predictions (B, T) or (T,)
            confidences: Optional confidence scores
            
        Returns:
            Smoothed motif sequence
        """
        # Handle batch dimension
        squeeze_output = False
        if motifs.ndim == 1:
            motifs = motifs[np.newaxis, :]
            squeeze_output = True
            if confidences is not None and confidences.ndim == 1:
                confidences = confidences[np.newaxis, :]
        
        B, T = motifs.shape
        smoothed = np.zeros_like(motifs)
        
        for b in range(B):
            # Reset state for each sequence
            self.reset()
            
            # Get sequence and confidences
            seq = motifs[b]
            conf = confidences[b] if confidences is not None else None
            
            # Step 1: Apply median filter
            seq_median = self._apply_median_filter(seq)
            
            # Step 2: Apply hysteresis if confidences provided
            if conf is not None:
                seq_hysteresis = self._apply_hysteresis(seq_median, conf)
            else:
                seq_hysteresis = seq_median
            
            # Step 3: Enforce minimum dwell time
            seq_final = self._enforce_min_dwell(seq_hysteresis)
            
            smoothed[b] = seq_final
        
        # Log statistics
        if logger.isEnabledFor(logging.DEBUG):
            changes_before = np.sum(np.diff(motifs, axis=1) != 0)
            changes_after = np.sum(np.diff(smoothed, axis=1) != 0)
            logger.debug(
                f"MedianHysteresis: transitions {changes_before} → {changes_after} "
                f"(reduced by {(1 - changes_after/max(changes_before, 1)):.1%})"
            )
        
        if squeeze_output:
            smoothed = smoothed.squeeze(0)
        
        return smoothed
    
    def _apply_median_filter(
        self, 
        motifs: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        """Apply sliding median filter to motif sequence.
        
        For discrete labels, this becomes a mode filter.
        
        Args:
            motifs: Motif sequence (T,)
            
        Returns:
            Filtered sequence
        """
        T = len(motifs)
        filtered = motifs.copy()
        half_window = self.window_size // 2
        
        for t in range(T):
            # Get window bounds
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            
            # Get mode in window (most common motif)
            window = motifs[start:end]
            if len(window) > 0:
                mode_result = mode(window, keepdims=False)
                # Handle scipy version differences
                if hasattr(mode_result, 'mode'):
                    filtered[t] = mode_result.mode
                else:
                    filtered[t] = mode_result[0]
        
        return filtered
    
    def _apply_weighted_voting(
        self,
        motifs: NDArray[np.int32],
        confidences: NDArray[np.float32],
    ) -> NDArray[np.int32]:
        """Apply weighted voting in sliding window.
        
        Args:
            motifs: Motif sequence (T,)
            confidences: Confidence scores (T, K)
            
        Returns:
            Smoothed sequence using confidence-weighted voting
        """
        T = len(motifs)
        K = confidences.shape[1] if confidences.ndim == 2 else 1
        smoothed = motifs.copy()
        half_window = self.window_size // 2
        
        for t in range(T):
            # Get window bounds
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            
            # Accumulate weighted votes
            votes = np.zeros(K, dtype=np.float32)
            
            for i in range(start, end):
                if confidences.ndim == 2:
                    # Add confidence-weighted vote
                    votes += confidences[i]
                else:
                    # Binary vote
                    votes[motifs[i]] += 1.0
            
            # Select motif with highest weighted vote
            smoothed[t] = np.argmax(votes)
        
        return smoothed
    
    def get_statistics(
        self,
        original: NDArray[np.int32],
        smoothed: NDArray[np.int32],
    ) -> Dict[str, float]:
        """Compute smoothing statistics.
        
        Args:
            original: Original motif sequence
            smoothed: Smoothed motif sequence
            
        Returns:
            Dictionary of statistics
        """
        # Flatten if needed
        if original.ndim > 1:
            original = original.flatten()
        if smoothed.ndim > 1:
            smoothed = smoothed.flatten()
        
        # Count transitions
        trans_orig = np.sum(np.diff(original) != 0)
        trans_smooth = np.sum(np.diff(smoothed) != 0)
        
        # Compute dwell times
        dwell_orig = self._compute_dwell_times(original)
        dwell_smooth = self._compute_dwell_times(smoothed)
        
        return {
            "transitions_original": int(trans_orig),
            "transitions_smoothed": int(trans_smooth),
            "transition_reduction": float(1 - trans_smooth / max(trans_orig, 1)),
            "mean_dwell_original": float(np.mean(dwell_orig)) if len(dwell_orig) > 0 else 0,
            "mean_dwell_smoothed": float(np.mean(dwell_smooth)) if len(dwell_smooth) > 0 else 0,
            "min_dwell_smoothed": int(np.min(dwell_smooth)) if len(dwell_smooth) > 0 else 0,
            "flicker_removed": int(np.sum(dwell_orig < self.config.min_dwell)),
        }
    
    def _compute_dwell_times(
        self, 
        motifs: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        """Compute dwell times for each motif segment.
        
        Args:
            motifs: Motif sequence
            
        Returns:
            Array of dwell times
        """
        if len(motifs) == 0:
            return np.array([], dtype=np.int32)
        
        # Find transition points
        transitions = np.where(np.diff(motifs) != 0)[0] + 1
        transitions = np.concatenate([[0], transitions, [len(motifs)]])
        
        # Compute segment lengths
        dwell_times = np.diff(transitions)
        
        return dwell_times


from typing import Dict