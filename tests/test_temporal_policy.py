#!/usr/bin/env python3
"""Temporal policy tests - proves min-dwell/hysteresis are respected (no 1-2 frame flicker)."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conv2d.temporal.median import MedianHysteresisPolicy
from conv2d.temporal.interface import TemporalPolicy


class TestTemporalPolicy:
    """Test temporal policies enforce behavioral constraints - critical for realistic behavioral analysis."""
    
    def test_min_dwell_enforcement_strict(self):
        """Min-dwell policy MUST eliminate segments shorter than threshold."""
        
        # Create sequence with known short segments
        labels = np.array([
            [0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 0, 0],  # 1-frame flickers at positions 2, 8
            [1, 2, 1, 1, 2, 2, 3, 3, 3, 1, 2, 1],  # Multiple short segments
        ])
        
        policy = MedianHysteresisPolicy(min_dwell=3, window_size=5)
        smoothed = policy.smooth(labels)
        
        # Check that NO segment is shorter than min_dwell=3
        for batch_idx in range(labels.shape[0]):
            segments = self._find_segments(smoothed[batch_idx])
            
            for start, end, state in segments:
                duration = end - start
                assert duration >= 3, \
                    f"Batch {batch_idx}: Segment {state} at [{start}:{end}] has duration {duration} < 3"
    
    def test_hysteresis_prevents_oscillation(self):
        """Hysteresis thresholds must prevent rapid state oscillation."""
        
        # Create rapidly oscillating sequence
        oscillating = np.tile([0, 1, 0, 1], 25)  # 100 timesteps of oscillation
        labels = oscillating.reshape(1, -1)
        
        policy = MedianHysteresisPolicy(
            min_dwell=1,  # Don't use min-dwell for this test
            window_size=5,
            enter_threshold=0.8,  # High threshold to enter new state
            exit_threshold=0.2,   # Low threshold to exit current state
        )
        
        smoothed = policy.smooth(labels)
        
        # Count transitions before and after
        transitions_before = np.sum(np.diff(labels[0]) != 0)
        transitions_after = np.sum(np.diff(smoothed[0]) != 0)
        
        # Hysteresis should dramatically reduce transitions
        reduction_ratio = transitions_after / transitions_before
        assert reduction_ratio < 0.3, \
            f"Hysteresis failed to prevent oscillation: {reduction_ratio:.1%} transitions remaining"
        
        print(f"Transitions: {transitions_before} → {transitions_after} ({reduction_ratio:.1%})")
    
    def test_no_new_states_introduced(self):
        """Temporal smoothing must NEVER introduce new behavioral states."""
        
        # Create labels with specific states
        np.random.seed(42)
        original_states = [0, 1, 2, 3]
        labels = np.random.choice(original_states, size=(10, 50))
        
        policy = MedianHysteresisPolicy(min_dwell=3, window_size=7)
        smoothed = policy.smooth(labels)
        
        # Check that no new states were introduced
        original_set = set(labels.flatten())
        smoothed_set = set(smoothed.flatten())
        
        assert smoothed_set <= original_set, \
            f"New states introduced: {smoothed_set - original_set}"
        
        print(f"States preserved: {original_set} → {smoothed_set}")
    
    def test_single_frame_flicker_elimination(self):
        """Single-frame flickers (1-frame segments) must be completely eliminated."""
        
        # Create sequence with known single-frame flickers
        base_sequence = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        labels = np.array([base_sequence])
        
        # Inject single-frame flickers
        labels[0, 2] = 3    # Single frame flicker
        labels[0, 6] = 0    # Single frame flicker  
        labels[0, 9] = 1    # Single frame flicker
        
        policy = MedianHysteresisPolicy(min_dwell=2, window_size=3)
        smoothed = policy.smooth(labels)
        
        # Find all segments in smoothed sequence
        segments = self._find_segments(smoothed[0])
        
        # NO segment should have duration 1
        for start, end, state in segments:
            duration = end - start
            assert duration >= 2, \
                f"Single-frame flicker not eliminated: state {state} at [{start}:{end}]"
        
        # Original flicker positions should be fixed
        assert smoothed[0, 2] != 3, "Single-frame flicker at position 2 not fixed"
        assert smoothed[0, 6] != 0, "Single-frame flicker at position 6 not fixed" 
        assert smoothed[0, 9] != 1, "Single-frame flicker at position 9 not fixed"
    
    def test_two_frame_flicker_elimination(self):
        """Two-frame flickers must be eliminated when min_dwell >= 3."""
        
        # Create sequence with 2-frame segments
        labels = np.array([
            [0, 0, 1, 1, 0, 0, 2, 2, 2, 0, 0, 0],  # 2-frame segments of 1 and 2
            [3, 2, 2, 3, 3, 1, 1, 3, 3, 3, 2, 2],  # Multiple 2-frame segments
        ])
        
        policy = MedianHysteresisPolicy(min_dwell=3, window_size=5)
        smoothed = policy.smooth(labels)
        
        # Check that no segment is exactly 2 frames
        for batch_idx in range(labels.shape[0]):
            segments = self._find_segments(smoothed[batch_idx])
            
            for start, end, state in segments:
                duration = end - start
                assert duration >= 3, \
                    f"Two-frame segment not eliminated: state {state} duration {duration}"
    
    def test_temporal_consistency_across_batches(self):
        """Same sequence should produce identical smoothing across different batches."""
        
        # Create identical sequences in different batch positions
        sequence = [0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 0, 0]
        labels = np.array([sequence, sequence, sequence])
        
        policy = MedianHysteresisPolicy(min_dwell=3, window_size=5)
        smoothed = policy.smooth(labels)
        
        # All batches should have identical smoothed sequences
        reference = smoothed[0]
        for batch_idx in range(1, labels.shape[0]):
            assert np.array_equal(reference, smoothed[batch_idx]), \
                f"Batch {batch_idx} differs from reference - inconsistent smoothing"
    
    def test_monotonic_transition_reduction(self):
        """Temporal smoothing should never INCREASE the number of transitions."""
        
        np.random.seed(42)
        # Create noisy sequences
        for _ in range(10):
            labels = np.random.randint(0, 4, size=(5, 100))
            
            policy = MedianHysteresisPolicy(min_dwell=3, window_size=7)
            smoothed = policy.smooth(labels)
            
            for batch_idx in range(labels.shape[0]):
                transitions_before = np.sum(np.diff(labels[batch_idx]) != 0)
                transitions_after = np.sum(np.diff(smoothed[batch_idx]) != 0)
                
                assert transitions_after <= transitions_before, \
                    f"Smoothing increased transitions: {transitions_before} → {transitions_after}"
    
    def test_edge_case_short_sequences(self):
        """Policy should handle sequences shorter than min_dwell gracefully."""
        
        # Very short sequences
        short_labels = np.array([
            [0],          # 1 timestep
            [1, 2],       # 2 timesteps  
            [0, 1, 0],    # 3 timesteps
        ])
        
        policy = MedianHysteresisPolicy(min_dwell=5, window_size=3)  # min_dwell > sequence length
        smoothed = policy.smooth(short_labels)
        
        # Should handle gracefully without crashing
        assert smoothed.shape == short_labels.shape, \
            "Short sequence shape changed"
        
        # For very short sequences, might become constant
        for batch_idx in range(short_labels.shape[0]):
            if short_labels.shape[1] < 5:  # Shorter than min_dwell
                # Should be mostly constant or handle gracefully
                transitions = np.sum(np.diff(smoothed[batch_idx]) != 0)
                assert transitions <= 1, \
                    f"Too many transitions in short sequence: {transitions}"
    
    def test_window_size_effects(self):
        """Different window sizes should affect smoothing behavior predictably."""
        
        # Create sequence with noise
        base = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0]
        noisy = base.copy()
        noisy[2] = 1   # noise
        noisy[5] = 2   # noise
        noisy[8] = 0   # noise
        labels = np.array([noisy])
        
        # Test different window sizes
        window_sizes = [3, 5, 7, 9]
        transition_counts = []
        
        for window_size in window_sizes:
            policy = MedianHysteresisPolicy(min_dwell=2, window_size=window_size)
            smoothed = policy.smooth(labels)
            transitions = np.sum(np.diff(smoothed[0]) != 0)
            transition_counts.append(transitions)
        
        # Larger windows should generally produce fewer transitions
        # (though this isn't guaranteed, it's the expected trend)
        print(f"Window sizes {window_sizes} → Transitions {transition_counts}")
        
        # At minimum, should not crash and should respect min_dwell
        for window_size, transitions in zip(window_sizes, transition_counts):
            policy = MedianHysteresisPolicy(min_dwell=3, window_size=window_size)
            smoothed = policy.smooth(labels)
            segments = self._find_segments(smoothed[0])
            
            for start, end, state in segments:
                duration = end - start
                assert duration >= 3, \
                    f"Window size {window_size}: segment duration {duration} < 3"
    
    def test_hysteresis_threshold_behavior(self):
        """Enter/exit thresholds should create proper hysteresis behavior."""
        
        # Create sequence that alternates between two states
        alternating = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5  # 50 timesteps
        labels = np.array([alternating])
        
        # Test different threshold combinations
        configs = [
            (0.9, 0.1),  # Very strict (hard to enter, easy to exit)
            (0.7, 0.3),  # Moderate
            (0.6, 0.4),  # Mild hysteresis
        ]
        
        for enter_thresh, exit_thresh in configs:
            policy = MedianHysteresisPolicy(
                min_dwell=1,  # Focus on hysteresis, not min_dwell
                window_size=5,
                enter_threshold=enter_thresh,
                exit_threshold=exit_thresh
            )
            
            smoothed = policy.smooth(labels)
            transitions_after = np.sum(np.diff(smoothed[0]) != 0)
            transitions_before = np.sum(np.diff(labels[0]) != 0)
            
            reduction = (transitions_before - transitions_after) / transitions_before
            
            # Higher thresholds should create more reduction
            assert reduction > 0, \
                f"No transition reduction with thresholds ({enter_thresh}, {exit_thresh})"
            
            print(f"Thresholds ({enter_thresh}, {exit_thresh}): {reduction:.1%} reduction")
    
    def test_deterministic_behavior(self):
        """Same input should produce identical output every time."""
        
        np.random.seed(42)
        labels = np.random.randint(0, 4, size=(5, 50))
        
        policy1 = MedianHysteresisPolicy(min_dwell=3, window_size=7)
        policy2 = MedianHysteresisPolicy(min_dwell=3, window_size=7)
        
        smoothed1 = policy1.smooth(labels)
        smoothed2 = policy2.smooth(labels)
        
        assert np.array_equal(smoothed1, smoothed2), \
            "Temporal policy not deterministic"
    
    @staticmethod
    def _find_segments(sequence):
        """Find contiguous segments in sequence."""
        segments = []
        if len(sequence) == 0:
            return segments
        
        start = 0
        current_state = sequence[0]
        
        for i in range(1, len(sequence)):
            if sequence[i] != current_state:
                segments.append((start, i, current_state))
                start = i
                current_state = sequence[i]
        
        # Final segment
        segments.append((start, len(sequence), current_state))
        return segments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])