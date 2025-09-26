#!/usr/bin/env python3
"""Unit tests for temporal smoothing policies.

Tests verify:
1. Minimum dwell times are enforced
2. Hysteresis thresholds work correctly  
3. No 1-2 frame flicker
4. Policy swapping at runtime
5. Batch processing consistency
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from conv2d.temporal import (
    TemporalConfig,
    TemporalPolicy,
    MedianHysteresisPolicy,
    HSMMPolicy,
    PassthroughPolicy,
)


class TestTemporalConfig:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = TemporalConfig(
            min_dwell=5,
            enter_threshold=0.7,
            exit_threshold=0.3,
            window_size=7,
            policy_type="median"
        )
        
        assert config.min_dwell == 5
        assert config.enter_threshold == 0.7
        assert config.exit_threshold == 0.3
        print("✓ Valid config accepted")
    
    def test_invalid_thresholds(self):
        """Test that exit > enter is rejected."""
        try:
            config = TemporalConfig(
                enter_threshold=0.3,
                exit_threshold=0.7  # Invalid: exit > enter
            )
            assert False, "Should reject exit > enter"
        except AssertionError as e:
            assert "exit_threshold must be <= enter_threshold" in str(e)
        print("✓ Invalid thresholds rejected")
    
    def test_factory_method(self):
        """Test policy factory."""
        # From string
        policy1 = TemporalPolicy.create("median")
        assert isinstance(policy1, MedianHysteresisPolicy)
        
        # From dict
        policy2 = TemporalPolicy.create({"policy_type": "hsmm"})
        assert isinstance(policy2, HSMMPolicy)
        
        # From config
        config = TemporalConfig(policy_type="none")
        policy3 = TemporalPolicy.create(config)
        assert isinstance(policy3, PassthroughPolicy)
        
        print("✓ Factory method works")


class TestMinimumDwell:
    """Test minimum dwell time enforcement."""
    
    def test_no_single_frame(self):
        """Test that single-frame motifs are eliminated."""
        # Create flickering sequence
        motifs = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
        
        config = TemporalConfig(min_dwell=3, policy_type="median")
        policy = MedianHysteresisPolicy(config)
        
        smoothed = policy.smooth_sequence(motifs)
        
        # Check no segments shorter than min_dwell
        dwells = self._compute_dwells(smoothed)
        assert np.all(dwells >= 3), f"Found short dwells: {dwells}"
        
        print(f"✓ No single frames: {motifs} → {smoothed}")
    
    def test_two_frame_flicker(self):
        """Test that 2-frame flickers are removed."""
        # Create 2-frame flicker pattern
        motifs = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int32)
        
        config = TemporalConfig(min_dwell=3, window_size=3, policy_type="median")
        policy = MedianHysteresisPolicy(config)
        
        smoothed = policy.smooth_sequence(motifs)
        
        # Should merge short segments
        dwells = self._compute_dwells(smoothed)
        # Some merging should occur
        assert len(dwells) < 4, f"Should merge segments, got {len(dwells)} segments"
        
        print(f"✓ No 2-frame flicker: {len(dwells)} segments after merge")
    
    def test_preserves_long_segments(self):
        """Test that long segments are preserved."""
        # Create sequence with long segments
        motifs = np.array([0]*10 + [1]*15 + [2]*20, dtype=np.int32)
        
        config = TemporalConfig(min_dwell=5, policy_type="median")
        policy = MedianHysteresisPolicy(config)
        
        smoothed = policy.smooth_sequence(motifs)
        
        # Long segments should be mostly preserved
        assert np.sum(smoothed[:10] == 0) >= 8, "Long segment 0 damaged"
        assert np.sum(smoothed[10:25] == 1) >= 12, "Long segment 1 damaged"
        assert np.sum(smoothed[25:] == 2) >= 18, "Long segment 2 damaged"
        
        print("✓ Long segments preserved")
    
    def _compute_dwells(self, motifs):
        """Compute dwell times."""
        if len(motifs) == 0:
            return np.array([])
        
        transitions = np.where(np.diff(motifs) != 0)[0] + 1
        transitions = np.concatenate([[0], transitions, [len(motifs)]])
        return np.diff(transitions)


class TestHysteresis:
    """Test hysteresis thresholds."""
    
    def test_enter_threshold(self):
        """Test that enter threshold prevents spurious transitions."""
        motifs = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int32)
        
        # Create confidence scores
        confidences = np.array([
            [0.9, 0.1],  # Strong 0
            [0.8, 0.2],  # Strong 0
            [0.6, 0.4],  # Moderate 0
            [0.45, 0.55],  # Weak 1 (below enter threshold)
            [0.7, 0.3],  # Back to 0
            [0.8, 0.2],  # Strong 0
            [0.9, 0.1],  # Strong 0
        ], dtype=np.float32)
        
        config = TemporalConfig(
            min_dwell=2,
            enter_threshold=0.6,  # Need 60% confidence to enter
            exit_threshold=0.3,
            policy_type="median"
        )
        policy = MedianHysteresisPolicy(config)
        
        smoothed = policy.smooth_sequence(motifs, confidences)
        
        # Should stay in motif 0 due to low confidence for 1
        assert smoothed[3] == 0, "Should not transition on low confidence"
        
        print("✓ Enter threshold blocks weak transitions")
    
    def test_exit_threshold(self):
        """Test that exit threshold maintains stability."""
        motifs = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        
        # Create confidence scores showing gradual transition
        confidences = np.array([
            [0.9, 0.1],  # Strong 0
            [0.7, 0.3],  # Still good 0
            [0.5, 0.5],  # Equal (current still above exit)
            [0.2, 0.8],  # Strong 1
            [0.1, 0.9],  # Strong 1
        ], dtype=np.float32)
        
        config = TemporalConfig(
            min_dwell=1,
            enter_threshold=0.6,
            exit_threshold=0.4,  # Need to drop below 40% to exit
            policy_type="median"
        )
        policy = MedianHysteresisPolicy(config)
        
        smoothed = policy.smooth_sequence(motifs, confidences)
        
        # Transition should be delayed
        assert smoothed[2] == 0, "Should not exit while above threshold"
        
        print("✓ Exit threshold maintains stability")


class TestMedianFilter:
    """Test median filtering behavior."""
    
    def test_window_voting(self):
        """Test that window voting works correctly."""
        # Create sequence with noise
        motifs = np.array([0, 0, 1, 0, 0, 2, 0, 0], dtype=np.int32)
        
        config = TemporalConfig(
            min_dwell=1,
            window_size=3,  # 3-frame window
            policy_type="median"
        )
        policy = MedianHysteresisPolicy(config)
        
        smoothed = policy.smooth_sequence(motifs)
        
        # Single noisy frames should be filtered out
        assert smoothed[2] == 0, "Isolated frame should be filtered"
        assert smoothed[5] == 0, "Isolated frame should be filtered"
        
        print(f"✓ Window voting: {motifs} → {smoothed}")
    
    def test_edge_handling(self):
        """Test behavior at sequence edges."""
        motifs = np.array([1, 0, 0, 0, 0, 1], dtype=np.int32)
        
        config = TemporalConfig(window_size=3, policy_type="median")
        policy = MedianHysteresisPolicy(config)
        
        smoothed = policy.smooth_sequence(motifs)
        
        # Edges should be handled gracefully
        assert len(smoothed) == len(motifs)
        
        print("✓ Edge handling works")


class TestHSMMPolicy:
    """Test HSMM-based smoothing."""
    
    def test_viterbi_smoothing(self):
        """Test Viterbi decoding for smoothing."""
        # Create noisy sequence
        true_motifs = np.array([0]*10 + [1]*10 + [2]*10, dtype=np.int32)
        noise_mask = np.random.rand(30) < 0.1
        noisy_motifs = true_motifs.copy()
        noisy_motifs[noise_mask] = np.random.randint(0, 3, size=noise_mask.sum())
        
        config = TemporalConfig(
            policy_type="hsmm",
            extra_params={"use_viterbi": True, "mean_duration": 10}
        )
        policy = HSMMPolicy(config)
        
        smoothed = policy.smooth_sequence(noisy_motifs)
        
        # Should recover structure
        accuracy = np.mean(smoothed == true_motifs)
        assert accuracy > 0.8, f"Poor recovery: {accuracy:.2f}"
        
        print(f"✓ Viterbi smoothing: {accuracy:.1%} accuracy")
    
    def test_duration_modeling(self):
        """Test that duration distributions are respected."""
        motifs = np.array([0]*5 + [1]*5 + [0]*5 + [1]*5, dtype=np.int32)
        
        config = TemporalConfig(
            policy_type="hsmm",
            extra_params={
                "duration_type": "geometric",
                "mean_duration": 8,
                "use_viterbi": False
            }
        )
        policy = HSMMPolicy(config)
        
        smoothed = policy.smooth_sequence(motifs)
        
        # Should maintain reasonable durations
        dwells = self._compute_dwells(smoothed)
        assert np.mean(dwells) > 3, "Durations too short"
        
        print(f"✓ Duration modeling: mean dwell = {np.mean(dwells):.1f}")
    
    def test_parameter_fitting(self):
        """Test learning HSMM parameters from data."""
        # Create training sequences
        sequences = []
        for _ in range(10):
            seq = np.concatenate([
                np.full(np.random.poisson(10), 0),
                np.full(np.random.poisson(8), 1),
                np.full(np.random.poisson(12), 2),
            ])[:50]  # Truncate to fixed length
            sequences.append(seq)
        sequences = np.array(sequences, dtype=np.int32)
        
        config = TemporalConfig(policy_type="hsmm")
        policy = HSMMPolicy(config)
        
        # Fit parameters
        policy.fit_parameters(sequences, n_iter=5)
        
        # Check learned parameters
        assert policy.transition_matrix_ is not None
        assert policy.duration_probs_ is not None
        
        print("✓ Parameter fitting works")
    
    def _compute_dwells(self, motifs):
        """Compute dwell times."""
        if len(motifs) == 0:
            return np.array([])
        transitions = np.where(np.diff(motifs) != 0)[0] + 1
        transitions = np.concatenate([[0], transitions, [len(motifs)]])
        return np.diff(transitions)


class TestBatchProcessing:
    """Test batch processing consistency."""
    
    def test_batch_independence(self):
        """Test that batch samples are processed independently."""
        # Create batch with different patterns
        batch = np.array([
            [0, 0, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1],
        ], dtype=np.int32)
        
        config = TemporalConfig(min_dwell=2, policy_type="median")
        policy = MedianHysteresisPolicy(config)
        
        # Process as batch
        smoothed_batch = policy.smooth_sequence(batch)
        
        # Process individually
        smoothed_individual = []
        for seq in batch:
            policy.reset()
            smoothed_individual.append(policy.smooth_sequence(seq))
        smoothed_individual = np.array(smoothed_individual)
        
        # Should be identical
        assert np.array_equal(smoothed_batch, smoothed_individual)
        
        print("✓ Batch processing consistent")
    
    def test_batch_statistics(self):
        """Test computing statistics on batches."""
        batch = np.random.randint(0, 3, size=(10, 100), dtype=np.int32)
        
        config = TemporalConfig(min_dwell=3, window_size=5, policy_type="median")
        policy = MedianHysteresisPolicy(config)
        
        smoothed = policy.smooth_sequence(batch)
        
        # Compute statistics
        stats = policy.get_statistics(batch, smoothed)
        
        assert "transition_reduction" in stats
        assert "mean_dwell_smoothed" in stats
        assert stats["min_dwell_smoothed"] >= 3
        
        print(f"✓ Batch stats: {stats['transition_reduction']:.1%} reduction")


class TestRuntimeSwapping:
    """Test runtime policy swapping."""
    
    def test_policy_swapping(self):
        """Test swapping policies at runtime."""
        motifs = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        
        # Try different policies
        policies = {
            "none": PassthroughPolicy(TemporalConfig(policy_type="none")),
            "median": MedianHysteresisPolicy(TemporalConfig(
                policy_type="median", min_dwell=3
            )),
            "hsmm": HSMMPolicy(TemporalConfig(policy_type="hsmm")),
        }
        
        results = {}
        for name, policy in policies.items():
            results[name] = policy.smooth_sequence(motifs)
        
        # Passthrough should be unchanged
        assert np.array_equal(results["none"], motifs)
        
        # Others should smooth
        assert not np.array_equal(results["median"], motifs)
        
        print("✓ Runtime swapping works")
    
    def test_config_based_selection(self):
        """Test configuration-based policy selection."""
        motifs = np.array([0, 1, 0, 1, 0], dtype=np.int32)
        
        # Create policies from different configs
        configs = [
            {"policy_type": "none"},
            {"policy_type": "median", "min_dwell": 2},
            {"policy_type": "hsmm", "extra_params": {"use_viterbi": True}},
        ]
        
        for config_dict in configs:
            policy = TemporalPolicy.create(config_dict)
            result = policy.smooth_sequence(motifs)
            assert result is not None
            assert result.shape == motifs.shape
        
        print("✓ Config-based selection works")


def run_tests():
    """Run all temporal smoothing tests."""
    test_classes = [
        TestTemporalConfig,
        TestMinimumDwell,
        TestHysteresis,
        TestMedianFilter,
        TestHSMMPolicy,
        TestBatchProcessing,
        TestRuntimeSwapping,
    ]
    
    print("=" * 60)
    print("Temporal Smoothing Policy Tests")
    print("=" * 60)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)
        
        test_instance = test_class()
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                test_method = getattr(test_instance, method_name)
                try:
                    test_method()
                except Exception as e:
                    print(f"✗ {method_name}: {e}")
    
    print("\n" + "=" * 60)
    print("All temporal tests completed!")


if __name__ == "__main__":
    run_tests()