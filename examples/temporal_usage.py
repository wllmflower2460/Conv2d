#!/usr/bin/env python3
"""Examples demonstrating temporal smoothing policies.

This module shows how to:
1. Apply median hysteresis smoothing
2. Use HSMM-based temporal models
3. Configure policies at runtime
4. Prevent motif flickering
5. Enforce minimum behavioral durations
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from conv2d.temporal import (
    TemporalConfig,
    TemporalPolicy,
    MedianHysteresisPolicy,
    HSMMPolicy,
    PassthroughPolicy,
)


def visualize_smoothing(original, smoothed, title="Temporal Smoothing"):
    """Visualize before and after smoothing."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    
    # Original
    ax1.plot(original, 'o-', alpha=0.7, label='Original')
    ax1.set_ylabel('Motif ID')
    ax1.set_title(f'{title} - Original')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Smoothed
    ax2.plot(smoothed, 's-', alpha=0.7, color='green', label='Smoothed')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Motif ID')
    ax2.set_title(f'{title} - Smoothed')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def example_flickering_removal():
    """Example 1: Remove 1-2 frame flickering."""
    print("=" * 60)
    print("Example 1: Flickering Removal")
    print("=" * 60)
    
    # Create sequence with annoying flickers
    np.random.seed(42)
    base = np.array([0]*20 + [1]*20 + [2]*20)
    
    # Add random single-frame flickers
    flicker_positions = [5, 15, 25, 35, 45, 55]
    flickered = base.copy()
    for pos in flicker_positions:
        if pos < len(flickered):
            flickered[pos] = (flickered[pos] + 1) % 3
    
    print(f"Original: {flickered}")
    print(f"  Transitions: {np.sum(np.diff(flickered) != 0)}")
    
    # Apply median hysteresis
    config = TemporalConfig(
        min_dwell=3,
        window_size=5,
        policy_type="median"
    )
    policy = MedianHysteresisPolicy(config)
    
    smoothed = policy.smooth_sequence(flickered)
    
    print(f"\nSmoothed: {smoothed}")
    print(f"  Transitions: {np.sum(np.diff(smoothed) != 0)}")
    
    # Get statistics
    stats = policy.get_statistics(flickered, smoothed)
    print(f"\nStatistics:")
    print(f"  Transition reduction: {stats['transition_reduction']:.1%}")
    print(f"  Mean dwell: {stats['mean_dwell_original']:.1f} → {stats['mean_dwell_smoothed']:.1f}")
    print(f"  Flickers removed: {stats['flicker_removed']}")
    
    print()


def example_hysteresis_control():
    """Example 2: Hysteresis threshold control."""
    print("=" * 60)
    print("Example 2: Hysteresis Control")
    print("=" * 60)
    
    # Create uncertain transitions
    T = 100
    motifs = np.zeros(T, dtype=np.int32)
    motifs[30:70] = 1  # Middle section different
    
    # Add noise near boundaries
    noise_positions = list(range(25, 35)) + list(range(65, 75))
    for pos in noise_positions:
        if np.random.rand() > 0.5:
            motifs[pos] = 1 - motifs[pos]
    
    # Create confidence scores (low near transitions)
    confidences = np.ones((T, 2), dtype=np.float32) * 0.9
    for t in range(T):
        if 25 <= t <= 35 or 65 <= t <= 75:
            # Low confidence near transitions
            confidences[t, motifs[t]] = 0.55
            confidences[t, 1-motifs[t]] = 0.45
    
    print(f"Noisy input with {np.sum(np.diff(motifs) != 0)} transitions")
    
    # Apply different hysteresis settings
    configs = [
        ("No hysteresis", TemporalConfig(
            enter_threshold=0.5,
            exit_threshold=0.5,
            min_dwell=1,
            policy_type="median"
        )),
        ("Moderate hysteresis", TemporalConfig(
            enter_threshold=0.6,
            exit_threshold=0.4,
            min_dwell=3,
            policy_type="median"
        )),
        ("Strong hysteresis", TemporalConfig(
            enter_threshold=0.7,
            exit_threshold=0.3,
            min_dwell=5,
            policy_type="median"
        )),
    ]
    
    for name, config in configs:
        policy = MedianHysteresisPolicy(config)
        smoothed = policy.smooth_sequence(motifs, confidences)
        transitions = np.sum(np.diff(smoothed) != 0)
        print(f"  {name:20s}: {transitions} transitions")
    
    print()


def example_hsmm_smoothing():
    """Example 3: HSMM-based temporal smoothing."""
    print("=" * 60)
    print("Example 3: HSMM Temporal Model")
    print("=" * 60)
    
    # Create realistic behavioral sequence
    np.random.seed(42)
    
    # Generate with realistic durations
    durations = [
        np.random.poisson(15),  # Motif 0
        np.random.poisson(10),  # Motif 1
        np.random.poisson(20),  # Motif 2
        np.random.poisson(8),   # Motif 1
        np.random.poisson(12),  # Motif 0
    ]
    
    sequence = []
    motif_order = [0, 1, 2, 1, 0]
    for motif, duration in zip(motif_order, durations):
        sequence.extend([motif] * duration)
    
    # Add 10% noise
    sequence = np.array(sequence, dtype=np.int32)
    noise_mask = np.random.rand(len(sequence)) < 0.1
    sequence[noise_mask] = np.random.randint(0, 3, noise_mask.sum())
    
    print(f"Noisy sequence: length={len(sequence)}, noise=10%")
    
    # Apply HSMM smoothing
    config = TemporalConfig(
        policy_type="hsmm",
        extra_params={
            "duration_type": "poisson",
            "mean_duration": 12,
            "use_viterbi": True,
        }
    )
    policy = HSMMPolicy(config)
    
    smoothed = policy.smooth_sequence(sequence)
    
    # Compare
    transitions_before = np.sum(np.diff(sequence) != 0)
    transitions_after = np.sum(np.diff(smoothed) != 0)
    
    print(f"Transitions: {transitions_before} → {transitions_after}")
    print(f"Reduction: {(1 - transitions_after/transitions_before):.1%}")
    
    # Extract durations
    durations_smoothed = []
    if len(smoothed) > 0:
        current = smoothed[0]
        count = 1
        for i in range(1, len(smoothed)):
            if smoothed[i] == current:
                count += 1
            else:
                durations_smoothed.append(count)
                current = smoothed[i]
                count = 1
        durations_smoothed.append(count)
    
    print(f"Smoothed durations: {durations_smoothed}")
    print()


def example_runtime_switching():
    """Example 4: Runtime policy switching."""
    print("=" * 60)
    print("Example 4: Runtime Policy Switching")
    print("=" * 60)
    
    # Create test sequence
    sequence = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5, dtype=np.int32)
    
    # Define policies to compare
    policy_configs = {
        "none": {
            "policy_type": "none"
        },
        "median_weak": {
            "policy_type": "median",
            "min_dwell": 2,
            "window_size": 3,
        },
        "median_strong": {
            "policy_type": "median",
            "min_dwell": 5,
            "window_size": 7,
        },
        "hsmm": {
            "policy_type": "hsmm",
            "extra_params": {
                "mean_duration": 5,
                "use_viterbi": True,
            }
        },
    }
    
    print(f"Input sequence: {len(sequence)} frames with rapid alternation")
    print()
    
    # Apply each policy
    for name, config_dict in policy_configs.items():
        # Create policy from config
        policy = TemporalPolicy.create(config_dict)
        
        # Apply smoothing
        smoothed = policy.smooth_sequence(sequence)
        
        # Compute metrics
        transitions = np.sum(np.diff(smoothed) != 0)
        unique = len(np.unique(smoothed))
        
        print(f"{name:15s}: {transitions:3d} transitions, {unique} unique motifs")
    
    print("\n✓ Policies swappable at runtime via config")
    print()


def example_batch_processing():
    """Example 5: Batch processing of multiple sequences."""
    print("=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)
    
    # Create batch of sequences with different patterns
    batch = np.array([
        [0, 0, 0, 1, 1, 1, 0, 0, 0],  # Clean transitions
        [0, 1, 0, 1, 0, 1, 0, 1, 0],  # Rapid flickering
        [0, 0, 2, 0, 0, 1, 1, 1, 1],  # Single noise spike
        [0, 1, 1, 2, 2, 2, 1, 1, 0],  # Multiple transitions
    ], dtype=np.int32)
    
    print(f"Batch shape: {batch.shape}")
    
    # Apply smoothing to entire batch
    config = TemporalConfig(
        policy_type="median",
        min_dwell=3,
        window_size=5,
    )
    policy = MedianHysteresisPolicy(config)
    
    smoothed_batch = policy.smooth_sequence(batch)
    
    # Analyze each sequence
    for i in range(len(batch)):
        trans_before = np.sum(np.diff(batch[i]) != 0)
        trans_after = np.sum(np.diff(smoothed_batch[i]) != 0)
        reduction = (1 - trans_after / max(trans_before, 1)) * 100
        
        print(f"  Sequence {i+1}: {trans_before} → {trans_after} transitions ({reduction:.0f}% reduction)")
    
    print()


def example_confidence_weighted():
    """Example 6: Confidence-weighted smoothing."""
    print("=" * 60)
    print("Example 6: Confidence-Weighted Smoothing")
    print("=" * 60)
    
    # Create sequence with varying confidence
    T = 50
    motifs = np.zeros(T, dtype=np.int32)
    motifs[15:35] = 1
    
    # Add uncertain predictions
    uncertain_regions = [(10, 20), (30, 40)]
    for start, end in uncertain_regions:
        for t in range(start, min(end, T)):
            if np.random.rand() > 0.6:
                motifs[t] = 1 - motifs[t]
    
    # Create confidence scores
    K = 2  # Number of motifs
    confidences = np.zeros((T, K), dtype=np.float32)
    
    for t in range(T):
        # High confidence in stable regions
        if any(start <= t < end for start, end in uncertain_regions):
            # Low confidence in uncertain regions
            confidences[t, motifs[t]] = 0.55
            confidences[t, 1-motifs[t]] = 0.45
        else:
            # High confidence elsewhere
            confidences[t, motifs[t]] = 0.95
            confidences[t, 1-motifs[t]] = 0.05
    
    print("Input with uncertain regions:")
    print(f"  Motifs: {motifs}")
    
    # Apply smoothing without confidence
    policy_no_conf = MedianHysteresisPolicy(
        TemporalConfig(min_dwell=3, window_size=5)
    )
    smoothed_no_conf = policy_no_conf.smooth_sequence(motifs)
    
    # Apply smoothing with confidence
    policy_with_conf = MedianHysteresisPolicy(
        TemporalConfig(
            min_dwell=3,
            window_size=5,
            enter_threshold=0.7,
            exit_threshold=0.3,
        )
    )
    smoothed_with_conf = policy_with_conf.smooth_sequence(motifs, confidences)
    
    print(f"\nWithout confidence: {np.sum(np.diff(smoothed_no_conf) != 0)} transitions")
    print(f"With confidence:    {np.sum(np.diff(smoothed_with_conf) != 0)} transitions")
    print()


def example_minimum_dwell_enforcement():
    """Example 7: Minimum dwell time enforcement."""
    print("=" * 60)
    print("Example 7: Minimum Dwell Time")
    print("=" * 60)
    
    # Create sequence with various segment lengths
    segments = [
        (0, 1),   # 1 frame - too short
        (1, 2),   # 2 frames - too short
        (0, 3),   # 3 frames - borderline
        (2, 5),   # 5 frames - ok
        (1, 10),  # 10 frames - good
        (0, 1),   # 1 frame - too short
        (2, 15),  # 15 frames - good
    ]
    
    sequence = []
    for motif, duration in segments:
        sequence.extend([motif] * duration)
    sequence = np.array(sequence, dtype=np.int32)
    
    print(f"Original segments: {[d for _, d in segments]}")
    print(f"Total length: {len(sequence)}")
    
    # Apply different minimum dwell times
    for min_dwell in [1, 3, 5]:
        config = TemporalConfig(
            min_dwell=min_dwell,
            policy_type="median",
            window_size=3,
        )
        policy = MedianHysteresisPolicy(config)
        
        smoothed = policy.smooth_sequence(sequence)
        
        # Count actual dwells
        dwells = []
        if len(smoothed) > 0:
            current = smoothed[0]
            count = 1
            for i in range(1, len(smoothed)):
                if smoothed[i] == current:
                    count += 1
                else:
                    dwells.append(count)
                    current = smoothed[i]
                    count = 1
            dwells.append(count)
        
        min_actual = min(dwells) if dwells else 0
        print(f"\nmin_dwell={min_dwell}:")
        print(f"  Segments: {dwells}")
        print(f"  Min actual: {min_actual}")
        assert min_actual >= min_dwell or min_dwell == 1, f"Dwell constraint violated"
    
    print()


def main():
    """Run all temporal smoothing examples."""
    examples = [
        example_flickering_removal,
        example_hysteresis_control,
        example_hsmm_smoothing,
        example_runtime_switching,
        example_batch_processing,
        example_confidence_weighted,
        example_minimum_dwell_enforcement,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example {example_func.__name__} failed: {e}")
            print()


if __name__ == "__main__":
    main()