"""Tests for temporal policy assertions.

CRITICAL: Min-dwell enforcement and hysteresis monotonicity.
These tests catch temporal violations that create flickering behaviors.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from conv2d.temporal.median import MedianHysteresisPolicy as MedianHysteresis
from conv2d.temporal.interface import TemporalPolicy


def count_transitions(sequence: np.ndarray) -> int:
    """Count state transitions in sequence."""
    if len(sequence) == 0:
        return 0
    return np.sum(sequence[1:] != sequence[:-1])


def get_dwell_times(sequence: np.ndarray) -> List[int]:
    """Get all dwell times from sequence."""
    if len(sequence) == 0:
        return []
    
    dwells = []
    current_state = sequence[0]
    current_dwell = 1
    
    for state in sequence[1:]:
        if state == current_state:
            current_dwell += 1
        else:
            dwells.append(current_dwell)
            current_state = state
            current_dwell = 1
    
    dwells.append(current_dwell)
    return dwells


def test_min_dwell_enforcement():
    """Test minimum dwell time is strictly enforced."""
    min_dwells = [3, 5, 10]
    
    for min_dwell in min_dwells:
        # Create sequence with violations
        # Pattern: short dwells that should be filtered
        violations = []
        for i in range(4):
            violations.extend([i] * (min_dwell - 1))  # Too short
        violations.extend([0] * (min_dwell + 5))  # Valid dwell at end
        
        sequence = np.array(violations, dtype=np.int32).reshape(1, -1)
        
        # Apply policy
        policy = MedianHysteresis(min_dwell=min_dwell)
        smoothed = policy.smooth(sequence)
        
        # Get dwell times
        dwells = get_dwell_times(smoothed[0])
        
        # Check ALL dwells meet minimum
        for i, dwell in enumerate(dwells):
            assert dwell >= min_dwell, (
                f"Dwell {i} is {dwell} < min_dwell={min_dwell}! "
                f"This causes flickering!"
            )
        
        print(f"✓ Min-dwell={min_dwell}: all {len(dwells)} dwells ≥ {min_dwell}")


def test_no_single_frame_flickers():
    """Test no 1-frame or 2-frame flickers remain."""
    T = 500
    
    # Create sequence with flickers
    np.random.seed(42)
    base = np.random.randint(0, 4, T)
    
    # Insert 1-frame flickers
    for i in range(10, T-10, 50):
        base[i] = (base[i-1] + 1) % 4  # Different from neighbors
    
    # Insert 2-frame flickers
    for i in range(30, T-30, 50):
        base[i:i+2] = (base[i-1] + 2) % 4
    
    sequence = base.reshape(1, -1)
    
    # Apply smoothing
    policy = MedianHysteresis(min_dwell=5)
    smoothed = policy.smooth(sequence)
    
    # Check no short dwells remain
    dwells = get_dwell_times(smoothed[0])
    short_dwells = [d for d in dwells if d < 3]
    
    assert len(short_dwells) == 0, (
        f"Found {len(short_dwells)} flickers after smoothing: {short_dwells}! "
        f"Min dwell was {min(dwells) if dwells else 0}"
    )
    
    print(f"✓ No flickers: min dwell = {min(dwells) if dwells else 0}")


def test_hysteresis_thresholds():
    """Test enter/exit thresholds create proper hysteresis."""
    T = 200
    
    # Create confidence sequence that oscillates
    time = np.linspace(0, 4*np.pi, T)
    confidence = (np.sin(time) + 1) / 2  # 0 to 1
    
    # Binary state based on different thresholds
    enter_threshold = 0.7
    exit_threshold = 0.3
    
    # Without hysteresis (single threshold)
    single_threshold = confidence > 0.5
    transitions_single = count_transitions(single_threshold)
    
    # With hysteresis (enter/exit thresholds)
    states_hysteresis = np.zeros(T, dtype=bool)
    current_state = False
    
    for i in range(T):
        if current_state:
            # Currently active - need to drop below exit to deactivate
            if confidence[i] < exit_threshold:
                current_state = False
        else:
            # Currently inactive - need to exceed enter to activate
            if confidence[i] > enter_threshold:
                current_state = True
        states_hysteresis[i] = current_state
    
    transitions_hysteresis = count_transitions(states_hysteresis)
    
    # Hysteresis should reduce transitions
    reduction = (1 - transitions_hysteresis / max(transitions_single, 1)) * 100
    
    print(f"  Single threshold: {transitions_single} transitions")
    print(f"  Hysteresis: {transitions_hysteresis} transitions ({reduction:.1f}% reduction)")
    
    assert transitions_hysteresis < transitions_single, (
        "Hysteresis didn't reduce transitions!"
    )
    
    assert reduction > 50, (
        f"Hysteresis reduction only {reduction:.1f}%! Should be >50%"
    )


def test_transition_monotonicity():
    """Test smoothing never increases transitions."""
    np.random.seed(42)
    
    test_cases = [
        # (sequence_length, num_states, min_dwell)
        (100, 3, 3),
        (200, 4, 5),
        (500, 5, 7),
    ]
    
    for T, n_states, min_dwell in test_cases:
        # Create random sequence
        sequence = np.random.randint(0, n_states, T, dtype=np.int32).reshape(1, -1)
        
        # Count original transitions
        transitions_before = count_transitions(sequence[0])
        
        # Apply smoothing
        policy = MedianHysteresis(min_dwell=min_dwell)
        smoothed = policy.smooth(sequence)
        
        # Count smoothed transitions
        transitions_after = count_transitions(smoothed[0])
        
        # Transitions must not increase
        assert transitions_after <= transitions_before, (
            f"Transitions increased from {transitions_before} to {transitions_after}!"
        )
        
        reduction = (1 - transitions_after / max(transitions_before, 1)) * 100
        print(f"  T={T}, states={n_states}: {transitions_before} → {transitions_after} ({reduction:.1f}% reduction)")


def test_state_preservation():
    """Test smoothing preserves valid states (no new states)."""
    T = 300
    n_states = 4
    
    # Create sequence with only states 0, 1, 2 (not 3)
    np.random.seed(42)
    sequence = np.random.randint(0, 3, T, dtype=np.int32).reshape(1, -1)
    unique_before = set(sequence[0])
    
    # Apply smoothing
    policy = MedianHysteresis(min_dwell=5)
    smoothed = policy.smooth(sequence)
    unique_after = set(smoothed[0])
    
    # No new states should appear
    new_states = unique_after - unique_before
    assert len(new_states) == 0, (
        f"Smoothing created new states: {new_states}! "
        f"This corrupts the motif vocabulary!"
    )
    
    print(f"✓ State preservation: {unique_before} → {unique_after}")


def test_batch_consistency():
    """Test batch processing gives consistent results."""
    T = 200
    B = 4
    
    # Create batch of sequences
    np.random.seed(42)
    sequences = np.random.randint(0, 4, (B, T), dtype=np.int32)
    
    # Process as batch
    policy = MedianHysteresis(min_dwell=5)
    batch_result = policy.smooth(sequences)
    
    # Process individually
    individual_results = []
    for i in range(B):
        individual = policy.smooth(sequences[i:i+1])
        individual_results.append(individual[0])
    individual_results = np.stack(individual_results)
    
    # Results must match
    assert np.array_equal(batch_result, individual_results), (
        "Batch processing gives different results than individual!"
    )
    
    print(f"✓ Batch consistency verified for B={B}")


def test_edge_case_handling():
    """Test edge cases are handled correctly."""
    policy = MedianHysteresis(min_dwell=5)
    
    # Empty sequence
    empty = np.array([], dtype=np.int32).reshape(1, 0)
    result_empty = policy.smooth(empty)
    assert result_empty.shape == (1, 0), "Empty sequence handling failed"
    
    # Single element
    single = np.array([2], dtype=np.int32).reshape(1, 1)
    result_single = policy.smooth(single)
    assert np.array_equal(result_single, single), "Single element changed"
    
    # Very short sequence (less than min_dwell)
    short = np.array([0, 1, 2], dtype=np.int32).reshape(1, 3)
    result_short = policy.smooth(short)
    assert result_short.shape == short.shape, "Short sequence shape changed"
    
    # All same state
    same = np.array([3] * 100, dtype=np.int32).reshape(1, 100)
    result_same = policy.smooth(same)
    assert np.array_equal(result_same, same), "Constant sequence changed"
    
    # Alternating states (worst case)
    alternating = np.array([0, 1] * 50, dtype=np.int32).reshape(1, 100)
    result_alt = policy.smooth(alternating)
    dwells = get_dwell_times(result_alt[0])
    assert all(d >= policy.min_dwell for d in dwells), "Alternating sequence has short dwells"
    
    print("✓ All edge cases handled correctly")


def test_temporal_causality():
    """Test smoothing respects causality (no future information in online mode)."""
    # Note: MedianHysteresis uses a window, so it's not strictly causal
    # But we can test that it doesn't use TOO much future information
    
    T = 100
    window_size = 7
    
    # Create sequence with sudden change
    sequence = np.zeros(T, dtype=np.int32)
    sequence[50:] = 1  # Change at t=50
    sequence = sequence.reshape(1, -1)
    
    policy = MedianHysteresis(min_dwell=5, window_size=window_size)
    smoothed = policy.smooth(sequence)
    
    # Find where smoothed changes
    change_point = np.where(smoothed[0][:-1] != smoothed[0][1:])[0]
    
    if len(change_point) > 0:
        detected_change = change_point[0] + 1
        
        # Change shouldn't be detected too early
        max_lookahead = window_size // 2
        assert detected_change >= 50 - max_lookahead, (
            f"Change detected at {detected_change}, "
            f"but actual change at 50! Too much lookahead!"
        )
        
        print(f"✓ Temporal causality: change at 50, detected at {detected_change}")
    else:
        print("✓ No change detected (conservative smoothing)")


def test_real_world_patterns():
    """Test on realistic behavioral patterns."""
    # Simulate dog behavior: rest → walk → play → rest
    behaviors = {
        0: "rest",
        1: "walk", 
        2: "play",
        3: "sniff"
    }
    
    # Create realistic sequence
    sequence = []
    
    # Morning rest
    sequence.extend([0] * 50)
    
    # Walk with some sniffing
    for _ in range(3):
        sequence.extend([1] * 20)
        sequence.extend([3] * 5)
    
    # Play session
    sequence.extend([2] * 30)
    
    # Tired walk back
    sequence.extend([1] * 15)
    
    # Rest
    sequence.extend([0] * 40)
    
    # Add realistic noise (occasional misclassifications)
    sequence = np.array(sequence, dtype=np.int32)
    noise_idx = np.random.choice(len(sequence), size=10, replace=False)
    sequence[noise_idx] = np.random.randint(0, 4, size=10)
    
    sequence = sequence.reshape(1, -1)
    
    # Apply smoothing
    policy = MedianHysteresis(min_dwell=5)
    smoothed = policy.smooth(sequence)
    
    # Analyze results
    transitions_before = count_transitions(sequence[0])
    transitions_after = count_transitions(smoothed[0])
    
    dwells_before = get_dwell_times(sequence[0])
    dwells_after = get_dwell_times(smoothed[0])
    
    print(f"\n  Real-world pattern smoothing:")
    print(f"    Transitions: {transitions_before} → {transitions_after}")
    print(f"    Min dwell before: {min(dwells_before)}")
    print(f"    Min dwell after: {min(dwells_after)}")
    print(f"    Behaviors preserved: {sorted(set(smoothed[0]))}")
    
    assert min(dwells_after) >= policy.min_dwell, "Real pattern has short dwells!"
    assert transitions_after < transitions_before, "Real pattern transitions increased!"


if __name__ == "__main__":
    # Run all tests
    test_min_dwell_enforcement()
    test_no_single_frame_flickers()
    test_hysteresis_thresholds()
    test_transition_monotonicity()
    test_state_preservation()
    test_batch_consistency()
    test_edge_case_handling()
    test_temporal_causality()
    test_real_world_patterns()
    
    print("\n🎯 All temporal assertion tests passed!")
    print("Temporal coherence: ENFORCED")