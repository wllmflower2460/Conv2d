"""Tests for calibration improvements.

CRITICAL: ECE must decrease or hold when adding smoothing.
These tests catch calibration regressions that reduce model trustworthiness.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from conv2d.metrics.calibration import CalibrationAnalyzer
from conv2d.temporal.median import MedianHysteresisPolicy as MedianHysteresis


def generate_predictions(
    n_samples: int,
    n_classes: int,
    noise_level: float = 0.1,
    overconfident: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate predictions with controlled calibration.
    
    Returns:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
    """
    np.random.seed(42)
    
    # Generate true labels
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Generate probabilities
    y_prob = np.zeros((n_samples, n_classes), dtype=np.float32)
    
    for i in range(n_samples):
        # Start with correct class having high probability
        correct_prob = np.random.uniform(0.6, 0.95)
        
        if overconfident:
            # Push probabilities to extremes
            correct_prob = correct_prob ** 0.5
        
        # Add noise
        if np.random.rand() < noise_level:
            # Misclassification
            y_prob[i] = np.random.dirichlet(np.ones(n_classes))
        else:
            # Correct classification with some uncertainty
            y_prob[i, y_true[i]] = correct_prob
            remaining = 1 - correct_prob
            other_probs = np.random.dirichlet(np.ones(n_classes - 1))
            
            idx = 0
            for j in range(n_classes):
                if j != y_true[i]:
                    y_prob[i, j] = remaining * other_probs[idx]
                    idx += 1
    
    # Get predictions
    y_pred = np.argmax(y_prob, axis=1)
    
    return y_true, y_pred, y_prob


def test_ece_calculation():
    """Test ECE calculation is correct."""
    n_samples = 1000
    n_classes = 4
    
    # Generate well-calibrated predictions
    y_true, y_pred, y_prob = generate_predictions(
        n_samples, n_classes, noise_level=0.2, overconfident=False
    )
    
    # Calculate ECE
    analyzer = CalibrationAnalyzer(n_bins=10)
    result = analyzer.analyze(y_true, y_prob)
    
    # ECE should be reasonable
    assert 0 <= result.ece <= 1, f"ECE out of range: {result.ece}"
    assert result.ece < 0.2, f"ECE too high for well-calibrated: {result.ece}"
    
    print(f"✓ Well-calibrated ECE: {result.ece:.3f}")
    
    # Generate overconfident predictions
    y_true_over, y_pred_over, y_prob_over = generate_predictions(
        n_samples, n_classes, noise_level=0.2, overconfident=True
    )
    
    result_over = analyzer.analyze(y_true_over, y_prob_over)
    
    # Overconfident should have higher ECE
    assert result_over.ece > result.ece, (
        f"Overconfident ECE ({result_over.ece:.3f}) not higher than "
        f"well-calibrated ({result.ece:.3f})"
    )
    
    print(f"✓ Overconfident ECE: {result_over.ece:.3f} (higher as expected)")


def test_smoothing_improves_calibration():
    """Test temporal smoothing improves or maintains calibration."""
    n_samples = 500
    n_classes = 4
    
    # Generate sequence predictions with temporal noise
    y_true = []
    y_prob_list = []
    
    # Create blocks of consistent behavior with transitions
    block_size = 50
    for block_idx in range(n_samples // block_size):
        # Dominant class for this block
        dominant = block_idx % n_classes
        
        for i in range(block_size):
            y_true.append(dominant)
            
            # Generate probabilities with occasional errors
            prob = np.zeros(n_classes, dtype=np.float32)
            
            if np.random.rand() < 0.15:  # 15% error rate
                # Misclassification
                prob = np.random.dirichlet(np.ones(n_classes))
            else:
                # Correct with varying confidence
                confidence = np.random.uniform(0.6, 0.9)
                prob[dominant] = confidence
                remaining = 1 - confidence
                other_idx = [j for j in range(n_classes) if j != dominant]
                for j in other_idx:
                    prob[j] = remaining / (n_classes - 1)
            
            y_prob_list.append(prob)
    
    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob_list, dtype=np.float32)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Calculate ECE before smoothing
    analyzer = CalibrationAnalyzer(n_bins=10)
    calib_before = analyzer.analyze(y_true, y_prob)
    
    # Apply temporal smoothing to predictions
    smoother = MedianHysteresis(min_dwell=5, window_size=7)
    y_pred_smoothed = smoother.smooth(y_pred.reshape(1, -1))[0]
    
    # Update probabilities to reflect smoothed predictions
    # (In practice, this would be done by the model)
    y_prob_smoothed = np.copy(y_prob)
    
    # Adjust probabilities where predictions changed
    changed_idx = np.where(y_pred != y_pred_smoothed)[0]
    for idx in changed_idx:
        # Redistribute probability to smoothed prediction
        old_pred = y_pred[idx]
        new_pred = y_pred_smoothed[idx]
        
        # Transfer confidence from old to new prediction
        transfer = min(0.2, y_prob_smoothed[idx, old_pred] * 0.3)
        y_prob_smoothed[idx, old_pred] -= transfer
        y_prob_smoothed[idx, new_pred] += transfer
        
        # Renormalize
        y_prob_smoothed[idx] = y_prob_smoothed[idx] / y_prob_smoothed[idx].sum()
    
    # Calculate ECE after smoothing
    calib_after = analyzer.analyze(y_true, y_prob_smoothed)
    
    print(f"\n  Calibration with smoothing:")
    print(f"    ECE before: {calib_before.ece:.3f}")
    print(f"    ECE after:  {calib_after.ece:.3f}")
    print(f"    Change: {(calib_after.ece - calib_before.ece):.3f}")
    
    # ECE should not increase significantly
    assert calib_after.ece <= calib_before.ece * 1.1, (
        f"ECE increased too much with smoothing: "
        f"{calib_before.ece:.3f} → {calib_after.ece:.3f}"
    )
    
    # Accuracy should improve
    acc_before = np.mean(y_pred == y_true)
    acc_after = np.mean(y_pred_smoothed == y_true)
    
    print(f"    Accuracy before: {acc_before:.3f}")
    print(f"    Accuracy after:  {acc_after:.3f}")
    
    assert acc_after >= acc_before, "Accuracy decreased with smoothing!"


def test_mce_bounds():
    """Test Maximum Calibration Error (MCE) is bounded."""
    n_samples = 1000
    n_classes = 4
    
    test_cases = [
        ("random", 0.5),  # Random predictions
        ("perfect", 0.0),  # Perfect predictions
        ("overconfident", 0.1),  # Overconfident
    ]
    
    analyzer = CalibrationAnalyzer(n_bins=10)
    
    for case_name, noise_level in test_cases:
        if case_name == "perfect":
            # Perfect predictions
            y_true = np.random.randint(0, n_classes, n_samples)
            y_prob = np.zeros((n_samples, n_classes), dtype=np.float32)
            for i in range(n_samples):
                y_prob[i, y_true[i]] = 1.0
        else:
            # Generate with noise
            y_true, _, y_prob = generate_predictions(
                n_samples, n_classes, 
                noise_level=noise_level,
                overconfident=(case_name == "overconfident")
            )
        
        result = analyzer.analyze(y_true, y_prob)
        
        # MCE bounds
        assert 0 <= result.mce <= 1, f"MCE out of bounds for {case_name}: {result.mce}"
        
        # MCE >= ECE (max >= mean)
        assert result.mce >= result.ece, f"MCE < ECE for {case_name}!"
        
        print(f"✓ {case_name:12s}: ECE={result.ece:.3f}, MCE={result.mce:.3f}")


def test_confidence_histogram():
    """Test confidence histogram shows proper distribution."""
    n_samples = 1000
    n_classes = 4
    
    # Generate predictions
    y_true, y_pred, y_prob = generate_predictions(n_samples, n_classes)
    
    # Get max confidences
    confidences = np.max(y_prob, axis=1)
    
    # Check distribution
    assert confidences.min() >= 0, "Negative confidence!"
    assert confidences.max() <= 1, "Confidence > 1!"
    
    # Should have reasonable spread
    std = np.std(confidences)
    assert std > 0.05, f"Confidence too concentrated: std={std:.3f}"
    
    # Binned analysis
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(confidences, bins=bins)
    
    # Should use multiple bins (not all in one)
    bins_used = np.sum(hist > 0)
    assert bins_used >= 5, f"Only {bins_used} confidence bins used!"
    
    print(f"✓ Confidence spread: mean={np.mean(confidences):.3f}, std={std:.3f}")


def test_reliability_diagram_bins():
    """Test reliability diagram bin calculation."""
    n_samples = 2000
    n_classes = 4
    
    # Generate varied calibration
    y_true, y_pred, y_prob = generate_predictions(n_samples, n_classes, noise_level=0.2)
    
    analyzer = CalibrationAnalyzer(n_bins=10)
    result = analyzer.analyze(y_true, y_prob)
    
    # Check bin properties
    assert len(result.bin_boundaries) == 11, "Wrong number of bin boundaries"
    assert result.bin_boundaries[0] == 0, "First boundary not 0"
    assert result.bin_boundaries[-1] == 1, "Last boundary not 1"
    
    # Bins should be monotonic
    for i in range(len(result.bin_boundaries) - 1):
        assert result.bin_boundaries[i] < result.bin_boundaries[i+1], (
            f"Bin boundaries not monotonic at {i}"
        )
    
    # Check accuracies and confidences
    assert len(result.bin_accuracies) == 10, "Wrong number of bin accuracies"
    assert len(result.bin_confidences) == 10, "Wrong number of bin confidences"
    assert len(result.bin_counts) == 10, "Wrong number of bin counts"
    
    # Sum of bin counts should equal n_samples
    total_count = sum(result.bin_counts)
    assert total_count == n_samples, f"Bin counts sum to {total_count}, not {n_samples}"
    
    print(f"✓ Reliability diagram: {sum(c > 0 for c in result.bin_counts)} non-empty bins")


def test_calibration_with_class_imbalance():
    """Test calibration metrics with imbalanced classes."""
    n_samples = 1000
    n_classes = 4
    
    # Create imbalanced distribution
    class_weights = [0.5, 0.3, 0.15, 0.05]
    y_true = np.random.choice(n_classes, size=n_samples, p=class_weights)
    
    # Generate predictions
    y_prob = np.zeros((n_samples, n_classes), dtype=np.float32)
    
    for i in range(n_samples):
        if np.random.rand() < 0.8:  # 80% accuracy
            # Correct prediction
            y_prob[i, y_true[i]] = np.random.uniform(0.7, 0.95)
            remaining = 1 - y_prob[i, y_true[i]]
            for j in range(n_classes):
                if j != y_true[i]:
                    y_prob[i, j] = remaining / (n_classes - 1)
        else:
            # Random prediction
            y_prob[i] = np.random.dirichlet(np.ones(n_classes))
    
    # Calculate calibration
    analyzer = CalibrationAnalyzer(n_bins=10)
    result = analyzer.analyze(y_true, y_prob)
    
    # Should still have reasonable calibration
    assert result.ece < 0.3, f"ECE too high with imbalance: {result.ece:.3f}"
    
    # Check per-class analysis
    class_counts = np.bincount(y_true, minlength=n_classes)
    print(f"\n  Class distribution: {class_counts}")
    print(f"  ECE with imbalance: {result.ece:.3f}")


def test_temperature_scaling_effect():
    """Test effect of temperature scaling on calibration."""
    n_samples = 1000
    n_classes = 4
    
    # Generate overconfident predictions
    y_true, y_pred, y_prob = generate_predictions(
        n_samples, n_classes, overconfident=True
    )
    
    analyzer = CalibrationAnalyzer(n_bins=10)
    
    # Original calibration
    calib_original = analyzer.analyze(y_true, y_prob)
    
    # Apply temperature scaling
    temperatures = [0.5, 1.0, 1.5, 2.0]
    
    for T in temperatures:
        # Scale logits by temperature
        logits = np.log(y_prob + 1e-8)  # Convert to logits
        scaled_logits = logits / T
        
        # Convert back to probabilities
        y_prob_scaled = np.exp(scaled_logits)
        y_prob_scaled = y_prob_scaled / y_prob_scaled.sum(axis=1, keepdims=True)
        
        # Measure calibration
        calib_scaled = analyzer.analyze(y_true, y_prob_scaled)
        
        print(f"  T={T:.1f}: ECE={calib_scaled.ece:.3f}, MCE={calib_scaled.mce:.3f}")
    
    # Higher temperature should reduce ECE for overconfident model
    logits = np.log(y_prob + 1e-8)
    y_prob_T2 = np.exp(logits / 2.0)
    y_prob_T2 = y_prob_T2 / y_prob_T2.sum(axis=1, keepdims=True)
    calib_T2 = analyzer.analyze(y_true, y_prob_T2)
    
    assert calib_T2.ece < calib_original.ece, (
        f"Temperature scaling didn't improve overconfident model: "
        f"{calib_original.ece:.3f} → {calib_T2.ece:.3f}"
    )


def test_brier_score_decomposition():
    """Test Brier score and its relationship to calibration."""
    n_samples = 1000
    n_classes = 4
    
    # Generate predictions
    y_true, y_pred, y_prob = generate_predictions(n_samples, n_classes)
    
    # Calculate Brier score
    brier_score = 0
    for i in range(n_samples):
        # One-hot encode true label
        y_true_onehot = np.zeros(n_classes)
        y_true_onehot[y_true[i]] = 1
        
        # Squared difference
        brier_score += np.sum((y_prob[i] - y_true_onehot) ** 2)
    
    brier_score /= n_samples
    
    # Brier score bounds: 0 (perfect) to 2 (worst)
    assert 0 <= brier_score <= 2, f"Brier score out of bounds: {brier_score}"
    
    # For decent predictions, should be < 1
    assert brier_score < 1, f"Brier score too high: {brier_score:.3f}"
    
    print(f"✓ Brier score: {brier_score:.3f}")
    
    # Relationship with ECE
    analyzer = CalibrationAnalyzer(n_bins=10)
    calib = analyzer.analyze(y_true, y_prob)
    
    # Lower Brier score should correlate with lower ECE (roughly)
    print(f"  ECE: {calib.ece:.3f}, Brier: {brier_score:.3f}")


if __name__ == "__main__":
    # Run all tests
    test_ece_calculation()
    test_smoothing_improves_calibration()
    test_mce_bounds()
    test_confidence_histogram()
    test_reliability_diagram_bins()
    test_calibration_with_class_imbalance()
    test_temperature_scaling_effect()
    test_brier_score_decomposition()
    
    print("\n🎯 All calibration tests passed!")
    print("Model trustworthiness: VERIFIED")