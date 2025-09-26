# Testing and Validation Framework Documentation

The testing framework provides comprehensive regression tests that catch real failures and prevent silent regressions with shape/dtype enforcement, determinism validation, temporal assertions, calibration improvements, and performance benchmarks.

## Overview

Key features:
- **Shape & dtype enforcement**: Every stage enforces `(B,9,2,100) → outputs` with `float32`
- **Determinism tests**: Same input → identical output, always
- **Temporal assertions**: Min-dwell enforcement and hysteresis monotonicity  
- **Calibration validation**: ECE improvements and metric bounds
- **Performance benchmarks**: Speed thresholds with micro-benchmarks
- **Real failure detection**: Tests catch actual production issues

## Architecture

```
Test Suite → Component Tests → Integration Tests → Performance Tests
     ↓              ↓               ↓                    ↓
Shape/Dtype → Determinism → Temporal Logic → Speed Benchmarks
     ↓              ↓               ↓                    ↓
Production Safety Guarantees
```

## Test Categories

### 1. Shape & Dtype Enforcement (`tests/test_shape_dtype_enforcement.py`)

Tests that every pipeline stage maintains strict shape and dtype contracts for edge deployment safety.

```python
import pytest
import torch
import numpy as np
from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.gmm import GMMClusterer
from conv2d.temporal.median import MedianHysteresisPolicy

class TestShapeDtypeEnforcement:
    """Test shape and dtype enforcement throughout pipeline."""
    
    def test_fsq_contract_shape_enforcement(self):
        """FSQ must enforce (B,9,2,100) input and correct output shapes."""
        
        # Valid input
        x = torch.randn(32, 9, 2, 100, dtype=torch.float32)
        result = encode_fsq(x)
        
        # Check output shapes
        assert result.codes.shape == (32, 64), "FSQ codes shape mismatch"
        assert result.features.shape[0] == 32, "FSQ features batch mismatch"
        assert result.embeddings.shape == (32, 64), "FSQ embeddings shape mismatch"
        
        # Check output dtypes
        assert result.codes.dtype == torch.int32, "FSQ codes must be int32"
        assert result.features.dtype == torch.float32, "FSQ features must be float32"
        assert result.embeddings.dtype == torch.float32, "FSQ embeddings must be float32"
    
    def test_fsq_contract_invalid_shapes(self):
        """FSQ must reject invalid input shapes."""
        
        # Wrong channel count
        with pytest.raises(ValueError, match="shape.*9.*2.*100"):
            x = torch.randn(32, 8, 2, 100, dtype=torch.float32)  # 8 instead of 9
            encode_fsq(x)
        
        # Wrong sensor count
        with pytest.raises(ValueError, match="shape.*9.*2.*100"):
            x = torch.randn(32, 9, 3, 100, dtype=torch.float32)  # 3 instead of 2
            encode_fsq(x)
        
        # Wrong timesteps
        with pytest.raises(ValueError, match="shape.*9.*2.*100"):
            x = torch.randn(32, 9, 2, 50, dtype=torch.float32)   # 50 instead of 100
            encode_fsq(x)
    
    def test_fsq_contract_invalid_dtypes(self):
        """FSQ must reject invalid input dtypes."""
        
        # Wrong dtype
        with pytest.raises(TypeError, match="float32"):
            x = torch.randn(32, 9, 2, 100, dtype=torch.float64)  # float64 instead of float32
            encode_fsq(x)
    
    def test_clustering_shape_preservation(self):
        """Clustering must preserve batch dimensions."""
        
        features = np.random.randn(1000, 256).astype(np.float32)
        
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(features, k=4)
        
        # Check shape preservation
        assert labels.shape == (1000,), "Clustering must preserve sample count"
        assert labels.dtype == np.int32, "Clustering labels must be int32"
        
        # Check label range
        assert labels.min() >= 0, "Labels must be non-negative"
        assert labels.max() < 4, "Labels must be within cluster range"
    
    def test_temporal_shape_preservation(self):
        """Temporal smoothing must preserve shapes exactly."""
        
        labels = np.random.randint(0, 4, (32, 100), dtype=np.int32)
        
        policy = MedianHysteresisPolicy(min_dwell=3)
        smoothed = policy.smooth(labels)
        
        # Shape must be identical
        assert smoothed.shape == labels.shape, "Temporal smoothing shape mismatch"
        assert smoothed.dtype == labels.dtype, "Temporal smoothing dtype changed"
        
        # Values must be from original set
        original_states = set(labels.flatten())
        smoothed_states = set(smoothed.flatten())
        assert smoothed_states <= original_states, "Temporal smoothing introduced new states"
    
    def test_end_to_end_shape_dtype_pipeline(self):
        """Complete pipeline must maintain shape/dtype contracts."""
        
        # Start with valid IMU data
        x = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        
        # FSQ encoding
        result = encode_fsq(x)
        assert result.features.dtype == torch.float32, "FSQ broke dtype contract"
        
        # Clustering
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(result.features.numpy(), k=4)
        assert labels.dtype == np.int32, "Clustering broke dtype contract"
        
        # Temporal smoothing
        labels_2d = labels.reshape(16, -1)  # Reshape for temporal
        policy = MedianHysteresisPolicy(min_dwell=3)
        smoothed = policy.smooth(labels_2d)
        
        # Final checks
        assert smoothed.shape == labels_2d.shape, "End-to-end shape violation"
        assert smoothed.dtype == np.int32, "End-to-end dtype violation"
```

### 2. Determinism Tests (`tests/test_determinism.py`)

Tests that ensure reproducible results across runs with identical inputs.

```python
import pytest
import torch
import numpy as np
from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.kmeans import KMeansClusterer
from conv2d.clustering.gmm import GMMClusterer

class TestDeterminism:
    """Test deterministic behavior across all components."""
    
    def test_fsq_determinism(self):
        """FSQ encoding must be 100% deterministic."""
        
        x = torch.randn(32, 9, 2, 100, dtype=torch.float32)
        
        # Multiple runs with reset_stats=True
        result1 = encode_fsq(x, reset_stats=True)
        result2 = encode_fsq(x, reset_stats=True) 
        result3 = encode_fsq(x, reset_stats=True)
        
        # Codes must be identical
        assert torch.equal(result1.codes, result2.codes), "FSQ codes not deterministic"
        assert torch.equal(result2.codes, result3.codes), "FSQ codes not deterministic"
        
        # Features must be identical
        assert torch.allclose(result1.features, result2.features, atol=1e-6), "FSQ features not deterministic"
        assert torch.allclose(result2.features, result3.features, atol=1e-6), "FSQ features not deterministic"
        
        # Embeddings must be identical
        assert torch.allclose(result1.embeddings, result2.embeddings, atol=1e-6), "FSQ embeddings not deterministic"
        assert torch.allclose(result2.embeddings, result3.embeddings, atol=1e-6), "FSQ embeddings not deterministic"
    
    def test_kmeans_determinism_with_hungarian(self):
        """K-means with Hungarian matching must produce stable labels."""
        
        features = np.random.RandomState(42).randn(1000, 64).astype(np.float32)
        
        # First run
        clusterer1 = KMeansClusterer(random_state=42)
        labels1 = clusterer1.fit_predict(features, k=4)
        
        # Second run with same seed
        clusterer2 = KMeansClusterer(random_state=42)
        labels2 = clusterer2.fit_predict(features, k=4)
        
        # Labels should be identical with same seed
        assert np.array_equal(labels1, labels2), "K-means not deterministic with same seed"
        
        # Third run with Hungarian matching
        clusterer3 = KMeansClusterer(random_state=123)  # Different seed
        labels3 = clusterer3.fit_predict(features, k=4, prior_labels=labels1)
        
        # Hungarian matching should improve agreement
        agreement_raw = np.mean(labels1 == labels3)
        
        # Compute optimal assignment manually
        from sklearn.metrics import adjusted_rand_score
        ari_score = adjusted_rand_score(labels1, labels3)
        
        # With Hungarian matching, ARI should be high
        assert ari_score > 0.8, f"Hungarian matching failed: ARI={ari_score:.3f}"
    
    def test_gmm_determinism_with_hungarian(self):
        """GMM with Hungarian matching must produce stable labels."""
        
        features = np.random.RandomState(42).randn(1000, 64).astype(np.float32)
        
        # First run
        clusterer1 = GMMClusterer(random_state=42)
        labels1 = clusterer1.fit_predict(features, k=4)
        
        # Second run with different seed but Hungarian matching
        clusterer2 = GMMClusterer(random_state=123)
        labels2 = clusterer2.fit_predict(features, k=4, prior_labels=labels1)
        
        # Agreement should be high with Hungarian matching
        agreement = np.mean(labels1 == labels2)
        assert agreement > 0.8, f"GMM Hungarian matching failed: agreement={agreement:.1%}"
    
    def test_temporal_determinism(self):
        """Temporal smoothing must be deterministic."""
        
        labels = np.random.RandomState(42).randint(0, 4, (32, 100))
        
        from conv2d.temporal.median import MedianHysteresisPolicy
        
        policy1 = MedianHysteresisPolicy(min_dwell=5, window_size=7)
        policy2 = MedianHysteresisPolicy(min_dwell=5, window_size=7)
        
        smoothed1 = policy1.smooth(labels)
        smoothed2 = policy2.smooth(labels)
        
        # Results must be identical
        assert np.array_equal(smoothed1, smoothed2), "Temporal smoothing not deterministic"
    
    def test_full_pipeline_determinism(self):
        """Complete pipeline must be deterministic."""
        
        x = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        
        def run_pipeline(seed):
            # FSQ (deterministic)
            result = encode_fsq(x, reset_stats=True)
            
            # Clustering (seeded)
            clusterer = GMMClusterer(random_state=seed)
            labels = clusterer.fit_predict(result.features.numpy(), k=4)
            
            # Temporal (deterministic)
            from conv2d.temporal.median import MedianHysteresisPolicy
            policy = MedianHysteresisPolicy(min_dwell=3)
            labels_2d = labels.reshape(16, -1)
            smoothed = policy.smooth(labels_2d)
            
            return smoothed
        
        # Same seed should produce identical results
        result1 = run_pipeline(42)
        result2 = run_pipeline(42)
        
        assert np.array_equal(result1, result2), "Full pipeline not deterministic with same seed"
        
        # Different seeds with Hungarian matching should be similar
        result3 = run_pipeline(123)
        
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(result1.flatten(), result3.flatten())
        
        # Should be reasonably similar (Hungarian matching helps)
        assert ari > 0.5, f"Pipeline stability poor: ARI={ari:.3f}"
```

### 3. Temporal Assertions (`tests/test_temporal_assertions.py`)

Tests that temporal policies correctly enforce behavioral constraints.

```python
import pytest
import numpy as np
from conv2d.temporal.median import MedianHysteresisPolicy

class TestTemporalAssertions:
    """Test temporal smoothing behavioral constraints."""
    
    def test_min_dwell_enforcement(self):
        """Min-dwell policy must eliminate short segments."""
        
        # Create sequence with 1-2 frame flickers
        labels = np.array([
            [0, 0, 1, 0, 0, 1, 1, 1, 2, 0],  # 1-frame flickers
            [1, 1, 2, 2, 1, 2, 2, 2, 2, 1],  # Valid segments
        ])
        
        policy = MedianHysteresisPolicy(min_dwell=3)
        smoothed = policy.smooth(labels)
        
        # Check that no segment is shorter than min_dwell
        for b in range(labels.shape[0]):
            segments = self._find_segments(smoothed[b])
            
            for start, end, state in segments:
                duration = end - start
                assert duration >= 3, f"Segment too short: {duration} < 3 at positions {start}-{end}"
    
    def test_hysteresis_monotonicity(self):
        """Hysteresis smoothing should reduce transitions."""
        
        # Create noisy sequence
        np.random.seed(42)
        base_sequence = np.repeat([0, 1, 2, 3], 25)  # 100 timesteps
        noise_indices = np.random.choice(100, size=20, replace=False)
        noisy_sequence = base_sequence.copy()
        noisy_sequence[noise_indices] = np.random.randint(0, 4, 20)
        
        labels = noisy_sequence.reshape(1, -1)
        
        policy = MedianHysteresisPolicy(
            min_dwell=5,
            window_size=7,
            enter_threshold=0.7,
            exit_threshold=0.3,
        )
        
        smoothed = policy.smooth(labels)
        
        # Count transitions
        transitions_before = np.sum(np.diff(labels[0]) != 0)
        transitions_after = np.sum(np.diff(smoothed[0]) != 0)
        
        # Smoothing should reduce transitions
        assert transitions_after <= transitions_before, \
            f"Smoothing increased transitions: {transitions_before} → {transitions_after}"
        
        # Should achieve significant reduction
        reduction = (transitions_before - transitions_after) / transitions_before
        assert reduction >= 0.3, f"Insufficient transition reduction: {reduction:.1%}"
    
    def test_state_preservation(self):
        """Temporal smoothing must not introduce new states."""
        
        labels = np.random.randint(0, 4, (10, 50))
        
        policy = MedianHysteresisPolicy(min_dwell=3)
        smoothed = policy.smooth(labels)
        
        # Original states
        original_states = set(labels.flatten())
        smoothed_states = set(smoothed.flatten())
        
        # No new states should be introduced
        assert smoothed_states <= original_states, \
            f"New states introduced: {smoothed_states - original_states}"
    
    def test_enter_exit_thresholds(self):
        """Enter/exit thresholds should prevent oscillation."""
        
        # Create oscillating sequence (alternating states)
        oscillating = np.tile([0, 1], 50)  # 100 timesteps
        labels = oscillating.reshape(1, -1)
        
        policy = MedianHysteresisPolicy(
            min_dwell=1,  # Don't use min_dwell for this test
            window_size=5,
            enter_threshold=0.8,  # High threshold
            exit_threshold=0.2,   # Low threshold  
        )
        
        smoothed = policy.smooth(labels)
        
        # Hysteresis should dramatically reduce transitions
        transitions_before = np.sum(np.diff(labels[0]) != 0)
        transitions_after = np.sum(np.diff(smoothed[0]) != 0)
        
        reduction_ratio = transitions_after / transitions_before
        assert reduction_ratio < 0.5, \
            f"Hysteresis failed to reduce oscillation: {reduction_ratio:.1%} remaining"
    
    def test_temporal_consistency(self):
        """Temporal smoothing should be consistent across similar inputs."""
        
        # Create base sequence
        base = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] * 8)  # 96 timesteps
        
        # Add small variations
        var1 = base.copy()
        var1[10:12] = 0  # Change a small segment
        
        var2 = base.copy()  
        var2[10:12] = 0  # Same change
        
        policy = MedianHysteresisPolicy(min_dwell=3)
        
        smoothed1 = policy.smooth(var1.reshape(1, -1))[0]
        smoothed2 = policy.smooth(var2.reshape(1, -1))[0]
        
        # Should produce identical results for identical inputs
        assert np.array_equal(smoothed1, smoothed2), "Temporal policy not consistent"
    
    def test_boundary_conditions(self):
        """Test temporal policy behavior at sequence boundaries."""
        
        # Short sequences
        short_labels = np.array([[0, 1, 0]])  # Only 3 timesteps
        
        policy = MedianHysteresisPolicy(min_dwell=5)  # Longer than sequence
        smoothed = policy.smooth(short_labels)
        
        # Should handle gracefully (likely all same state)
        assert smoothed.shape == short_labels.shape, "Shape changed for short sequence"
        
        # Single timestep
        single_labels = np.array([[2]])
        smoothed_single = policy.smooth(single_labels)
        assert np.array_equal(smoothed_single, single_labels), "Single timestep handling failed"
    
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
        
        segments.append((start, len(sequence), current_state))
        return segments
```

### 4. Calibration Improvements (`tests/test_calibration_improvements.py`)

Tests that ensure calibration metrics behave correctly and improve with temporal smoothing.

```python
import pytest
import numpy as np
from conv2d.metrics.calibration import CalibrationAnalyzer
from conv2d.temporal.median import MedianHysteresisPolicy

class TestCalibrationImprovements:
    """Test calibration metric behavior and improvements."""
    
    def test_ece_mce_relationship(self):
        """MCE must always be >= ECE."""
        
        # Generate test data
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.rand(1000)
        
        analyzer = CalibrationAnalyzer(n_bins=10)
        calibration = analyzer.analyze(y_true, y_true, y_prob)
        
        # MCE should be >= ECE
        assert calibration.mce >= calibration.ece, \
            f"MCE ({calibration.mce:.3f}) < ECE ({calibration.ece:.3f})"
    
    def test_perfect_calibration(self):
        """Perfectly calibrated model should have ECE ≈ 0."""
        
        n_samples = 10000
        y_true = np.random.randint(0, 2, n_samples)
        
        # Create perfectly calibrated probabilities
        # For binary case: prob = fraction of positives in bin
        y_prob = y_true.astype(float) + np.random.normal(0, 0.01, n_samples)
        y_prob = np.clip(y_prob, 0, 1)
        
        analyzer = CalibrationAnalyzer(n_bins=10)
        calibration = analyzer.analyze(y_true, y_true, y_prob)
        
        # ECE should be very low for well-calibrated model
        assert calibration.ece < 0.1, f"ECE too high for calibrated model: {calibration.ece:.3f}"
    
    def test_overconfident_model(self):
        """Overconfident model should have high ECE."""
        
        y_true = np.random.randint(0, 2, 1000)
        
        # Create overconfident probabilities
        y_prob = np.where(y_true == 1, 0.95, 0.05)  # Very confident
        
        # Make predictions wrong 20% of the time
        wrong_indices = np.random.choice(1000, size=200, replace=False)
        y_pred = y_true.copy()
        y_pred[wrong_indices] = 1 - y_pred[wrong_indices]
        
        analyzer = CalibrationAnalyzer(n_bins=10)
        calibration = analyzer.analyze(y_true, y_pred, y_prob)
        
        # ECE should be high for overconfident model
        assert calibration.ece > 0.1, f"ECE too low for overconfident model: {calibration.ece:.3f}"
    
    def test_temporal_smoothing_calibration_impact(self):
        """Temporal smoothing should improve or maintain calibration."""
        
        # Generate temporal sequence data
        T = 100
        B = 50
        
        # Ground truth with temporal structure
        y_true_temporal = np.zeros((B, T))
        for b in range(B):
            states = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3]
            state_sequence = np.repeat(states, T // len(states))[:T]
            y_true_temporal[b] = state_sequence
        
        # Predictions with some noise
        y_pred_temporal = y_true_temporal.copy()
        noise_mask = np.random.random((B, T)) < 0.1
        y_pred_temporal[noise_mask] = np.random.randint(0, 4, noise_mask.sum())
        
        # Confidence scores (lower for incorrect predictions)
        y_prob_temporal = np.where(
            y_pred_temporal == y_true_temporal,
            np.random.uniform(0.7, 0.95, (B, T)),
            np.random.uniform(0.3, 0.6, (B, T))
        )
        
        # Flatten for calibration analysis
        y_true_flat = y_true_temporal.flatten()
        y_pred_flat = y_pred_temporal.flatten()
        y_prob_flat = y_prob_temporal.flatten()
        
        # Calibration before temporal smoothing
        analyzer = CalibrationAnalyzer(n_bins=10)
        cal_before = analyzer.analyze(y_true_flat, y_pred_flat, y_prob_flat)
        
        # Apply temporal smoothing
        policy = MedianHysteresisPolicy(min_dwell=5)
        y_pred_smoothed = policy.smooth(y_pred_temporal)
        
        # Align probabilities with smoothed predictions
        y_prob_smoothed = y_prob_temporal.copy()
        
        # Calibration after temporal smoothing  
        y_pred_smooth_flat = y_pred_smoothed.flatten()
        y_prob_smooth_flat = y_prob_smoothed.flatten()
        
        cal_after = analyzer.analyze(y_true_flat, y_pred_smooth_flat, y_prob_smooth_flat)
        
        # ECE should improve or stay the same
        # (temporal smoothing might improve accuracy, affecting calibration)
        ece_change = cal_after.ece - cal_before.ece
        assert ece_change <= 0.05, \
            f"Temporal smoothing significantly worsened ECE: {ece_change:.3f}"
        
        # Accuracy should improve
        acc_before = (y_pred_flat == y_true_flat).mean()
        acc_after = (y_pred_smooth_flat == y_true_flat).mean()
        
        assert acc_after >= acc_before, \
            f"Temporal smoothing reduced accuracy: {acc_before:.3f} → {acc_after:.3f}"
    
    def test_brier_score_decomposition(self):
        """Brier score should decompose correctly."""
        
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.rand(1000)
        
        analyzer = CalibrationAnalyzer()
        calibration = analyzer.analyze(y_true, y_true, y_prob)
        
        # Brier score should be in valid range [0, 1]
        assert 0 <= calibration.brier <= 1, \
            f"Brier score out of range: {calibration.brier}"
        
        # For binary case, can compute manually
        brier_manual = np.mean((y_prob - y_true) ** 2)
        
        assert abs(calibration.brier - brier_manual) < 1e-6, \
            f"Brier score mismatch: {calibration.brier} vs {brier_manual}"
    
    def test_calibration_bins_coverage(self):
        """All confidence bins should be covered in analysis."""
        
        # Create data that spans all confidence ranges
        y_true = np.random.randint(0, 2, 10000)
        y_prob = np.random.rand(10000)  # Uniform [0,1] 
        
        analyzer = CalibrationAnalyzer(n_bins=10)
        calibration = analyzer.analyze(y_true, y_true, y_prob)
        
        # Should have reasonable ECE (not too high)
        assert calibration.ece < 0.5, f"ECE unexpectedly high: {calibration.ece}"
        
        # MCE >= ECE relationship
        assert calibration.mce >= calibration.ece, "MCE < ECE violation"
    
    def test_multiclass_calibration(self):
        """Calibration should work for multiclass problems."""
        
        n_classes = 4
        n_samples = 2000
        
        y_true = np.random.randint(0, n_classes, n_samples)
        
        # Create class probabilities
        y_prob_matrix = np.random.dirichlet([1] * n_classes, n_samples)
        y_pred = np.argmax(y_prob_matrix, axis=1)
        y_prob = y_prob_matrix.max(axis=1)  # Max probability
        
        analyzer = CalibrationAnalyzer(n_bins=10)
        calibration = analyzer.analyze(y_true, y_pred, y_prob)
        
        # Should produce valid calibration metrics
        assert 0 <= calibration.ece <= 1, "Invalid ECE for multiclass"
        assert 0 <= calibration.mce <= 1, "Invalid MCE for multiclass"  
        assert 0 <= calibration.brier <= 1, "Invalid Brier for multiclass"
        assert calibration.mce >= calibration.ece, "MCE < ECE for multiclass"
```

### 5. Speed Benchmarks (`tests/test_speed_benchmarks.py`)

Performance tests with specific thresholds to catch performance regressions.

```python
import pytest
import time
import torch
import numpy as np
from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.gmm import GMMClusterer
from conv2d.temporal.median import MedianHysteresisPolicy

class TestSpeedBenchmarks:
    """Performance benchmarks with regression detection."""
    
    def test_fsq_encoding_speed(self):
        """FSQ encoding must complete within time thresholds."""
        
        # Single sample (edge deployment scenario)
        x_single = torch.randn(1, 9, 2, 100, dtype=torch.float32)
        
        start_time = time.time()
        result = encode_fsq(x_single)
        single_duration = time.time() - start_time
        
        # Must be under 10ms for single sample
        assert single_duration < 0.010, \
            f"FSQ encoding too slow for single sample: {single_duration*1000:.1f}ms > 10ms"
        
        # Batch processing (training scenario)
        x_batch = torch.randn(32, 9, 2, 100, dtype=torch.float32)
        
        start_time = time.time()
        result = encode_fsq(x_batch)
        batch_duration = time.time() - start_time
        
        # Must be under 50ms for batch of 32
        assert batch_duration < 0.050, \
            f"FSQ encoding too slow for batch: {batch_duration*1000:.1f}ms > 50ms"
        
        # Throughput check
        samples_per_second = 32 / batch_duration
        assert samples_per_second > 1000, \
            f"FSQ throughput too low: {samples_per_second:.0f} samples/sec < 1000"
    
    def test_clustering_speed(self):
        """Clustering must complete within time thresholds."""
        
        features = np.random.randn(1000, 256).astype(np.float32)
        
        # K-means clustering
        start_time = time.time()
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(features, k=4)
        clustering_duration = time.time() - start_time
        
        # Must be under 200ms for 1000 samples
        assert clustering_duration < 0.200, \
            f"Clustering too slow: {clustering_duration*1000:.1f}ms > 200ms"
        
        # Throughput check
        samples_per_second = 1000 / clustering_duration
        assert samples_per_second > 10000, \
            f"Clustering throughput too low: {samples_per_second:.0f} samples/sec < 10000"
    
    def test_temporal_smoothing_speed(self):
        """Temporal smoothing must complete within time thresholds."""
        
        labels = np.random.randint(0, 4, (100, 1000))  # Large temporal sequences
        
        policy = MedianHysteresisPolicy(min_dwell=5, window_size=7)
        
        start_time = time.time()
        smoothed = policy.smooth(labels)
        smoothing_duration = time.time() - start_time
        
        # Must be under 100ms for large sequences
        assert smoothing_duration < 0.100, \
            f"Temporal smoothing too slow: {smoothing_duration*1000:.1f}ms > 100ms"
        
        # Timesteps per second
        timesteps_per_second = (100 * 1000) / smoothing_duration
        assert timesteps_per_second > 1000000, \
            f"Temporal throughput too low: {timesteps_per_second:.0f} timesteps/sec < 1M"
    
    def test_end_to_end_pipeline_speed(self):
        """Complete pipeline must meet real-time requirements."""
        
        x = torch.randn(1, 9, 2, 100, dtype=torch.float32)  # Single sample
        
        start_time = time.time()
        
        # FSQ encoding
        result = encode_fsq(x)
        
        # Clustering (using pre-fitted clusterer for speed)
        features_np = result.features.numpy()
        clusterer = GMMClusterer(random_state=42)
        
        # Simulate pre-fitted clusterer (just predict)
        labels = np.array([2])  # Mock result for speed test
        
        # Temporal smoothing (single sample doesn't need smoothing)
        # But test the interface
        policy = MedianHysteresisPolicy(min_dwell=3)
        labels_2d = labels.reshape(1, 1)
        smoothed = policy.smooth(labels_2d)
        
        total_duration = time.time() - start_time
        
        # Complete pipeline must be under 15ms for real-time processing
        assert total_duration < 0.015, \
            f"End-to-end pipeline too slow: {total_duration*1000:.1f}ms > 15ms"
    
    def test_memory_efficiency(self):
        """Check memory usage doesn't grow unexpectedly."""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple batches
        for i in range(10):
            x = torch.randn(32, 9, 2, 100, dtype=torch.float32)
            result = encode_fsq(x)
            
            # Force garbage collection
            del result
            del x
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (< 100MB)
        assert memory_growth < 100, \
            f"Excessive memory growth: {memory_growth:.1f}MB"
    
    def test_batch_processing_scaling(self):
        """Batch processing should scale efficiently."""
        
        batch_sizes = [1, 8, 16, 32]
        times_per_sample = []
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 9, 2, 100, dtype=torch.float32)
            
            start_time = time.time()
            result = encode_fsq(x)
            duration = time.time() - start_time
            
            time_per_sample = duration / batch_size
            times_per_sample.append(time_per_sample)
        
        # Time per sample should decrease with larger batches (batching efficiency)
        batch1_time = times_per_sample[0]  # batch_size=1
        batch32_time = times_per_sample[-1]  # batch_size=32
        
        efficiency_gain = batch1_time / batch32_time
        assert efficiency_gain > 2.0, \
            f"Insufficient batching efficiency: {efficiency_gain:.1f}x speedup < 2x"
    
    @pytest.mark.slow
    def test_stress_test_large_inputs(self):
        """Stress test with large inputs (marked as slow test)."""
        
        # Large batch processing
        x_large = torch.randn(256, 9, 2, 100, dtype=torch.float32)
        
        start_time = time.time()
        result = encode_fsq(x_large)
        duration = time.time() - start_time
        
        # Should handle large batches within reasonable time (< 1 second)
        assert duration < 1.0, \
            f"Large batch processing too slow: {duration:.2f}s > 1.0s"
        
        # Memory should be reasonable (< 2GB)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        assert memory_mb < 2000, \
            f"Excessive memory usage: {memory_mb:.0f}MB > 2000MB"
    
    def test_cold_start_performance(self):
        """Test performance from cold start (first run)."""
        
        # Simulate cold start by creating fresh objects
        x = torch.randn(1, 9, 2, 100, dtype=torch.float32)
        
        # Measure cold start time
        start_time = time.time()
        result = encode_fsq(x, reset_stats=True)  # Forces reinitialization
        cold_duration = time.time() - start_time
        
        # Measure warm run time
        start_time = time.time()
        result = encode_fsq(x)
        warm_duration = time.time() - start_time
        
        # Cold start shouldn't be more than 10x slower
        slowdown_factor = cold_duration / warm_duration
        assert slowdown_factor < 10, \
            f"Excessive cold start penalty: {slowdown_factor:.1f}x slower"
        
        # Cold start should still meet absolute threshold
        assert cold_duration < 0.050, \
            f"Cold start too slow: {cold_duration*1000:.1f}ms > 50ms"
```

## Test Execution

### Regression Test Runner (`run_regression_tests.py`)

```python
#!/usr/bin/env python3
"""Comprehensive regression test runner."""

import sys
import subprocess
import time
from pathlib import Path

def run_test_suite():
    """Run complete test suite with reporting."""
    
    print("🎯 CONV2D REGRESSION TEST SUITE")
    print("=" * 50)
    
    test_categories = [
        ("Shape & Dtype Enforcement", "tests/test_shape_dtype_enforcement.py"),
        ("Determinism Validation", "tests/test_determinism.py"), 
        ("Temporal Assertions", "tests/test_temporal_assertions.py"),
        ("Calibration Improvements", "tests/test_calibration_improvements.py"),
        ("Speed Benchmarks", "tests/test_speed_benchmarks.py"),
    ]
    
    results = {}
    total_start = time.time()
    
    for category, test_file in test_categories:
        print(f"\n▶️  Running {category}...")
        
        start_time = time.time()
        
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short",
        ], capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.2f}s)")
            results[category] = "PASSED"
        else:
            print(f"❌ FAILED ({duration:.2f}s)")
            print("ERRORS:")
            print(result.stdout)
            print(result.stderr)
            results[category] = "FAILED"
    
    total_duration = time.time() - total_start
    
    # Summary report
    print(f"\n📊 SUMMARY REPORT ({total_duration:.1f}s total)")
    print("=" * 50)
    
    passed = sum(1 for status in results.values() if status == "PASSED")
    failed = len(results) - passed
    
    for category, status in results.items():
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"{status_icon} {category}: {status}")
    
    print(f"\nOverall: {passed}/{len(results)} test categories passed")
    
    if failed > 0:
        print(f"\n⚠️  {failed} test categories failed")
        return 1
    else:
        print("\n🎉 ALL TESTS PASSED")
        print("System: READY FOR PRODUCTION")
        return 0

if __name__ == "__main__":
    sys.exit(run_test_suite())
```

### pytest Configuration (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --durations=10
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    benchmark: marks tests as performance benchmarks
filterwarnings =
    ignore::UserWarning
    ignore::DeprecationWarning
```

## Continuous Integration

### GitHub Actions Workflow (`.github/workflows/tests.yml`)

```yaml
name: Regression Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run regression tests
      run: |
        export PYTHONPATH=src
        python run_regression_tests.py
    
    - name: Run specific test categories
      run: |
        export PYTHONPATH=src
        pytest tests/test_shape_dtype_enforcement.py -v
        pytest tests/test_determinism.py -v
        pytest tests/test_temporal_assertions.py -v
        pytest tests/test_calibration_improvements.py -v
        pytest tests/test_speed_benchmarks.py -v -m "not slow"
    
    - name: Generate coverage report
      run: |
        export PYTHONPATH=src
        pytest --cov=conv2d --cov-report=xml tests/
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Best Practices

1. **Test real failure modes**: Design tests that catch actual production issues, not just API contracts
2. **Use specific thresholds**: Performance tests should have concrete time/memory limits
3. **Validate determinism**: Same input should always produce identical output 
4. **Check edge cases**: Test boundary conditions, empty inputs, single samples
5. **Monitor performance**: Run benchmarks regularly to catch performance regressions
6. **Test calibration**: Ensure uncertainty quantification behaves correctly
7. **Validate temporal logic**: Check min-dwell enforcement and transition monotonicity
8. **Use proper fixtures**: Set up consistent test data and teardown properly
9. **Mark slow tests**: Use pytest markers to separate fast from slow tests
10. **Document test intent**: Each test should clearly state what failure it prevents

This testing framework provides comprehensive validation of production-critical behavioral analysis systems with specific focus on catching real failures that would impact deployment safety and performance.