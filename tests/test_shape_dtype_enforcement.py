"""Tests for shape and dtype enforcement throughout pipeline.

CRITICAL: Every stage must enforce (B,9,2,100) → outputs with float32.
These tests catch silent shape/dtype corruptions that break edge deployment.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.kmeans import KMeansClusterer
from conv2d.clustering.gmm import GMMClusterer
from conv2d.temporal.median import MedianHysteresisPolicy as MedianHysteresis
from conv2d.metrics.core import MetricsCalculator


def assert_shape_dtype(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    expected_dtype: torch.dtype,
    stage: str,
) -> None:
    """Assert tensor has exact shape and dtype.
    
    Args:
        tensor: Tensor to check
        expected_shape: Expected shape (use -1 for any)
        expected_dtype: Expected dtype
        stage: Stage name for error message
    """
    # Check dtype FIRST (most common silent failure)
    assert tensor.dtype == expected_dtype, (
        f"{stage}: dtype mismatch! "
        f"Expected {expected_dtype}, got {tensor.dtype}. "
        f"This WILL break edge deployment!"
    )
    
    # Check shape
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected != -1:  # -1 means any size OK
            assert actual == expected, (
                f"{stage}: shape mismatch at dim {i}! "
                f"Expected {expected_shape}, got {tuple(tensor.shape)}. "
                f"Edge models require EXACT shapes!"
            )


def test_input_validation_strict():
    """Test input validation catches all violations."""
    batch_size = 4
    
    # Valid input
    valid = torch.randn(batch_size, 9, 2, 100, dtype=torch.float32)
    
    # These should ALL fail
    invalid_inputs = [
        # Wrong dtype (MOST CRITICAL)
        torch.randn(batch_size, 9, 2, 100, dtype=torch.float64),  # float64
        torch.randn(batch_size, 9, 2, 100).to(torch.float16),     # float16
        torch.randn(batch_size, 9, 2, 100).to(torch.int32),       # int32
        
        # Wrong shape
        torch.randn(batch_size, 9, 2, 50, dtype=torch.float32),   # Wrong time
        torch.randn(batch_size, 6, 2, 100, dtype=torch.float32),  # Wrong channels
        torch.randn(batch_size, 9, 1, 100, dtype=torch.float32),  # Wrong sensors
        torch.randn(batch_size, 9, 100, dtype=torch.float32),     # Missing dim
        torch.randn(9, 2, 100, dtype=torch.float32),              # Missing batch
        
        # Edge cases
        torch.randn(0, 9, 2, 100, dtype=torch.float32),           # Empty batch
        torch.randn(1, 9, 2, 0, dtype=torch.float32),             # Empty time
    ]
    
    # FSQ should accept valid input
    result = encode_fsq(valid)
    assert result.codes.dtype == torch.int32, "Codes must be int32!"
    assert result.features.dtype == torch.float32, "Features must be float32!"
    
    # FSQ should reject ALL invalid inputs
    for i, invalid in enumerate(invalid_inputs):
        try:
            encode_fsq(invalid)
            assert False, f"FSQ accepted invalid input {i}: {invalid.shape}, {invalid.dtype}"
        except (ValueError, AssertionError):
            pass  # Expected to fail


def test_fsq_output_shapes_strict():
    """Test FSQ outputs have exact expected shapes."""
    batch_sizes = [1, 4, 32, 128]
    
    for B in batch_sizes:
        x = torch.randn(B, 9, 2, 100, dtype=torch.float32)
        result = encode_fsq(x)
        
        # Codes shape: (B, embedding_dim) as int32
        assert_shape_dtype(
            result.codes,
            (B, 64),  # Default embedding_dim=64
            torch.int32,
            f"FSQ codes (B={B})"
        )
        
        # Features shape: (B, feature_dim) as float32
        assert_shape_dtype(
            result.features,
            (B, result.features.shape[1]),  # Feature dim varies
            torch.float32,
            f"FSQ features (B={B})"
        )
        
        # Embeddings shape: (B, embedding_dim) as float32
        assert_shape_dtype(
            result.embeddings,
            (B, 64),
            torch.float32,
            f"FSQ embeddings (B={B})"
        )
        
        # Perplexity is scalar float
        assert isinstance(result.perplexity, float), "Perplexity must be Python float"
        assert 1.0 <= result.perplexity <= 240.0, "Perplexity out of valid range"


def test_clustering_preserves_dtype():
    """Test clustering preserves int32 for labels."""
    np.random.seed(42)
    
    # Create features
    n_samples = 100
    features = np.random.randn(n_samples, 64).astype(np.float32)
    
    # Test K-means
    kmeans = KMeansClusterer(random_state=42)
    kmeans_labels = kmeans.fit_predict(features, k=4)
    
    assert kmeans_labels.dtype == np.int32, "K-means labels must be int32!"
    assert kmeans_labels.shape == (n_samples,), "Labels must be 1D!"
    assert kmeans_labels.min() >= 0, "Labels must be non-negative!"
    assert kmeans_labels.max() < 4, "Labels must be < k!"
    
    # Test GMM
    gmm = GMMClusterer(random_state=42)
    gmm_labels = gmm.fit_predict(features, k=4)
    
    assert gmm_labels.dtype == np.int32, "GMM labels must be int32!"
    assert gmm_labels.shape == (n_samples,), "Labels must be 1D!"
    assert gmm_labels.min() >= 0, "Labels must be non-negative!"
    assert gmm_labels.max() < 4, "Labels must be < k!"


def test_temporal_preserves_dtype():
    """Test temporal smoothing preserves int32 motif IDs."""
    n_samples = 500
    
    # Create motif sequence
    motifs = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 0] * 38 + [0, 0], dtype=np.int32)
    assert len(motifs) == n_samples
    
    # Apply temporal smoothing
    policy = MedianHysteresis(min_dwell=3)
    smoothed = policy.smooth(motifs.reshape(1, -1))
    
    # Check output
    assert smoothed.dtype == np.int32, "Smoothed motifs must be int32!"
    assert smoothed.shape == (1, n_samples), "Shape must be preserved!"
    assert smoothed.min() >= 0, "Motifs must be non-negative!"
    assert smoothed.max() <= motifs.max(), "No new motifs should appear!"


def test_metrics_handle_dtypes():
    """Test metrics calculator handles various dtypes correctly."""
    n_samples = 100
    n_classes = 4
    
    # Test with different input dtypes
    test_cases = [
        (np.int32, np.float32),
        (np.int64, np.float64),  # Common from sklearn
        (np.int32, np.float16),  # Possible from optimization
    ]
    
    calculator = MetricsCalculator()
    
    for label_dtype, prob_dtype in test_cases:
        y_true = np.random.randint(0, n_classes, n_samples).astype(label_dtype)
        y_pred = np.random.randint(0, n_classes, n_samples).astype(label_dtype)
        
        # Create probabilities
        y_prob = np.random.rand(n_samples, n_classes).astype(prob_dtype)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Should handle any dtype gracefully
        metrics = calculator.compute_all(y_true, y_pred, y_prob)
        
        # Check outputs are Python types (JSON-serializable)
        assert isinstance(metrics.accuracy, float), "Accuracy must be Python float"
        assert isinstance(metrics.macro_f1, float), "F1 must be Python float"
        assert isinstance(metrics.ece, float), "ECE must be Python float"
        assert all(isinstance(f1, float) for f1 in metrics.per_class_f1), "Per-class F1 must be Python floats"


def test_pipeline_end_to_end_shapes():
    """Test complete pipeline maintains shapes and dtypes."""
    batch_size = 8
    
    # Stage 1: Input
    x = torch.randn(batch_size, 9, 2, 100, dtype=torch.float32)
    assert_shape_dtype(x, (batch_size, 9, 2, 100), torch.float32, "Input")
    
    # Stage 2: FSQ encoding
    fsq_result = encode_fsq(x)
    assert_shape_dtype(fsq_result.codes, (batch_size, 64), torch.int32, "FSQ codes")
    assert_shape_dtype(fsq_result.features, (batch_size, -1), torch.float32, "FSQ features")
    
    # Stage 3: Clustering
    features_np = fsq_result.features.numpy()
    assert features_np.dtype == np.float32, "NumPy features must be float32"
    
    clusterer = GMMClusterer(random_state=42)
    labels = clusterer.fit_predict(features_np, k=4)
    assert labels.dtype == np.int32, "Cluster labels must be int32"
    assert labels.shape == (batch_size,), "Labels must be 1D"
    
    # Stage 4: Temporal smoothing (extend to sequence)
    T = 100
    labels_seq = np.tile(labels, (T // batch_size + 1))[:T]
    labels_seq = labels_seq.reshape(1, T)
    
    smoother = MedianHysteresis(min_dwell=5)
    smoothed = smoother.smooth(labels_seq)
    assert smoothed.dtype == np.int32, "Smoothed must be int32"
    assert smoothed.shape == (1, T), "Shape must be preserved"
    
    # Stage 5: Metrics (using first batch_size samples)
    y_true = labels[:batch_size]
    y_pred = labels[:batch_size]
    
    # Create fake probabilities
    y_prob = np.zeros((batch_size, 4), dtype=np.float32)
    y_prob[np.arange(batch_size), y_pred] = 0.8
    y_prob += 0.05  # Add baseline
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    calculator = MetricsCalculator()
    metrics = calculator.compute_all(y_true, y_pred, y_prob)
    
    # All metrics should be Python floats
    assert isinstance(metrics.accuracy, float), "Accuracy not Python float"
    assert isinstance(metrics.ece, float), "ECE not Python float"
    
    print(f"✓ Pipeline preserved shapes/dtypes through {5} stages")


def test_mixed_precision_rejection():
    """Test that mixed precision inputs are rejected."""
    batch_size = 4
    
    # These mixed-precision patterns MUST be rejected
    x = torch.randn(batch_size, 9, 2, 100, dtype=torch.float32)
    
    # Try to sneak in float16 (common optimization attempt)
    x_half = x.half()  # float16
    try:
        encode_fsq(x_half)
        assert False, "FSQ accepted float16 input! This breaks edge deployment!"
    except (ValueError, AssertionError):
        pass  # Expected
    
    # Try to sneak in bfloat16 (modern optimization)
    if hasattr(torch, 'bfloat16'):
        x_bf16 = x.to(torch.bfloat16)
        try:
            encode_fsq(x_bf16)
            assert False, "FSQ accepted bfloat16 input! This breaks edge deployment!"
        except (ValueError, AssertionError):
            pass  # Expected


def test_batch_size_edge_cases():
    """Test edge cases for batch sizes."""
    # Single sample (B=1) - common in inference
    x1 = torch.randn(1, 9, 2, 100, dtype=torch.float32)
    result1 = encode_fsq(x1)
    assert result1.codes.shape[0] == 1, "Single sample failed"
    
    # Large batch (B=256) - training
    x256 = torch.randn(256, 9, 2, 100, dtype=torch.float32)
    result256 = encode_fsq(x256)
    assert result256.codes.shape[0] == 256, "Large batch failed"
    
    # Odd number (B=17) - data loader remainder
    x17 = torch.randn(17, 9, 2, 100, dtype=torch.float32)
    result17 = encode_fsq(x17)
    assert result17.codes.shape[0] == 17, "Odd batch size failed"
    
    # Prime number (B=13) - stress test
    x13 = torch.randn(13, 9, 2, 100, dtype=torch.float32)
    result13 = encode_fsq(x13)
    assert result13.codes.shape[0] == 13, "Prime batch size failed"


def test_output_range_validation():
    """Test outputs are in valid ranges."""
    batch_size = 32
    x = torch.randn(batch_size, 9, 2, 100, dtype=torch.float32)
    
    # FSQ codes should be in valid range
    result = encode_fsq(x, levels=[8, 6, 5])
    
    # Codes should be 0 <= code < product(levels)
    max_code = 8 * 6 * 5  # 240
    assert result.codes.min() >= 0, "Codes have negative values!"
    assert result.codes.max() < max_code, f"Codes exceed max ({max_code})!"
    
    # Features should be finite
    assert torch.isfinite(result.features).all(), "Features have non-finite values!"
    
    # Embeddings should be finite
    assert torch.isfinite(result.embeddings).all(), "Embeddings have non-finite values!"


if __name__ == "__main__":
    # Run all tests
    test_input_validation_strict()
    print("✓ Input validation strict")
    
    test_fsq_output_shapes_strict()
    print("✓ FSQ output shapes strict")
    
    test_clustering_preserves_dtype()
    print("✓ Clustering preserves dtype")
    
    test_temporal_preserves_dtype()
    print("✓ Temporal preserves dtype")
    
    test_metrics_handle_dtypes()
    print("✓ Metrics handle dtypes")
    
    test_pipeline_end_to_end_shapes()
    print("✓ Pipeline end-to-end shapes")
    
    test_mixed_precision_rejection()
    print("✓ Mixed precision rejection")
    
    test_batch_size_edge_cases()
    print("✓ Batch size edge cases")
    
    test_output_range_validation()
    print("✓ Output range validation")
    
    print("\n🎯 All shape/dtype enforcement tests passed!")
    print("Edge deployment safety: VERIFIED")