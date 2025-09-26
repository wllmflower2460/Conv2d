"""Tests for deterministic behavior across pipeline.

CRITICAL: Same input → identical outputs. Every time.
These tests catch non-deterministic behavior that breaks reproducibility.
"""

from __future__ import annotations

import numpy as np
import torch
import hashlib
import json
from typing import Any, Dict

from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.kmeans import KMeansClusterer
from conv2d.clustering.gmm import GMMClusterer
from conv2d.temporal.median import MedianHysteresisPolicy as MedianHysteresis


def hash_array(arr: np.ndarray) -> str:
    """Create hash of array for comparison."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def test_fsq_determinism():
    """Test FSQ encoding is perfectly deterministic."""
    batch_size = 32
    torch.manual_seed(42)
    
    # Create input
    x = torch.randn(batch_size, 9, 2, 100, dtype=torch.float32)
    
    # Encode multiple times
    results = []
    for _ in range(5):
        result = encode_fsq(x, reset_stats=True)
        results.append(result)
    
    # All codes must be IDENTICAL
    first_codes = results[0].codes
    for i, result in enumerate(results[1:], 1):
        assert torch.equal(first_codes, result.codes), (
            f"FSQ non-deterministic at iteration {i}! "
            f"This breaks reproducibility!"
        )
    
    # All features must be IDENTICAL
    first_features = results[0].features
    for i, result in enumerate(results[1:], 1):
        assert torch.allclose(first_features, result.features, atol=1e-7), (
            f"FSQ features non-deterministic at iteration {i}!"
        )
    
    # All embeddings must be IDENTICAL
    first_embeddings = results[0].embeddings
    for i, result in enumerate(results[1:], 1):
        assert torch.allclose(first_embeddings, result.embeddings, atol=1e-7), (
            f"FSQ embeddings non-deterministic at iteration {i}!"
        )
    
    print(f"✓ FSQ deterministic across {len(results)} runs")


def test_fsq_code_usage():
    """Test FSQ uses codes from all active levels."""
    batch_size = 1000  # Large batch to ensure coverage
    torch.manual_seed(42)
    
    # Test different level configurations
    level_configs = [
        [8, 6, 5],  # Standard (240 codes)
        [4, 4, 4, 4],  # Uniform (256 codes)  
        [16, 15],  # Two-level (240 codes)
    ]
    
    for levels in level_configs:
        x = torch.randn(batch_size, 9, 2, 100, dtype=torch.float32)
        result = encode_fsq(x, levels=levels)
        
        # Check code usage
        unique_codes = torch.unique(result.codes)
        max_possible = np.prod(levels)
        usage_ratio = len(unique_codes) / max_possible
        
        print(f"  Levels {levels}: {len(unique_codes)}/{max_possible} codes used ({usage_ratio:.1%})")
        
        # Should use at least 10% of codes in large batch
        assert usage_ratio > 0.10, (
            f"FSQ using only {usage_ratio:.1%} of codes! "
            f"Codebook underutilized for levels {levels}"
        )
        
        # Check perplexity indicates good usage
        assert result.perplexity > 1.0, "Perplexity too low (codebook collapse)"
        assert result.perplexity < max_possible, "Perplexity exceeds max codes"


def test_clustering_determinism_with_seed():
    """Test clustering is deterministic with fixed seed."""
    n_samples = 200
    n_features = 64
    
    # Create features
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Test K-means determinism
    kmeans_labels = []
    for _ in range(3):
        clusterer = KMeansClusterer(random_state=42)
        labels = clusterer.fit_predict(features, k=4)
        kmeans_labels.append(labels)
    
    # All K-means runs must produce identical labels
    first_kmeans = kmeans_labels[0]
    for i, labels in enumerate(kmeans_labels[1:], 1):
        assert np.array_equal(first_kmeans, labels), (
            f"K-means non-deterministic at run {i}!"
        )
    
    print(f"✓ K-means deterministic across {len(kmeans_labels)} runs")
    
    # Test GMM determinism
    gmm_labels = []
    for _ in range(3):
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(features, k=4)
        gmm_labels.append(labels)
    
    # All GMM runs must produce identical labels
    first_gmm = gmm_labels[0]
    for i, labels in enumerate(gmm_labels[1:], 1):
        assert np.array_equal(first_gmm, labels), (
            f"GMM non-deterministic at run {i}!"
        )
    
    print(f"✓ GMM deterministic across {len(gmm_labels)} runs")


def test_hungarian_matching_stability():
    """Test Hungarian matching produces stable label mappings."""
    n_samples = 100
    n_features = 64
    
    # Create features with clear clusters
    np.random.seed(42)
    features = []
    true_labels = []
    
    for i in range(4):
        cluster = np.random.randn(25, n_features) + i * 3
        features.append(cluster)
        true_labels.extend([i] * 25)
    
    features = np.vstack(features).astype(np.float32)
    true_labels = np.array(true_labels, dtype=np.int32)
    
    # First clustering
    clusterer1 = KMeansClusterer(random_state=42)
    labels1 = clusterer1.fit_predict(features, k=4)
    
    # Second clustering (different initialization)
    clusterer2 = KMeansClusterer(random_state=43)
    labels2_raw = clusterer2.fit_predict(features, k=4)
    
    # Apply Hungarian matching using prior
    labels2_matched = clusterer2.fit_predict(features, k=4, prior_labels=labels1)
    
    # Check matching improved alignment
    raw_agreement = np.mean(labels1 == labels2_raw)
    matched_agreement = np.mean(labels1 == labels2_matched)
    
    print(f"  Raw agreement: {raw_agreement:.1%}")
    print(f"  Matched agreement: {matched_agreement:.1%}")
    
    assert matched_agreement > raw_agreement, (
        "Hungarian matching didn't improve alignment!"
    )
    
    # Should have high agreement after matching
    assert matched_agreement > 0.8, (
        f"Hungarian matching poor: only {matched_agreement:.1%} agreement"
    )


def test_temporal_determinism():
    """Test temporal smoothing is deterministic."""
    T = 500
    
    # Create motif sequence
    np.random.seed(42)
    motifs = np.random.randint(0, 4, T, dtype=np.int32)
    
    # Apply smoothing multiple times
    smoother = MedianHysteresis(min_dwell=5, window_size=7)
    
    results = []
    for _ in range(3):
        smoothed = smoother.smooth(motifs.reshape(1, -1))
        results.append(smoothed)
    
    # All results must be identical
    first = results[0]
    for i, result in enumerate(results[1:], 1):
        assert np.array_equal(first, result), (
            f"Temporal smoothing non-deterministic at run {i}!"
        )
    
    print(f"✓ Temporal smoothing deterministic across {len(results)} runs")


def test_pipeline_determinism_end_to_end():
    """Test complete pipeline is deterministic."""
    # Fix all seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    def run_pipeline():
        """Run complete pipeline once."""
        # Input
        x = torch.randn(32, 9, 2, 100, dtype=torch.float32)
        
        # FSQ
        fsq_result = encode_fsq(x, reset_stats=True)
        
        # Clustering
        features = fsq_result.features.numpy()
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(features, k=4)
        
        # Temporal (simulate sequence)
        labels_seq = np.tile(labels, 10)[:300].reshape(1, -1)
        smoother = MedianHysteresis(min_dwell=5)
        smoothed = smoother.smooth(labels_seq)
        
        return {
            "codes": fsq_result.codes.numpy(),
            "labels": labels,
            "smoothed": smoothed,
        }
    
    # Run pipeline multiple times
    results = []
    for i in range(3):
        # Reset seeds for each run
        torch.manual_seed(42)
        np.random.seed(42)
        result = run_pipeline()
        results.append(result)
        
        # Create hash of results
        codes_hash = hash_array(result["codes"])
        labels_hash = hash_array(result["labels"])
        smoothed_hash = hash_array(result["smoothed"])
        
        print(f"  Run {i}: codes={codes_hash}, labels={labels_hash}, smoothed={smoothed_hash}")
    
    # All runs must produce identical results
    first = results[0]
    for i, result in enumerate(results[1:], 1):
        assert np.array_equal(first["codes"], result["codes"]), f"Codes differ at run {i}"
        assert np.array_equal(first["labels"], result["labels"]), f"Labels differ at run {i}"
        assert np.array_equal(first["smoothed"], result["smoothed"]), f"Smoothed differ at run {i}"
    
    print(f"✓ Pipeline deterministic across {len(results)} runs")


def test_determinism_with_different_batch_sizes():
    """Test determinism is maintained across batch sizes."""
    torch.manual_seed(42)
    
    # Create large input
    x_large = torch.randn(64, 9, 2, 100, dtype=torch.float32)
    
    # Process as single batch
    result_single = encode_fsq(x_large, reset_stats=True)
    
    # Process in smaller batches (simulating data loader)
    batch_size = 16
    results_batched = []
    
    for i in range(0, 64, batch_size):
        batch = x_large[i:i+batch_size]
        result = encode_fsq(batch, reset_stats=(i==0))
        results_batched.append(result.codes)
    
    codes_batched = torch.cat(results_batched, dim=0)
    
    # Results should be identical
    assert torch.equal(result_single.codes, codes_batched), (
        "Different results with different batch sizes! "
        "This breaks distributed training!"
    )
    
    print("✓ Determinism maintained across batch sizes")


def test_config_hash_determinism():
    """Test configuration hashing is deterministic."""
    config = {
        "model": "conv2d_fsq",
        "levels": [8, 6, 5],
        "embedding_dim": 64,
        "batch_size": 32,
        "random_seed": 42,
    }
    
    # Hash config multiple times
    hashes = []
    for _ in range(5):
        # Serialize to JSON (sorted keys for determinism)
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        hashes.append(config_hash)
    
    # All hashes must be identical
    assert len(set(hashes)) == 1, "Config hashing non-deterministic!"
    
    print(f"✓ Config hash deterministic: {hashes[0]}")


def test_numpy_torch_consistency():
    """Test NumPy and PyTorch operations are consistent."""
    # Create data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # PyTorch operations
    x_torch = torch.randn(100, 64, dtype=torch.float32)
    mean_torch = x_torch.mean().item()
    std_torch = x_torch.std().item()
    
    # Convert to NumPy
    x_numpy = x_torch.numpy()
    mean_numpy = float(x_numpy.mean())
    std_numpy = float(x_numpy.std())
    
    # Should be very close (allowing for minor numerical differences)
    assert abs(mean_torch - mean_numpy) < 1e-6, "Mean inconsistent between PyTorch and NumPy"
    assert abs(std_torch - std_numpy) < 1e-4, "Std inconsistent between PyTorch and NumPy"
    
    print("✓ NumPy-PyTorch consistency verified")


if __name__ == "__main__":
    # Run all tests
    test_fsq_determinism()
    test_fsq_code_usage()
    test_clustering_determinism_with_seed()
    test_hungarian_matching_stability()
    test_temporal_determinism()
    test_pipeline_determinism_end_to_end()
    test_determinism_with_different_batch_sizes()
    test_config_hash_determinism()
    test_numpy_torch_consistency()
    
    print("\n🎯 All determinism tests passed!")
    print("Reproducibility: GUARANTEED")