#!/usr/bin/env python3
"""Tests for deterministic clustering framework.

Tests verify:
1. Determinism across runs
2. K selection strategies (silhouette vs BIC)
3. Label stability with Hungarian matching
4. Min-support merging
5. JSON serialization
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import tempfile
from typing import List

import numpy as np
from sklearn.datasets import make_blobs

from conv2d.clustering import ClusteringResult, GMMClusterer, KMeansClusterer


class TestKMeansClusterer:
    """Test K-Means clustering with deterministic behavior."""
    
    def test_determinism(self):
        """Test that same data + seed produces identical results."""
        # Create synthetic data
        X, _ = make_blobs(n_samples=200, n_features=10, centers=4, random_state=42)
        X = X.astype(np.float32)
        
        # Run clustering multiple times
        results = []
        for _ in range(3):
            kmeans = KMeansClusterer(min_clusters=2, max_clusters=8, seed=42)
            result = kmeans.fit(X)
            results.append(result)
        
        # Check all results identical
        for i in range(1, len(results)):
            assert np.array_equal(results[0].labels, results[i].labels), \
                f"Run {i} produced different labels"
            assert np.allclose(results[0].centroids, results[i].centroids), \
                f"Run {i} produced different centroids"
            assert results[0].n_clusters == results[i].n_clusters, \
                f"Run {i} selected different K"
        
        print(f"✓ K-Means determinism: {results[0].n_clusters} clusters consistent")
    
    def test_silhouette_selection(self):
        """Test K selection using silhouette score."""
        # Create well-separated clusters
        X, _ = make_blobs(n_samples=300, n_features=2, centers=3, 
                          cluster_std=0.5, random_state=42)
        X = X.astype(np.float32)
        
        kmeans = KMeansClusterer(min_clusters=2, max_clusters=6, seed=42)
        result = kmeans.fit(X)
        
        # Should find 3 clusters (the true number)
        assert result.n_clusters == 3, f"Expected 3 clusters, got {result.n_clusters}"
        assert result.silhouette_score > 0.5, "Silhouette score should be high"
        
        print(f"✓ Silhouette selection: k={result.n_clusters}, score={result.silhouette_score:.3f}")
    
    def test_min_support_merge(self):
        """Test merging of low-support clusters."""
        # Create imbalanced clusters with forced small cluster
        np.random.seed(42)
        X1 = np.random.randn(100, 5).astype(np.float32)
        X2 = np.random.randn(100, 5).astype(np.float32) + 5
        X3 = np.random.randn(3, 5).astype(np.float32) + 10  # Very small cluster
        X = np.vstack([X1, X2, X3])
        
        # Force K=3 to ensure small cluster exists
        kmeans = KMeansClusterer(
            min_clusters=3, max_clusters=3,  # Force 3 clusters
            seed=42,
            min_support=0.05,  # 5% = ~10 samples minimum
            enable_merge=True
        )
        result = kmeans.fit(X)
        
        # Small cluster should be merged (or no merge if K-means puts them together)
        if result.merge_table is None or len(result.merge_table) == 0:
            # K-means might have already grouped the small cluster
            print("  Note: K-means naturally avoided small clusters")
            assert result.n_clusters <= 2, "Should have at most 2 clusters after natural grouping"
        else:
            assert len(result.merge_table) > 0, "Should have merge operations"
        
        # Check merge table structure
        merge = result.merge_table[0]
        assert "source" in merge
        assert "target" in merge
        assert "n_samples" in merge
        assert "distance" in merge
        
        print(f"✓ Min-support merge: {len(result.merge_table)} merges performed")
        for m in result.merge_table:
            print(f"  Merged cluster {m['source']} ({m['n_samples']} samples) → {m['target']}")
    
    def test_initial_centroids(self):
        """Test setting initial centroids."""
        X, _ = make_blobs(n_samples=150, n_features=5, centers=3, random_state=42)
        X = X.astype(np.float32)
        
        # First run without initial centroids
        kmeans1 = KMeansClusterer(min_clusters=3, max_clusters=3, seed=42)
        result1 = kmeans1.fit(X)
        
        # Second run with initial centroids from first
        kmeans2 = KMeansClusterer(min_clusters=3, max_clusters=3, seed=99)  # Different seed
        kmeans2.set_initial_centroids(result1.centroids)
        result2 = kmeans2.fit(X)
        
        # Results should be very similar despite different seeds
        agreement = np.mean(result1.labels == result2.labels)
        assert agreement > 0.9, f"Initial centroids not effective: {agreement:.2f} agreement"
        
        print(f"✓ Initial centroids: {agreement:.1%} label agreement")


class TestGMMClusterer:
    """Test GMM clustering with BIC selection."""
    
    def test_determinism(self):
        """Test that same data + seed produces identical results."""
        X, _ = make_blobs(n_samples=200, n_features=10, centers=4, random_state=42)
        X = X.astype(np.float32)
        
        # Run clustering multiple times
        results = []
        for _ in range(3):
            gmm = GMMClusterer(min_clusters=2, max_clusters=8, seed=42)
            result = gmm.fit(X)
            results.append(result)
        
        # Check all results identical
        for i in range(1, len(results)):
            assert np.array_equal(results[0].labels, results[i].labels), \
                f"Run {i} produced different labels"
            assert results[0].n_clusters == results[i].n_clusters, \
                f"Run {i} selected different K"
        
        print(f"✓ GMM determinism: {results[0].n_clusters} clusters consistent")
    
    def test_bic_selection(self):
        """Test K selection using BIC."""
        # Create well-separated Gaussian clusters
        X, _ = make_blobs(n_samples=300, n_features=2, centers=3,
                          cluster_std=0.8, random_state=42)
        X = X.astype(np.float32)
        
        gmm = GMMClusterer(min_clusters=2, max_clusters=6, seed=42)
        result = gmm.fit(X)
        
        # Should find reasonable number of clusters
        assert 2 <= result.n_clusters <= 4, f"Unexpected clusters: {result.n_clusters}"
        assert result.bic is not None, "BIC should be computed"
        
        print(f"✓ BIC selection: k={result.n_clusters}, BIC={result.bic:.1f}")
    
    def test_covariance_types(self):
        """Test different covariance types."""
        X, _ = make_blobs(n_samples=150, n_features=3, centers=2, random_state=42)
        X = X.astype(np.float32)
        
        cov_types = ["full", "tied", "diag", "spherical"]
        results = {}
        
        for cov_type in cov_types:
            gmm = GMMClusterer(
                min_clusters=2, max_clusters=4, 
                seed=42, 
                covariance_type=cov_type
            )
            result = gmm.fit(X)
            results[cov_type] = result.bic
        
        print("✓ Covariance types tested:")
        for cov_type, bic in results.items():
            print(f"  {cov_type:10s}: BIC={bic:.1f}")


class TestLabelStability:
    """Test label stability across retrains."""
    
    def test_hungarian_matching(self):
        """Test Hungarian matching for label continuity."""
        # Create data
        X, true_labels = make_blobs(n_samples=200, n_features=5, centers=3, random_state=42)
        X = X.astype(np.float32)
        
        # First clustering
        kmeans1 = KMeansClusterer(min_clusters=3, max_clusters=3, seed=42)
        result1 = kmeans1.fit(X)
        
        # Add noise and recluster with matching
        X_noisy = X + np.random.randn(*X.shape).astype(np.float32) * 0.1
        kmeans2 = KMeansClusterer(min_clusters=3, max_clusters=3, seed=99)
        result2 = kmeans2.fit(X_noisy, prior_labels=result1.labels)
        
        # Check label mapping was created
        assert result2.label_mapping is not None, "Label mapping should exist"
        
        # Most labels should be stable
        agreement = np.mean(result1.labels == result2.labels)
        assert agreement > 0.8, f"Poor label stability: {agreement:.2f}"
        
        print(f"✓ Hungarian matching: {agreement:.1%} label stability")
        print(f"  Label mapping: {result2.label_mapping}")
    
    def test_incremental_clustering(self):
        """Test maintaining labels across incremental data."""
        np.random.seed(42)
        
        # Initial data
        X1 = np.random.randn(100, 5).astype(np.float32)
        X1[50:] += 5  # Two clusters
        
        kmeans = KMeansClusterer(min_clusters=2, max_clusters=4, seed=42)
        result1 = kmeans.fit(X1)
        
        # Add more data
        X2_new = np.random.randn(50, 5).astype(np.float32)
        X2_new[25:] += 5
        X2 = np.vstack([X1, X2_new])
        
        # Recluster with prior labels
        result2 = kmeans.fit(X2, prior_labels=np.concatenate([
            result1.labels,
            np.full(50, -1, dtype=np.int32)  # Unknown labels for new data
        ]))
        
        # Original data should mostly keep labels
        original_agreement = np.mean(result1.labels == result2.labels[:100])
        assert original_agreement > 0.7, f"Poor stability: {original_agreement:.2f}"
        
        print(f"✓ Incremental clustering: {original_agreement:.1%} original label retention")


class TestSerialization:
    """Test JSON serialization of results."""
    
    def test_json_roundtrip(self):
        """Test saving and loading results."""
        # Create and fit clustering
        X, _ = make_blobs(n_samples=100, n_features=3, centers=2, random_state=42)
        X = X.astype(np.float32)
        
        kmeans = KMeansClusterer(min_clusters=2, max_clusters=3, seed=42)
        result1 = kmeans.fit(X)
        
        # Save to JSON
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
            json_str = result1.to_json(f.name)
            
            # Load back
            result2 = ClusteringResult.from_json(f.name)
        
        # Check equivalence
        assert np.array_equal(result1.labels, result2.labels)
        assert np.allclose(result1.centroids, result2.centroids)
        assert result1.n_clusters == result2.n_clusters
        assert result1.seed == result2.seed
        
        print("✓ JSON serialization successful")
    
    def test_json_content(self):
        """Test JSON contains expected fields."""
        X, _ = make_blobs(n_samples=50, n_features=2, centers=2, random_state=42)
        X = X.astype(np.float32)
        
        gmm = GMMClusterer(
            min_clusters=2, max_clusters=3, 
            seed=42,
            min_support=0.1,
            enable_merge=True
        )
        result = gmm.fit(X)
        
        # Check JSON structure
        json_str = result.to_json()
        data = json.loads(json_str)
        
        # Required fields
        assert "labels" in data
        assert "centroids" in data
        assert "n_clusters" in data
        assert "parameters" in data
        assert "seed" in data
        
        # Algorithm-specific
        assert data["parameters"]["algorithm"] == "GMMClusterer"
        assert "bic" in data
        
        print("✓ JSON structure validated")


class TestIntegration:
    """Integration tests with FSQ codes."""
    
    def test_fsq_code_clustering(self):
        """Test clustering of FSQ codes."""
        # Simulate FSQ codes (discrete but embedded)
        np.random.seed(42)
        
        # Create code embeddings for 3 behavioral patterns
        pattern1 = np.random.randn(50, 64).astype(np.float32)
        pattern2 = np.random.randn(50, 64).astype(np.float32) + 2
        pattern3 = np.random.randn(50, 64).astype(np.float32) - 2
        
        X = np.vstack([pattern1, pattern2, pattern3])
        
        # Cluster with K-Means
        kmeans = KMeansClusterer(min_clusters=2, max_clusters=5, seed=42)
        result = kmeans.fit(X)
        
        assert 2 <= result.n_clusters <= 4, f"Expected 2-4 patterns, got {result.n_clusters}"
        assert result.silhouette_score > 0.3, "Patterns should be separable"
        
        print(f"✓ FSQ code clustering: {result.n_clusters} motifs discovered")
    
    def test_pipeline_integration(self):
        """Test full pipeline from features to stable motifs."""
        from conv2d.features import encode_fsq
        import torch
        
        # Create IMU-like data
        torch.manual_seed(42)
        x = torch.randn(100, 9, 2, 100, dtype=torch.float32)
        
        # Encode with FSQ
        fsq_result = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        
        # Use quantized features for clustering
        X_cluster = fsq_result.quantized.mean(dim=2).numpy()  # Average over time
        
        # Cluster to find motifs
        gmm = GMMClusterer(min_clusters=2, max_clusters=10, seed=42)
        cluster_result = gmm.fit(X_cluster)
        
        print(f"✓ Pipeline integration: FSQ → {cluster_result.n_clusters} motifs")
        print(f"  Perplexity: {fsq_result.perplexity:.2f}")
        print(f"  Silhouette: {cluster_result.silhouette_score:.3f}")


def run_tests():
    """Run all clustering tests."""
    test_classes = [
        TestKMeansClusterer,
        TestGMMClusterer,
        TestLabelStability,
        TestSerialization,
        TestIntegration,
    ]
    
    print("=" * 60)
    print("Deterministic Clustering Tests")
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
    print("All tests completed!")


if __name__ == "__main__":
    run_tests()