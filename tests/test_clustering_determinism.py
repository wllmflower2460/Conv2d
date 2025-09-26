#!/usr/bin/env python3
"""Clustering determinism tests - fixed seed + K ⇒ identical labels with Hungarian matching."""

import pytest
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import adjusted_rand_score

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conv2d.clustering.kmeans import KMeansClusterer
from conv2d.clustering.gmm import GMMClusterer


class TestClusteringDeterminism:
    """Test clustering deterministic behavior - critical for reproducible behavioral analysis."""
    
    def test_kmeans_fixed_seed_identical_labels(self):
        """Fixed seed should produce identical labels."""
        
        # Create deterministic features
        np.random.seed(42)
        features = np.random.randn(1000, 64).astype(np.float32)
        
        # Multiple runs with same seed
        clusterer1 = KMeansClusterer(random_state=42)
        labels1 = clusterer1.fit_predict(features, k=4)
        
        clusterer2 = KMeansClusterer(random_state=42)
        labels2 = clusterer2.fit_predict(features, k=4)
        
        clusterer3 = KMeansClusterer(random_state=42)
        labels3 = clusterer3.fit_predict(features, k=4)
        
        # Labels must be identical with same seed
        assert np.array_equal(labels1, labels2), \
            "K-means not deterministic with same seed"
        assert np.array_equal(labels2, labels3), \
            "K-means not deterministic across multiple runs"
    
    def test_gmm_fixed_seed_identical_labels(self):
        """GMM with fixed seed should produce identical labels."""
        
        np.random.seed(42)
        features = np.random.randn(1000, 64).astype(np.float32)
        
        # Multiple runs with same seed
        clusterer1 = GMMClusterer(random_state=42)
        labels1 = clusterer1.fit_predict(features, k=4)
        
        clusterer2 = GMMClusterer(random_state=42)
        labels2 = clusterer2.fit_predict(features, k=4)
        
        # Labels must be identical with same seed
        assert np.array_equal(labels1, labels2), \
            "GMM not deterministic with same seed"
    
    def test_hungarian_matching_stability(self):
        """Hungarian matching should improve label stability across different seeds."""
        
        np.random.seed(42)
        features = np.random.randn(1000, 64).astype(np.float32)
        
        # First clustering run
        clusterer1 = KMeansClusterer(random_state=42)
        labels_reference = clusterer1.fit_predict(features, k=4)
        
        # Second run with different seed but Hungarian matching
        clusterer2 = KMeansClusterer(random_state=123)
        labels_matched = clusterer2.fit_predict(features, k=4, prior_labels=labels_reference)
        
        # Third run with different seed, no Hungarian matching
        clusterer3 = KMeansClusterer(random_state=123)
        labels_unmatched = clusterer3.fit_predict(features, k=4)
        
        # Hungarian matching should improve agreement
        agreement_matched = np.mean(labels_reference == labels_matched)
        agreement_unmatched = np.mean(labels_reference == labels_unmatched)
        
        assert agreement_matched >= agreement_unmatched, \
            f"Hungarian matching didn't improve agreement: {agreement_matched:.1%} vs {agreement_unmatched:.1%}"
        
        # With Hungarian matching, agreement should be high
        assert agreement_matched > 0.8, \
            f"Hungarian matching produced low agreement: {agreement_matched:.1%}"
    
    def test_hungarian_matching_ari_improvement(self):
        """Hungarian matching should improve ARI scores."""
        
        np.random.seed(42)
        features = np.random.randn(1000, 64).astype(np.float32)
        
        # Reference clustering
        clusterer_ref = GMMClusterer(random_state=42)
        labels_ref = clusterer_ref.fit_predict(features, k=4)
        
        # Different seed clustering with Hungarian matching
        clusterer_matched = GMMClusterer(random_state=456)
        labels_matched = clusterer_matched.fit_predict(features, k=4, prior_labels=labels_ref)
        
        # Different seed clustering without Hungarian matching
        clusterer_raw = GMMClusterer(random_state=456)
        labels_raw = clusterer_raw.fit_predict(features, k=4)
        
        # Compute ARI scores
        ari_matched = adjusted_rand_score(labels_ref, labels_matched)
        ari_raw = adjusted_rand_score(labels_ref, labels_raw)
        
        assert ari_matched >= ari_raw, \
            f"Hungarian matching reduced ARI: {ari_matched:.3f} vs {ari_raw:.3f}"
        
        # Matched ARI should be reasonably high
        assert ari_matched > 0.5, \
            f"Low ARI despite Hungarian matching: {ari_matched:.3f}"
    
    def test_cluster_count_consistency(self):
        """Number of clusters should be consistent across runs."""
        
        np.random.seed(42)
        features = np.random.randn(500, 64).astype(np.float32)
        
        for k in [3, 4, 5, 8]:
            # Multiple runs with same k
            clusterer1 = KMeansClusterer(random_state=42)
            labels1 = clusterer1.fit_predict(features, k=k)
            
            clusterer2 = KMeansClusterer(random_state=42)
            labels2 = clusterer2.fit_predict(features, k=k)
            
            # Should produce exactly k clusters
            unique1 = len(np.unique(labels1))
            unique2 = len(np.unique(labels2))
            
            assert unique1 == k, f"K-means produced {unique1} clusters, expected {k}"
            assert unique2 == k, f"K-means produced {unique2} clusters, expected {k}"
            assert unique1 == unique2, f"Inconsistent cluster count: {unique1} vs {unique2}"
    
    def test_min_support_merging(self):
        """Clusters below minimum support should be merged consistently."""
        
        np.random.seed(42)
        features = np.random.randn(1000, 64).astype(np.float32)
        
        # Use high k to force some small clusters
        clusterer1 = KMeansClusterer(random_state=42)
        labels1 = clusterer1.fit_predict(features, k=15)  # Many clusters
        
        # Apply min-support merging
        merged_labels1, merge_ops1 = clusterer1.merge_small_clusters(
            labels1, features, min_support=0.05  # 5% minimum
        )
        
        # Second run should be identical
        clusterer2 = KMeansClusterer(random_state=42)
        labels2 = clusterer2.fit_predict(features, k=15)
        merged_labels2, merge_ops2 = clusterer2.merge_small_clusters(
            labels2, features, min_support=0.05
        )
        
        # Merged results should be identical
        assert np.array_equal(merged_labels1, merged_labels2), \
            "Min-support merging not deterministic"
        
        # Check that small clusters were actually merged
        final_cluster_count = len(np.unique(merged_labels1))
        assert final_cluster_count < 15, \
            "Min-support merging didn't reduce cluster count"
        
        # Verify minimum support constraint
        for cluster_id in np.unique(merged_labels1):
            cluster_size = np.sum(merged_labels1 == cluster_id)
            support = cluster_size / len(merged_labels1)
            assert support >= 0.05, \
                f"Cluster {cluster_id} has {support:.1%} support < 5% minimum"
    
    def test_different_k_values_deterministic(self):
        """Different k values should produce deterministic results."""
        
        np.random.seed(42)
        features = np.random.randn(800, 64).astype(np.float32)
        
        k_values = [3, 4, 5, 6, 8, 12]
        
        for k in k_values:
            # Two runs with same k
            clusterer1 = GMMClusterer(random_state=42)
            labels1 = clusterer1.fit_predict(features, k=k)
            
            clusterer2 = GMMClusterer(random_state=42)
            labels2 = clusterer2.fit_predict(features, k=k)
            
            assert np.array_equal(labels1, labels2), \
                f"GMM not deterministic for k={k}"
            
            # Check cluster count
            actual_k = len(np.unique(labels1))
            assert actual_k == k, \
                f"GMM produced {actual_k} clusters, expected {k}"
    
    def test_feature_dimensionality_independence(self):
        """Clustering should be deterministic across different feature dimensions."""
        
        np.random.seed(42)
        
        # Test different feature dimensions
        dims = [32, 64, 128, 256]
        
        for dim in dims:
            features = np.random.randn(500, dim).astype(np.float32)
            
            clusterer1 = KMeansClusterer(random_state=42)
            labels1 = clusterer1.fit_predict(features, k=4)
            
            clusterer2 = KMeansClusterer(random_state=42)
            labels2 = clusterer2.fit_predict(features, k=4)
            
            assert np.array_equal(labels1, labels2), \
                f"K-means not deterministic for {dim}D features"
    
    def test_clustering_with_outliers(self):
        """Clustering should be deterministic even with outliers."""
        
        np.random.seed(42)
        # Create features with some outliers
        features = np.random.randn(800, 64).astype(np.float32)
        
        # Add outliers
        features[:10] *= 10.0  # 10 outliers
        
        # Multiple runs
        clusterer1 = KMeansClusterer(random_state=42)
        labels1 = clusterer1.fit_predict(features, k=4)
        
        clusterer2 = KMeansClusterer(random_state=42)
        labels2 = clusterer2.fit_predict(features, k=4)
        
        assert np.array_equal(labels1, labels2), \
            "Clustering not deterministic with outliers"
    
    def test_empty_cluster_handling(self):
        """Clustering should handle empty clusters deterministically."""
        
        np.random.seed(42)
        # Create very clustered data (might lead to empty clusters)
        centers = np.array([[0, 0], [10, 10], [20, 20]])
        features = []
        
        for center in centers:
            cluster_data = np.random.multivariate_normal(center, np.eye(2), 100)
            features.append(cluster_data)
        
        features = np.vstack(features).astype(np.float32)
        
        # Try to create more clusters than natural groups
        clusterer1 = KMeansClusterer(random_state=42)
        labels1 = clusterer1.fit_predict(features, k=8)  # More clusters than groups
        
        clusterer2 = KMeansClusterer(random_state=42)
        labels2 = clusterer2.fit_predict(features, k=8)
        
        assert np.array_equal(labels1, labels2), \
            "K-means not deterministic when handling empty clusters"
    
    def test_label_range_validation(self):
        """Cluster labels should be in valid range [0, k-1]."""
        
        np.random.seed(42)
        features = np.random.randn(500, 64).astype(np.float32)
        
        for k in [3, 4, 5, 8]:
            clusterer = KMeansClusterer(random_state=42)
            labels = clusterer.fit_predict(features, k=k)
            
            # Check label range
            assert labels.min() >= 0, f"Labels contain negative values: {labels.min()}"
            assert labels.max() < k, f"Labels exceed k-1: max={labels.max()}, k={k}"
            
            # Check data types
            assert labels.dtype == np.int32, f"Labels not int32: {labels.dtype}"
    
    def test_clustering_audit_trail(self):
        """Clustering should produce consistent audit information."""
        
        np.random.seed(42)
        features = np.random.randn(500, 64).astype(np.float32)
        
        clusterer1 = GMMClusterer(random_state=42)
        labels1 = clusterer1.fit_predict(features, k=4)
        
        clusterer2 = GMMClusterer(random_state=42)
        labels2 = clusterer2.fit_predict(features, k=4)
        
        # Labels should be identical
        assert np.array_equal(labels1, labels2), \
            "GMM clustering not reproducible"
        
        # Check that clusterer provides audit info
        assert hasattr(clusterer1, 'model'), "Clusterer missing model attribute"
        
        # BIC scores should be identical for same data/seed
        if hasattr(clusterer1.model, 'bic'):
            bic1 = clusterer1.model.bic(features)
            bic2 = clusterer2.model.bic(features)
            assert abs(bic1 - bic2) < 1e-10, f"BIC scores differ: {bic1} vs {bic2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])