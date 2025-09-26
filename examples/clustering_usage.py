#!/usr/bin/env python3
"""Examples demonstrating deterministic clustering usage.

This module shows how to:
1. Use strategy pattern for different clustering algorithms
2. Ensure deterministic results across runs
3. Maintain label stability across retrains
4. Apply min-support merging
5. Export results for audit trails
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import torch
from sklearn.datasets import make_blobs

from conv2d.clustering import KMeansClusterer, GMMClusterer, ClusteringResult
from conv2d.features import encode_fsq


def example_deterministic_kmeans():
    """Example 1: Deterministic K-Means clustering."""
    print("=" * 60)
    print("Example 1: Deterministic K-Means")
    print("=" * 60)
    
    # Create synthetic behavioral data
    np.random.seed(42)
    X, true_labels = make_blobs(
        n_samples=300, 
        n_features=64,  # Like FSQ embedding dim
        centers=4, 
        cluster_std=1.5,
        random_state=42
    )
    X = X.astype(np.float32)
    
    # Run clustering with deterministic settings
    kmeans = KMeansClusterer(
        min_clusters=2,
        max_clusters=10,
        seed=42,  # Fixed seed for reproducibility
        min_support=0.05,  # At least 5% of samples per cluster
        enable_merge=True,
    )
    
    result = kmeans.fit(X)
    
    print(f"Results:")
    print(f"  Optimal K: {result.n_clusters}")
    print(f"  Silhouette score: {result.silhouette_score:.3f}")
    print(f"  Inertia: {result.inertia:.1f}")
    print(f"  Seed used: {result.seed}")
    
    # Verify determinism
    result2 = kmeans.fit(X)
    deterministic = np.array_equal(result.labels, result2.labels)
    print(f"  Deterministic: {deterministic} ✓" if deterministic else f"  Deterministic: ✗")
    
    print()


def example_gmm_with_bic():
    """Example 2: GMM clustering with BIC selection."""
    print("=" * 60)
    print("Example 2: GMM with BIC Selection")
    print("=" * 60)
    
    # Create data with overlapping clusters
    np.random.seed(42)
    X1 = np.random.randn(100, 20).astype(np.float32) * 2
    X2 = np.random.randn(100, 20).astype(np.float32) + 3
    X3 = np.random.randn(100, 20).astype(np.float32) * 0.5 + 6
    X = np.vstack([X1, X2, X3])
    
    # Use GMM with BIC for model selection
    gmm = GMMClusterer(
        min_clusters=2,
        max_clusters=6,
        seed=42,
        covariance_type="full",  # Full covariance for flexibility
    )
    
    result = gmm.fit(X)
    
    print(f"Results:")
    print(f"  Optimal K (via BIC): {result.n_clusters}")
    print(f"  BIC: {result.bic:.1f}")
    print(f"  Silhouette score: {result.silhouette_score:.3f}")
    
    # Show component weights
    weights = result.parameters.get("weights", [])
    if weights:
        print(f"  Component weights: {[f'{w:.2f}' for w in weights]}")
    
    print()


def example_label_stability():
    """Example 3: Maintaining label stability across retrains."""
    print("=" * 60)
    print("Example 3: Label Stability Across Retrains")
    print("=" * 60)
    
    # Initial training data
    X1, _ = make_blobs(n_samples=200, n_features=10, centers=3, random_state=42)
    X1 = X1.astype(np.float32)
    
    # First clustering establishes label convention
    kmeans = KMeansClusterer(min_clusters=3, max_clusters=3, seed=42)
    result1 = kmeans.fit(X1)
    
    print(f"Initial clustering: {result1.n_clusters} clusters")
    print(f"  Cluster sizes: {np.bincount(result1.labels)}")
    
    # New data arrives (simulating model update)
    X2_new, _ = make_blobs(n_samples=100, n_features=10, centers=3, random_state=99)
    X2_new = X2_new.astype(np.float32)
    X2 = np.vstack([X1, X2_new])
    
    # Extend prior labels for new data
    prior_labels_extended = np.concatenate([
        result1.labels,
        np.full(100, -1, dtype=np.int32)  # Unknown for new samples
    ])
    
    # Recluster with label matching
    kmeans2 = KMeansClusterer(min_clusters=3, max_clusters=3, seed=99)  # Different seed!
    result2 = kmeans2.fit(X2, prior_labels=prior_labels_extended)
    
    # Check stability
    original_labels_kept = result2.labels[:200]
    agreement = np.mean(result1.labels == original_labels_kept)
    
    print(f"\nAfter adding 100 new samples:")
    print(f"  New cluster sizes: {np.bincount(result2.labels)}")
    print(f"  Label stability: {agreement:.1%}")
    print(f"  Label mapping: {result2.label_mapping}")
    
    print()


def example_min_support_merge():
    """Example 4: Merging low-support clusters."""
    print("=" * 60)
    print("Example 4: Min-Support Cluster Merging")
    print("=" * 60)
    
    # Create imbalanced data
    np.random.seed(42)
    large1 = np.random.randn(200, 10).astype(np.float32)
    large2 = np.random.randn(150, 10).astype(np.float32) + 5
    small1 = np.random.randn(10, 10).astype(np.float32) + 10  # Tiny cluster
    small2 = np.random.randn(5, 10).astype(np.float32) - 5   # Another tiny cluster
    
    X = np.vstack([large1, large2, small1, small2])
    
    print(f"Data: {len(X)} samples")
    print(f"  Expected: 2 large + 2 small clusters")
    
    # Cluster with min support
    kmeans = KMeansClusterer(
        min_clusters=2,
        max_clusters=6,
        seed=42,
        min_support=0.03,  # 3% minimum = ~11 samples
        enable_merge=True,
    )
    
    result = kmeans.fit(X)
    
    print(f"\nResults with min_support=3%:")
    print(f"  Final clusters: {result.n_clusters}")
    print(f"  Cluster sizes: {np.bincount(result.labels)}")
    
    if result.merge_table:
        print(f"  Merges performed: {len(result.merge_table)}")
        for merge in result.merge_table:
            print(f"    Cluster {merge['source']} ({merge['n_samples']} samples) "
                  f"→ Cluster {merge['target_final']}")
    else:
        print("  No merges needed")
    
    print()


def example_json_audit_trail():
    """Example 5: JSON serialization for audit trails."""
    print("=" * 60)
    print("Example 5: JSON Audit Trail")
    print("=" * 60)
    
    # Create and cluster data
    X, _ = make_blobs(n_samples=150, n_features=5, centers=3, random_state=42)
    X = X.astype(np.float32)
    
    gmm = GMMClusterer(
        min_clusters=2,
        max_clusters=5,
        seed=42,
        min_support=0.05,
    )
    
    result = gmm.fit(X)
    
    # Export to JSON
    json_str = result.to_json()
    data = json.loads(json_str)
    
    print("JSON audit trail contains:")
    print(f"  Algorithm: {data['parameters']['algorithm']}")
    print(f"  Seed: {data['seed']}")
    print(f"  N clusters: {data['n_clusters']}")
    print(f"  BIC: {data['bic']:.1f}")
    print(f"  Silhouette: {data['silhouette_score']:.3f}")
    print(f"  Parameters tracked: {len(data['parameters'])} items")
    
    # Save to file
    audit_file = Path("clustering_audit.json")
    result.to_json(audit_file)
    print(f"\nSaved to: {audit_file}")
    
    # Demonstrate loading
    loaded_result = ClusteringResult.from_json(str(audit_file))
    print(f"Loaded successfully: {loaded_result.n_clusters} clusters")
    
    # Clean up
    audit_file.unlink()
    print()


def example_strategy_pattern():
    """Example 6: Strategy pattern for algorithm selection."""
    print("=" * 60)
    print("Example 6: Strategy Pattern")
    print("=" * 60)
    
    # Create test data
    X, _ = make_blobs(n_samples=200, n_features=10, centers=3, random_state=42)
    X = X.astype(np.float32)
    
    # Define clustering strategies
    strategies = {
        "kmeans_silhouette": KMeansClusterer(
            min_clusters=2, max_clusters=6, seed=42
        ),
        "gmm_bic": GMMClusterer(
            min_clusters=2, max_clusters=6, seed=42
        ),
        "gmm_spherical": GMMClusterer(
            min_clusters=2, max_clusters=6, seed=42,
            covariance_type="spherical"
        ),
    }
    
    # Try each strategy
    results = {}
    for name, clusterer in strategies.items():
        result = clusterer.fit(X)
        results[name] = {
            "k": result.n_clusters,
            "silhouette": result.silhouette_score,
            "bic": result.bic,
        }
    
    # Compare results
    print("Strategy comparison:")
    for name, metrics in results.items():
        bic_str = f"BIC={metrics['bic']:.1f}" if metrics['bic'] else "N/A"
        print(f"  {name:20s}: k={metrics['k']}, "
              f"silhouette={metrics['silhouette']:.3f}, {bic_str}")
    
    print()


def example_fsq_to_motifs():
    """Example 7: Complete pipeline from FSQ codes to behavioral motifs."""
    print("=" * 60)
    print("Example 7: FSQ Codes to Behavioral Motifs")
    print("=" * 60)
    
    # Simulate IMU data
    torch.manual_seed(42)
    imu_data = torch.randn(200, 9, 2, 100, dtype=torch.float32)
    
    # Add some structure (3 behavioral patterns)
    imu_data[:70] *= 0.5  # Low activity
    imu_data[70:140] *= 2.0  # High activity
    imu_data[140:] += torch.sin(torch.linspace(0, 10, 100))  # Periodic
    
    # Step 1: FSQ encoding
    print("Step 1: FSQ Encoding")
    fsq_result = encode_fsq(imu_data, levels=[8, 6, 5], reset_stats=True)
    print(f"  Codes shape: {fsq_result.codes.shape}")
    print(f"  Perplexity: {fsq_result.perplexity:.2f}")
    print(f"  Unique codes: {len(torch.unique(fsq_result.codes))}")
    
    # Step 2: Prepare features for clustering
    # Use mean pooling over time dimension
    features = fsq_result.quantized.mean(dim=2).numpy()
    print(f"\nStep 2: Feature Extraction")
    print(f"  Features shape: {features.shape}")
    
    # Step 3: Discover motifs with clustering
    print(f"\nStep 3: Motif Discovery")
    clusterer = GMMClusterer(
        min_clusters=2,
        max_clusters=10,
        seed=42,
        min_support=0.05,
    )
    
    motif_result = clusterer.fit(features)
    
    print(f"  Discovered motifs: {motif_result.n_clusters}")
    print(f"  Motif distribution: {np.bincount(motif_result.labels)}")
    print(f"  BIC: {motif_result.bic:.1f}")
    print(f"  Silhouette: {motif_result.silhouette_score:.3f}")
    
    # Step 4: Narrative for committee
    print(f"\nNarrative for Committee:")
    print(f"  • Encoded {len(imu_data)} behavioral segments with FSQ (perplexity={fsq_result.perplexity:.2f})")
    print(f"  • Used {len(torch.unique(fsq_result.codes))}/{fsq_result.codebook_size} available codes")
    print(f"  • Discovered {motif_result.n_clusters} distinct behavioral motifs via GMM+BIC")
    print(f"  • Clustering quality: silhouette={motif_result.silhouette_score:.3f}")
    print(f"  • Full reproducibility: seed={motif_result.seed}, single init, logged parameters")
    
    print()


def example_incremental_learning():
    """Example 8: Incremental learning with stable motifs."""
    print("=" * 60)
    print("Example 8: Incremental Learning")
    print("=" * 60)
    
    # Simulate incremental data collection
    np.random.seed(42)
    
    # Week 1: Initial data
    X_week1, _ = make_blobs(n_samples=100, n_features=20, centers=2, random_state=42)
    X_week1 = X_week1.astype(np.float32)
    
    clusterer = KMeansClusterer(min_clusters=2, max_clusters=5, seed=42)
    result_week1 = clusterer.fit(X_week1)
    
    print(f"Week 1: {len(X_week1)} samples")
    print(f"  Motifs: {result_week1.n_clusters}")
    print(f"  Distribution: {np.bincount(result_week1.labels)}")
    
    # Week 2: More data arrives
    X_week2_new, _ = make_blobs(n_samples=50, n_features=20, centers=2, random_state=99)
    X_week2_new = X_week2_new.astype(np.float32)
    X_week2_new[:, 0] += 2  # Slight shift in first feature
    X_week2 = np.vstack([X_week1, X_week2_new])
    
    # Prepare prior labels
    prior_labels = np.concatenate([
        result_week1.labels,
        np.full(50, -1, dtype=np.int32)
    ])
    
    result_week2 = clusterer.fit(X_week2, prior_labels=prior_labels)
    
    print(f"\nWeek 2: {len(X_week2)} samples (+50 new)")
    print(f"  Motifs: {result_week2.n_clusters}")
    print(f"  Distribution: {np.bincount(result_week2.labels)}")
    print(f"  Label mapping: {result_week2.label_mapping}")
    
    # Week 3: New behavioral pattern emerges
    X_week3_new1, _ = make_blobs(n_samples=30, n_features=20, centers=1, random_state=123)
    X_week3_new1 = X_week3_new1.astype(np.float32) + 10  # New pattern!
    X_week3_new2, _ = make_blobs(n_samples=20, n_features=20, centers=2, random_state=124)
    X_week3_new2 = X_week3_new2.astype(np.float32)
    
    X_week3 = np.vstack([X_week2, X_week3_new1, X_week3_new2])
    
    prior_labels = np.concatenate([
        result_week2.labels,
        np.full(50, -1, dtype=np.int32)
    ])
    
    # Increase max clusters to allow discovery
    clusterer.max_clusters = 6
    result_week3 = clusterer.fit(X_week3, prior_labels=prior_labels)
    
    print(f"\nWeek 3: {len(X_week3)} samples (+50 new)")
    print(f"  Motifs: {result_week3.n_clusters}")
    print(f"  Distribution: {np.bincount(result_week3.labels)}")
    
    if result_week3.n_clusters > result_week2.n_clusters:
        print(f"  ✓ Discovered {result_week3.n_clusters - result_week2.n_clusters} new motif(s)!")
    
    print()


def main():
    """Run all clustering examples."""
    examples = [
        example_deterministic_kmeans,
        example_gmm_with_bic,
        example_label_stability,
        example_min_support_merge,
        example_json_audit_trail,
        example_strategy_pattern,
        example_fsq_to_motifs,
        example_incremental_learning,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example {example_func.__name__} failed: {e}")
            print()


if __name__ == "__main__":
    main()