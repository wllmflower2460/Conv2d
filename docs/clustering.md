# Clustering System Documentation

The clustering system provides deterministic, reproducible clustering with Hungarian matching for label stability and comprehensive audit trails.

## Overview

Key features:
- **Strategy pattern**: Pluggable algorithms (K-means, GMM)
- **Hungarian matching**: Label stability across runs
- **Min-support merging**: Automatic cluster consolidation
- **JSON audit trails**: Complete reproducibility
- **Deterministic**: Fixed seed → identical results

## Architecture

```
Features (N, D) → Clusterer → Labels (N,) → Hungarian Matching → Stable Labels (N,)
                     ↓              ↓              ↓                    ↓
                 Algorithm      Raw Clusters   Label Map         Final Assignment
                 Selection                                             ↓
                                                               JSON Audit Trail
```

## Core Components

### Abstract Base Class

```python
from conv2d.clustering.interface import Clusterer

class Clusterer(ABC):
    """Abstract base class for clustering algorithms."""
    
    @abstractmethod
    def fit_predict(
        self, 
        features: np.ndarray,
        k: int,
        prior_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit and predict cluster labels."""
        pass
    
    def merge_small_clusters(
        self,
        labels: np.ndarray,
        features: np.ndarray, 
        min_support: float = 0.05,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Merge clusters below minimum support threshold."""
        pass
```

### Concrete Implementations

#### K-means Clustering

```python
from conv2d.clustering.kmeans import KMeansClusterer

clusterer = KMeansClusterer(random_state=42)
labels = clusterer.fit_predict(features, k=4)
```

#### GMM Clustering

```python
from conv2d.clustering.gmm import GMMClusterer

clusterer = GMMClusterer(random_state=42)
labels = clusterer.fit_predict(features, k=4, prior_labels=previous_labels)
```

## Hungarian Matching

Ensures label consistency across runs by solving the assignment problem:

```python
def _match_labels(
    self,
    labels: np.ndarray,
    centroids: np.ndarray,
    prior_labels: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Match current labels to prior labels using Hungarian algorithm."""
    
    # Compute cost matrix (negative similarity)
    cost_matrix = self._compute_cost_matrix(centroids, prior_centroids)
    
    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Apply label mapping
    label_map = dict(zip(row_indices, col_indices))
    matched_labels = np.array([label_map[label] for label in labels])
    
    return matched_labels
```

### Cost Matrix Computation

```python
def _compute_cost_matrix(self, centroids_new, centroids_old):
    """Compute cost matrix for Hungarian assignment."""
    
    # Use negative cosine similarity as cost
    similarities = cosine_similarity(centroids_new, centroids_old)
    cost_matrix = -similarities  # Negative for minimization
    
    # Handle edge cases
    cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=-1.0)
    
    return cost_matrix
```

## Usage Examples

### Basic K-means Clustering

```python
import numpy as np
from conv2d.clustering.kmeans import KMeansClusterer

# Generate sample features
features = np.random.randn(1000, 64).astype(np.float32)

# Create clusterer
clusterer = KMeansClusterer(random_state=42)

# Fit and predict
labels = clusterer.fit_predict(features, k=4)

print(f"Cluster distribution: {np.bincount(labels)}")
```

### GMM with Hungarian Matching

```python
from conv2d.clustering.gmm import GMMClusterer

# First run
clusterer = GMMClusterer(random_state=42)
labels_run1 = clusterer.fit_predict(features, k=4)

# Second run with different initialization
clusterer2 = GMMClusterer(random_state=43)
labels_run2 = clusterer2.fit_predict(features, k=4, prior_labels=labels_run1)

# Check alignment improvement
raw_agreement = np.mean(labels_run1 == labels_run2)
print(f"Label agreement after Hungarian matching: {raw_agreement:.1%}")
```

### Min-Support Merging

```python
# Cluster with potential small clusters
labels = clusterer.fit_predict(features, k=8)

# Merge clusters with <5% support
labels_merged, merge_ops = clusterer.merge_small_clusters(
    labels, features, min_support=0.05
)

print(f"Merge operations: {len(merge_ops)}")
for op in merge_ops:
    print(f"  Merged cluster {op['source']} → {op['target']} ({op['n_samples']} samples)")
```

## Configuration

### K-means Parameters

```python
clusterer = KMeansClusterer(
    random_state=42,           # Reproducibility
    n_init=20,                 # Multiple initializations
    max_iter=300,              # Convergence limit
    tol=1e-4,                  # Convergence tolerance
    algorithm='lloyd',         # Algorithm variant
)
```

### GMM Parameters

```python
clusterer = GMMClusterer(
    random_state=42,           # Reproducibility
    covariance_type='full',    # Covariance structure
    n_init=10,                 # Multiple initializations
    max_iter=100,              # EM iterations
    tol=1e-3,                  # Convergence tolerance
    reg_covar=1e-6,           # Regularization
)
```

## Automatic K Selection

### Silhouette Score (K-means)

```python
def select_k_silhouette(features, k_range=(2, 10)):
    """Select optimal K using silhouette score."""
    
    best_k = 2
    best_score = -1
    
    for k in range(*k_range):
        clusterer = KMeansClusterer(random_state=42)
        labels = clusterer.fit_predict(features, k=k)
        
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k, best_score
```

### BIC Score (GMM)

```python
def select_k_bic(features, k_range=(2, 10)):
    """Select optimal K using BIC score."""
    
    best_k = 2
    best_bic = float('inf')
    
    for k in range(*k_range):
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(features, k=k)
        
        bic = clusterer.model.bic(features)
        if bic < best_bic:
            best_bic = bic
            best_k = k
    
    return best_k, best_bic
```

## Quality Metrics

### Cluster Quality Assessment

```python
def assess_cluster_quality(features, labels):
    """Comprehensive cluster quality assessment."""
    
    metrics = {}
    
    # Silhouette score
    metrics['silhouette'] = silhouette_score(features, labels)
    
    # Calinski-Harabasz index
    metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)
    
    # Davies-Bouldin index  
    metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
    
    # Inertia (within-cluster sum of squares)
    metrics['inertia'] = _compute_inertia(features, labels)
    
    # Cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    metrics['cluster_sizes'] = dict(zip(unique_labels, counts))
    
    # Balance (coefficient of variation)
    metrics['balance'] = np.std(counts) / np.mean(counts)
    
    return metrics
```

### Stability Analysis

```python
def analyze_stability(features, n_runs=10, k=4):
    """Analyze clustering stability across multiple runs."""
    
    all_labels = []
    
    # Multiple random initializations
    for seed in range(n_runs):
        clusterer = KMeansClusterer(random_state=seed)
        labels = clusterer.fit_predict(features, k=k)
        all_labels.append(labels)
    
    # Compute pairwise ARI scores
    ari_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            ari_scores.append(ari)
    
    stability = {
        'mean_ari': np.mean(ari_scores),
        'std_ari': np.std(ari_scores),
        'min_ari': np.min(ari_scores),
        'max_ari': np.max(ari_scores),
    }
    
    return stability
```

## Audit Trail System

### JSON Output Format

```python
{
    "algorithm": "gmm",
    "parameters": {
        "k": 4,
        "random_state": 42,
        "covariance_type": "full"
    },
    "input": {
        "n_samples": 1000,
        "n_features": 64,
        "data_hash": "a1b2c3d4..."
    },
    "results": {
        "labels": [0, 1, 2, 1, 0, ...],
        "cluster_centers": [[...], [...], ...],
        "n_clusters_final": 4
    },
    "quality_metrics": {
        "silhouette": 0.68,
        "calinski_harabasz": 156.8,
        "davies_bouldin": 0.89,
        "inertia": 2847.3
    },
    "merge_operations": [
        {
            "source": 5,
            "target": 3,
            "n_samples": 23,
            "reason": "min_support_violation"
        }
    ],
    "hungarian_matching": {
        "applied": true,
        "cost_matrix": [[...], [...], ...],
        "assignment": [0, 1, 2, 3],
        "total_cost": 1.23
    },
    "execution": {
        "timestamp": "2024-01-15T10:30:45",
        "duration_ms": 147.6,
        "convergence": true,
        "n_iterations": 8
    }
}
```

### Reproducibility Verification

```python
def verify_reproducibility(features, audit_trail):
    """Verify clustering can be reproduced from audit trail."""
    
    # Extract parameters
    params = audit_trail['parameters']
    algorithm = audit_trail['algorithm']
    
    # Recreate clusterer
    if algorithm == 'kmeans':
        clusterer = KMeansClusterer(**params)
    elif algorithm == 'gmm':
        clusterer = GMMClusterer(**params)
    
    # Re-run clustering
    labels_repro = clusterer.fit_predict(features, k=params['k'])
    
    # Compare with original
    labels_orig = np.array(audit_trail['results']['labels'])
    match = np.array_equal(labels_orig, labels_repro)
    
    return match, labels_repro
```

## Performance Optimization

### Batch Processing

```python
def cluster_batches(features_list, k=4, batch_size=1000):
    """Cluster multiple feature sets efficiently."""
    
    clusterer = GMMClusterer(random_state=42)
    all_labels = []
    
    # Process in batches
    for i in range(0, len(features_list), batch_size):
        batch = features_list[i:i + batch_size]
        
        # Concatenate batch
        features_batch = np.vstack(batch)
        
        # Cluster
        labels_batch = clusterer.fit_predict(features_batch, k=k)
        
        # Split back to original structure
        start_idx = 0
        for features in batch:
            n_samples = len(features)
            labels = labels_batch[start_idx:start_idx + n_samples]
            all_labels.append(labels)
            start_idx += n_samples
    
    return all_labels
```

### Memory-Efficient Processing

```python
def cluster_large_dataset(features, k=4, chunk_size=10000):
    """Handle large datasets with memory constraints."""
    
    n_samples = len(features)
    
    if n_samples <= chunk_size:
        # Process normally
        clusterer = GMMClusterer(random_state=42)
        return clusterer.fit_predict(features, k=k)
    
    # Use MiniBatchKMeans for large data
    from sklearn.cluster import MiniBatchKMeans
    
    clusterer = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=chunk_size,
        max_iter=100,
    )
    
    labels = clusterer.fit_predict(features)
    return labels.astype(np.int32)
```

## Integration with Pipeline

### With FSQ Features

```python
from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.gmm import GMMClusterer

# Encode features
x = torch.randn(100, 9, 2, 100, dtype=torch.float32)
fsq_result = encode_fsq(x)

# Cluster the features
clusterer = GMMClusterer(random_state=42)
behavior_labels = clusterer.fit_predict(fsq_result.features.numpy(), k=4)

print(f"Identified {len(np.unique(behavior_labels))} behavioral clusters")
```

### With Temporal Smoothing

```python
# Cluster sequence of features
sequence_labels = []
for t in range(T):
    fsq_result = encode_fsq(x_sequence[t])
    labels = clusterer.fit_predict(fsq_result.features.numpy(), k=4)
    sequence_labels.append(labels)

# Stack into temporal sequence
motif_sequence = np.array(sequence_labels)  # (T, B)

# Apply temporal smoothing
from conv2d.temporal.median import MedianHysteresisPolicy
policy = MedianHysteresisPolicy(min_dwell=5)
smoothed_sequence = policy.smooth(motif_sequence)
```

## Best Practices

1. **Always set random_state**: For reproducible results
2. **Use Hungarian matching**: When comparing across runs
3. **Monitor cluster balance**: Avoid degenerate solutions
4. **Apply min-support merging**: Remove tiny clusters
5. **Validate with multiple metrics**: Silhouette, CH index, DB index
6. **Keep audit trails**: For debugging and reproducibility
7. **Test stability**: Multiple random initializations
8. **Choose appropriate K**: Use automatic selection methods

This clustering system ensures stable, reproducible behavioral discovery with comprehensive quality monitoring and audit capabilities.