"""Deterministic clustering interface with strategy pattern.

This module provides a stable, reproducible clustering framework
designed for behavioral motif discovery with full auditability.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Result of clustering with full provenance.
    
    Attributes:
        labels: Cluster assignments for each sample
        centroids: Cluster centers (K x D)
        n_clusters: Number of clusters found
        inertia: Sum of squared distances to nearest cluster
        silhouette_score: Mean silhouette coefficient (-1 to 1)
        bic: Bayesian Information Criterion (for GMM)
        parameters: Dictionary of clustering parameters used
        merge_table: Optional merge operations applied
        label_mapping: Mapping from raw to stable labels
        seed: Random seed used for reproducibility
    """
    labels: NDArray[np.int32]
    centroids: NDArray[np.float32]
    n_clusters: int
    inertia: float
    silhouette_score: float
    bic: Optional[float]
    parameters: Dict[str, Any]
    merge_table: Optional[List[Dict[str, Any]]]
    label_mapping: Optional[Dict[int, int]]
    seed: int
    
    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export result to JSON for audit trail.
        
        Args:
            path: Optional path to save JSON file
            
        Returns:
            JSON string representation
        """
        # Convert numpy arrays to lists for JSON serialization
        data = {
            "labels": self.labels.tolist(),
            "centroids": self.centroids.tolist(),
            "n_clusters": int(self.n_clusters),
            "inertia": float(self.inertia),
            "silhouette_score": float(self.silhouette_score),
            "bic": float(self.bic) if self.bic is not None else None,
            "parameters": {k: (v.tolist() if isinstance(v, np.ndarray) else 
                              float(v) if isinstance(v, (np.floating, np.integer)) else v)
                         for k, v in self.parameters.items()},
            "merge_table": self.merge_table,
            "label_mapping": {int(k): int(v) for k, v in self.label_mapping.items()} 
                           if self.label_mapping else None,
            "seed": int(self.seed),
        }
        
        json_str = json.dumps(data, indent=2)
        
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str)
            logger.info(f"Saved clustering result to {path}")
            
        return json_str
    
    @classmethod
    def from_json(cls, json_str: str) -> ClusteringResult:
        """Load result from JSON.
        
        Args:
            json_str: JSON string or path to JSON file
            
        Returns:
            ClusteringResult instance
        """
        # Check if it's a path
        try:
            path = Path(json_str)
            if path.exists():
                json_str = path.read_text()
        except:
            pass
            
        data = json.loads(json_str)
        
        return cls(
            labels=np.array(data["labels"], dtype=np.int32),
            centroids=np.array(data["centroids"], dtype=np.float32),
            n_clusters=data["n_clusters"],
            inertia=data["inertia"],
            silhouette_score=data["silhouette_score"],
            bic=data.get("bic"),
            parameters=data["parameters"],
            merge_table=data.get("merge_table"),
            label_mapping=data.get("label_mapping"),
            seed=data["seed"],
        )


class Clusterer(ABC):
    """Abstract base class for clustering algorithms.
    
    Ensures deterministic, reproducible clustering with:
    - Fixed random seeds
    - Single initialization (n_init=1)
    - Logged parameters
    - Label stability across retrains
    """
    
    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 20,
        seed: int = 42,
        min_support: Optional[float] = 0.01,
        enable_merge: bool = True,
    ):
        """Initialize clusterer.
        
        Args:
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider
            seed: Random seed for reproducibility
            min_support: Minimum fraction of samples for valid cluster
            enable_merge: Whether to merge low-support clusters
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.seed = seed
        self.min_support = min_support
        self.enable_merge = enable_merge
        self.fitted_ = False
        self.result_: Optional[ClusteringResult] = None
        self.prior_centroids_: Optional[NDArray[np.float32]] = None
        
        # Set numpy seed for reproducibility
        np.random.seed(seed)
        
        logger.info(
            f"Initialized {self.__class__.__name__} with "
            f"k=[{min_clusters}, {max_clusters}], seed={seed}, "
            f"min_support={min_support}, merge={enable_merge}"
        )
    
    @abstractmethod
    def _fit_single_k(
        self, 
        X: NDArray[np.float32], 
        n_clusters: int
    ) -> Tuple[NDArray[np.int32], NDArray[np.float32], float, Dict[str, Any]]:
        """Fit clustering for a single K value.
        
        Args:
            X: Data matrix (N x D)
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (labels, centroids, metric, extra_params)
        """
        pass
    
    @abstractmethod
    def _select_optimal_k(
        self, 
        X: NDArray[np.float32], 
        metrics: List[Tuple[int, float]]
    ) -> int:
        """Select optimal number of clusters.
        
        Args:
            X: Data matrix (N x D)
            metrics: List of (k, metric_value) tuples
            
        Returns:
            Optimal number of clusters
        """
        pass
    
    def fit(
        self, 
        X: NDArray[np.float32],
        prior_labels: Optional[NDArray[np.int32]] = None,
    ) -> ClusteringResult:
        """Fit clustering model with deterministic behavior.
        
        Args:
            X: Data matrix (N x D)
            prior_labels: Optional prior labels for stability matching
            
        Returns:
            ClusteringResult with full provenance
        """
        # Ensure float32
        X = X.astype(np.float32)
        n_samples, n_features = X.shape
        
        logger.info(f"Fitting {self.__class__.__name__} on {n_samples} x {n_features} data")
        
        # Try different K values
        results = []
        for k in range(self.min_clusters, min(self.max_clusters + 1, n_samples)):
            labels, centroids, metric, extra = self._fit_single_k(X, k)
            results.append((k, metric, labels, centroids, extra))
            logger.debug(f"  k={k}: metric={metric:.4f}")
        
        # Select optimal K
        metrics = [(k, metric) for k, metric, _, _, _ in results]
        optimal_k = self._select_optimal_k(X, metrics)
        
        logger.info(f"Selected optimal k={optimal_k}")
        
        # Get optimal result
        for k, metric, labels, centroids, extra in results:
            if k == optimal_k:
                break
        
        # Apply label stability matching if prior labels provided
        if prior_labels is not None:
            labels, label_mapping = self._match_labels(
                labels, centroids, prior_labels, X
            )
        else:
            label_mapping = None
        
        # Apply min-support merging if enabled
        merge_table = None
        if self.enable_merge and self.min_support is not None:
            labels, merge_table = self._merge_low_support(
                labels, centroids, X, self.min_support
            )
            
            # Recompute centroids after merging
            centroids = self._compute_centroids(X, labels)
        
        # Compute final metrics
        from sklearn.metrics import silhouette_score
        
        inertia = self._compute_inertia(X, labels, centroids)
        sil_score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
        
        # Store result
        self.result_ = ClusteringResult(
            labels=labels,
            centroids=centroids,
            n_clusters=len(np.unique(labels)),
            inertia=inertia,
            silhouette_score=sil_score,
            bic=extra.get("bic"),
            parameters={
                "algorithm": self.__class__.__name__,
                "min_clusters": self.min_clusters,
                "max_clusters": self.max_clusters,
                "optimal_k": optimal_k,
                "min_support": self.min_support,
                "enable_merge": self.enable_merge,
                **extra,
            },
            merge_table=merge_table,
            label_mapping=label_mapping,
            seed=self.seed,
        )
        
        # Store centroids for next iteration
        self.prior_centroids_ = centroids.copy()
        self.fitted_ = True
        
        logger.info(
            f"Clustering complete: {self.result_.n_clusters} clusters, "
            f"silhouette={sil_score:.3f}, inertia={inertia:.2f}"
        )
        
        return self.result_
    
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        """Predict cluster labels for new data.
        
        Args:
            X: Data matrix (N x D)
            
        Returns:
            Cluster labels
        """
        if not self.fitted_:
            raise RuntimeError("Clusterer must be fitted before prediction")
        
        X = X.astype(np.float32)
        
        # Assign to nearest centroid
        distances = self._compute_distances(X, self.result_.centroids)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        
        return labels
    
    def _match_labels(
        self,
        labels: NDArray[np.int32],
        centroids: NDArray[np.float32],
        prior_labels: NDArray[np.int32],
        X: NDArray[np.float32],
    ) -> Tuple[NDArray[np.int32], Dict[int, int]]:
        """Match labels to prior using Hungarian algorithm.
        
        Args:
            labels: Current cluster labels
            centroids: Current centroids
            prior_labels: Prior cluster labels
            X: Original data
            
        Returns:
            Tuple of (matched_labels, label_mapping)
        """
        from scipy.optimize import linear_sum_assignment
        
        # Compute confusion matrix between current and prior
        n_current = len(np.unique(labels))
        n_prior = len(np.unique(prior_labels))
        
        # Create cost matrix (negative overlap for minimization)
        cost_matrix = np.zeros((n_current, n_prior))
        for i in range(n_current):
            for j in range(n_prior):
                mask_current = labels == i
                mask_prior = prior_labels == j
                overlap = np.sum(mask_current & mask_prior)
                cost_matrix[i, j] = -overlap
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create mapping
        label_mapping = {}
        matched_labels = labels.copy()
        
        for i, j in zip(row_ind, col_ind):
            label_mapping[i] = j
            matched_labels[labels == i] = j
        
        # Handle unmatched clusters (new clusters)
        next_label = n_prior
        for i in range(n_current):
            if i not in label_mapping:
                label_mapping[i] = next_label
                matched_labels[labels == i] = next_label
                next_label += 1
        
        logger.info(f"Label matching: {label_mapping}")
        
        return matched_labels, label_mapping
    
    def _merge_low_support(
        self,
        labels: NDArray[np.int32],
        centroids: NDArray[np.float32],
        X: NDArray[np.float32],
        min_support: float,
    ) -> Tuple[NDArray[np.int32], List[Dict[str, Any]]]:
        """Merge clusters with low support.
        
        Args:
            labels: Cluster labels
            centroids: Cluster centroids
            X: Original data
            min_support: Minimum fraction of samples
            
        Returns:
            Tuple of (merged_labels, merge_table)
        """
        n_samples = len(labels)
        min_samples = int(min_support * n_samples)
        
        merge_table = []
        merged_labels = labels.copy()
        
        # Find low-support clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        low_support = unique_labels[counts < min_samples]
        
        if len(low_support) == 0:
            return merged_labels, merge_table
        
        logger.info(f"Merging {len(low_support)} low-support clusters (< {min_samples} samples)")
        
        # Merge each low-support cluster to nearest high-support cluster
        high_support = unique_labels[counts >= min_samples]
        
        for low_label in low_support:
            if len(high_support) == 0:
                # No high-support clusters, keep as is
                break
                
            # Find nearest high-support cluster
            low_centroid = centroids[low_label]
            high_centroids = centroids[high_support]
            distances = np.linalg.norm(
                high_centroids - low_centroid[np.newaxis, :], 
                axis=1
            )
            nearest_idx = np.argmin(distances)
            target_label = high_support[nearest_idx]
            
            # Merge
            mask = merged_labels == low_label
            n_merged = np.sum(mask)
            merged_labels[mask] = target_label
            
            # Record merge
            merge_entry = {
                "source": int(low_label),
                "target": int(target_label),
                "n_samples": int(n_merged),
                "distance": float(distances[nearest_idx]),
            }
            merge_table.append(merge_entry)
            
            logger.debug(
                f"  Merged cluster {low_label} ({n_merged} samples) "
                f"into {target_label} (distance={distances[nearest_idx]:.3f})"
            )
        
        # Relabel to continuous integers
        unique_merged = np.unique(merged_labels)
        relabel_map = {old: new for new, old in enumerate(unique_merged)}
        merged_labels = np.array([relabel_map[l] for l in merged_labels], dtype=np.int32)
        
        # Update merge table with final labels
        for entry in merge_table:
            entry["source_final"] = relabel_map.get(entry["source"], entry["source"])
            entry["target_final"] = relabel_map.get(entry["target"], entry["target"])
        
        return merged_labels, merge_table
    
    def _compute_distances(
        self, 
        X: NDArray[np.float32], 
        centroids: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute distances from samples to centroids.
        
        Args:
            X: Data matrix (N x D)
            centroids: Centroid matrix (K x D)
            
        Returns:
            Distance matrix (N x K)
        """
        # Vectorized Euclidean distance
        X_expanded = X[:, np.newaxis, :]  # (N, 1, D)
        centroids_expanded = centroids[np.newaxis, :, :]  # (1, K, D)
        distances = np.linalg.norm(X_expanded - centroids_expanded, axis=2)
        return distances.astype(np.float32)
    
    def _compute_centroids(
        self, 
        X: NDArray[np.float32], 
        labels: NDArray[np.int32]
    ) -> NDArray[np.float32]:
        """Compute cluster centroids.
        
        Args:
            X: Data matrix (N x D)
            labels: Cluster labels
            
        Returns:
            Centroid matrix (K x D)
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_features = X.shape[1]
        
        centroids = np.zeros((n_clusters, n_features), dtype=np.float32)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            centroids[i] = X[mask].mean(axis=0)
        
        return centroids
    
    def _compute_inertia(
        self,
        X: NDArray[np.float32],
        labels: NDArray[np.int32],
        centroids: NDArray[np.float32],
    ) -> float:
        """Compute sum of squared distances to nearest cluster.
        
        Args:
            X: Data matrix (N x D)
            labels: Cluster labels
            centroids: Cluster centroids
            
        Returns:
            Inertia value
        """
        inertia = 0.0
        unique_labels = np.unique(labels)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            X_cluster = X[mask]
            centroid = centroids[i]
            distances = np.linalg.norm(X_cluster - centroid[np.newaxis, :], axis=1)
            inertia += np.sum(distances ** 2)
        
        return float(inertia)