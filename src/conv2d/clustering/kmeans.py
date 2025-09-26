"""K-Means clusterer with deterministic initialization.

Implements K-Means with:
- k-means++ initialization with fixed seed
- Single initialization (n_init=1)
- Silhouette score for K selection
- Full parameter logging
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from conv2d.clustering.interface import Clusterer

logger = logging.getLogger(__name__)


class KMeansClusterer(Clusterer):
    """K-Means clustering with deterministic behavior.
    
    Uses silhouette score for K selection and k-means++ 
    initialization with fixed seed for reproducibility.
    """
    
    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 20,
        seed: int = 42,
        min_support: Optional[float] = 0.01,
        enable_merge: bool = True,
        init: str = "k-means++",
        n_init: int = 1,
        max_iter: int = 300,
        tol: float = 1e-4,
    ):
        """Initialize K-Means clusterer.
        
        Args:
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider
            seed: Random seed for reproducibility
            min_support: Minimum fraction of samples for valid cluster
            enable_merge: Whether to merge low-support clusters
            init: Initialization method ('k-means++', 'random', or array)
            n_init: Number of initializations (1 for determinism)
            max_iter: Maximum iterations for convergence
            tol: Tolerance for convergence
        """
        super().__init__(min_clusters, max_clusters, seed, min_support, enable_merge)
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        
        # Store initial centroids for reproducibility
        self.initial_centroids_: Optional[NDArray[np.float32]] = None
        
        logger.info(
            f"KMeansClusterer: init={init}, n_init={n_init}, "
            f"max_iter={max_iter}, tol={tol}"
        )
    
    def _fit_single_k(
        self, 
        X: NDArray[np.float32], 
        n_clusters: int
    ) -> Tuple[NDArray[np.int32], NDArray[np.float32], float, Dict[str, Any]]:
        """Fit K-Means for a single K value.
        
        Args:
            X: Data matrix (N x D)
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (labels, centroids, silhouette, extra_params)
        """
        # Use stored initial centroids if available and matching K
        if (self.initial_centroids_ is not None and 
            len(self.initial_centroids_) == n_clusters):
            init = self.initial_centroids_
            logger.debug(f"Using stored initial centroids for k={n_clusters}")
        else:
            init = self.init
        
        # Create KMeans with deterministic settings
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=self.n_init,  # Single initialization for determinism
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.seed,
            verbose=0,
        )
        
        # Fit model
        labels = kmeans.fit_predict(X).astype(np.int32)
        centroids = kmeans.cluster_centers_.astype(np.float32)
        
        # Store initial centroids if first time
        if self.initial_centroids_ is None and n_clusters == self.max_clusters // 2:
            self.initial_centroids_ = centroids.copy()
        
        # Compute silhouette score
        if n_clusters > 1 and n_clusters < len(X):
            sil_score = silhouette_score(X, labels)
        else:
            sil_score = 0.0
        
        # Extra parameters
        extra = {
            "n_iter": kmeans.n_iter_,
            "inertia": kmeans.inertia_,
            "init_method": self.init if isinstance(init, str) else "stored",
        }
        
        return labels, centroids, sil_score, extra
    
    def _select_optimal_k(
        self, 
        X: NDArray[np.float32], 
        metrics: List[Tuple[int, float]]
    ) -> int:
        """Select optimal K using silhouette score.
        
        Args:
            X: Data matrix (N x D)
            metrics: List of (k, silhouette_score) tuples
            
        Returns:
            Optimal number of clusters
        """
        # Find K with maximum silhouette score
        best_k = max(metrics, key=lambda x: x[1])[0]
        
        # Log selection reasoning
        logger.info("K selection using silhouette score:")
        for k, score in metrics:
            marker = " <-- SELECTED" if k == best_k else ""
            logger.info(f"  k={k:2d}: silhouette={score:.4f}{marker}")
        
        return best_k
    
    def set_initial_centroids(
        self, 
        centroids: Optional[NDArray[np.float32]]
    ) -> None:
        """Set initial centroids for deterministic initialization.
        
        Args:
            centroids: Initial centroid positions (K x D) or None
        """
        if centroids is not None:
            self.initial_centroids_ = centroids.astype(np.float32).copy()
            logger.info(f"Set {len(centroids)} initial centroids")
        else:
            self.initial_centroids_ = None
            logger.info("Cleared initial centroids")