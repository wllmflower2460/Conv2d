"""Gaussian Mixture Model clusterer with BIC selection.

Implements GMM with:
- BIC (Bayesian Information Criterion) for K selection
- Single initialization with fixed seed
- Full covariance for flexibility
- Full parameter logging
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from conv2d.clustering.interface import Clusterer

logger = logging.getLogger(__name__)


class GMMClusterer(Clusterer):
    """Gaussian Mixture Model clustering with deterministic behavior.
    
    Uses BIC (Bayesian Information Criterion) for K selection
    and fixed seed for reproducibility.
    """
    
    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 20,
        seed: int = 42,
        min_support: Optional[float] = 0.01,
        enable_merge: bool = True,
        covariance_type: str = "full",
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
    ):
        """Initialize GMM clusterer.
        
        Args:
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider
            seed: Random seed for reproducibility
            min_support: Minimum fraction of samples for valid cluster
            enable_merge: Whether to merge low-support clusters
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            n_init: Number of initializations (1 for determinism)
            max_iter: Maximum iterations for convergence
            tol: Tolerance for convergence
            reg_covar: Regularization for covariance matrices
        """
        super().__init__(min_clusters, max_clusters, seed, min_support, enable_merge)
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        
        # Store initial parameters for reproducibility
        self.initial_means_: Optional[NDArray[np.float32]] = None
        self.initial_weights_: Optional[NDArray[np.float32]] = None
        
        logger.info(
            f"GMMClusterer: covariance_type={covariance_type}, n_init={n_init}, "
            f"max_iter={max_iter}, tol={tol}, reg_covar={reg_covar}"
        )
    
    def _fit_single_k(
        self, 
        X: NDArray[np.float32], 
        n_clusters: int
    ) -> Tuple[NDArray[np.int32], NDArray[np.float32], float, Dict[str, Any]]:
        """Fit GMM for a single K value.
        
        Args:
            X: Data matrix (N x D)
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (labels, centroids, bic, extra_params)
        """
        # Create GMM with deterministic settings
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=self.covariance_type,
            n_init=self.n_init,  # Single initialization for determinism
            max_iter=self.max_iter,
            tol=self.tol,
            reg_covar=self.reg_covar,
            random_state=self.seed,
            verbose=0,
        )
        
        # Use stored initial parameters if available and matching K
        if (self.initial_means_ is not None and 
            len(self.initial_means_) == n_clusters):
            gmm.means_init = self.initial_means_
            gmm.weights_init = self.initial_weights_
            logger.debug(f"Using stored initial parameters for k={n_clusters}")
        
        # Fit model
        gmm.fit(X)
        labels = gmm.predict(X).astype(np.int32)
        
        # Get means as centroids
        centroids = gmm.means_.astype(np.float32)
        
        # Store initial parameters if first time
        if self.initial_means_ is None and n_clusters == self.max_clusters // 2:
            self.initial_means_ = centroids.copy()
            self.initial_weights_ = gmm.weights_.astype(np.float32).copy()
        
        # Compute BIC (lower is better, but we negate for consistency)
        bic = gmm.bic(X)
        
        # Also compute silhouette for comparison
        if n_clusters > 1 and n_clusters < len(X):
            sil_score = silhouette_score(X, labels)
        else:
            sil_score = 0.0
        
        # Extra parameters
        extra = {
            "bic": bic,
            "aic": gmm.aic(X),
            "silhouette": sil_score,
            "n_iter": gmm.n_iter_,
            "converged": gmm.converged_,
            "weights": gmm.weights_.tolist(),
            "covariances_shape": gmm.covariances_.shape,
        }
        
        # Use negative BIC as metric (higher is better for selection)
        return labels, centroids, -bic, extra
    
    def _select_optimal_k(
        self, 
        X: NDArray[np.float32], 
        metrics: List[Tuple[int, float]]
    ) -> int:
        """Select optimal K using BIC.
        
        Args:
            X: Data matrix (N x D)
            metrics: List of (k, -bic) tuples
            
        Returns:
            Optimal number of clusters
        """
        # Find K with maximum metric (which is -BIC, so minimum BIC)
        best_k = max(metrics, key=lambda x: x[1])[0]
        
        # Log selection reasoning
        logger.info("K selection using BIC (Bayesian Information Criterion):")
        for k, neg_bic in metrics:
            bic = -neg_bic  # Convert back to actual BIC
            marker = " <-- SELECTED (lowest BIC)" if k == best_k else ""
            logger.info(f"  k={k:2d}: BIC={bic:10.2f}{marker}")
        
        return best_k
    
    def set_initial_parameters(
        self, 
        means: Optional[NDArray[np.float32]],
        weights: Optional[NDArray[np.float32]] = None,
    ) -> None:
        """Set initial GMM parameters for deterministic initialization.
        
        Args:
            means: Initial mean positions (K x D) or None
            weights: Initial component weights (K,) or None
        """
        if means is not None:
            self.initial_means_ = means.astype(np.float32).copy()
            
            if weights is not None:
                self.initial_weights_ = weights.astype(np.float32).copy()
            else:
                # Equal weights if not provided
                k = len(means)
                self.initial_weights_ = np.ones(k, dtype=np.float32) / k
                
            logger.info(f"Set initial parameters for {len(means)} components")
        else:
            self.initial_means_ = None
            self.initial_weights_ = None
            logger.info("Cleared initial parameters")
    
    def get_component_parameters(self) -> Optional[Dict[str, Any]]:
        """Get fitted GMM component parameters.
        
        Returns:
            Dictionary with means, weights, and covariances if fitted
        """
        if not self.fitted_ or self.result_ is None:
            return None
        
        # Return component parameters from last fit
        return {
            "means": self.result_.centroids,
            "weights": self.result_.parameters.get("weights"),
            "n_components": self.result_.n_clusters,
        }