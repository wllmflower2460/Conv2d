"""Deterministic clustering with strategy pattern.

This module provides stable, reproducible clustering for behavioral
motif discovery with full auditability and label stability.
"""

from __future__ import annotations

from conv2d.clustering.gmm import GMMClusterer
from conv2d.clustering.interface import Clusterer, ClusteringResult
from conv2d.clustering.kmeans import KMeansClusterer

__all__ = [
    "Clusterer",
    "ClusteringResult",
    "KMeansClusterer", 
    "GMMClusterer",
]