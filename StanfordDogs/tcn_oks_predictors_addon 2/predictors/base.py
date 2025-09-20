
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class KeypointPredictor(ABC):
    """Abstract base for single-instance keypoint predictors."""

    @abstractmethod
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Return dict with 'kpts': (K,2), 'kvis': (K,), 'bbox': (4,) for a single image batch."""
        raise NotImplementedError

def harmonize_kpts(kpts: np.ndarray, K: int) -> np.ndarray:
    if kpts.shape[0] == K:
        return kpts
    if kpts.shape[0] > K:
        return kpts[:K]
    pad = np.zeros((K - kpts.shape[0], 2), dtype=kpts.dtype)
    return np.concatenate([kpts, pad], axis=0)

def harmonize_vis(vis: np.ndarray, K: int) -> np.ndarray:
    if vis.shape[0] == K:
        return vis
    if vis.shape[0] > K:
        return vis[:K]
    pad = np.zeros((K - vis.shape[0],), dtype=vis.dtype)
    return np.concatenate([vis, pad], axis=0)
