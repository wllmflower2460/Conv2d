"""Kinematic feature extractor with type hints."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class KinematicFeatureExtractor:
    """Extract kinematic features from motion data."""
    
    def __init__(
        self,
        sampling_rate: float = 100.0,
        window_size: Optional[int] = None,
    ) -> None:
        """Initialize feature extractor.
        
        Args:
            sampling_rate: Data sampling rate in Hz
            window_size: Optional window size for features
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.window_size = window_size
        
    def extract_features(
        self,
        data: Tensor,
    ) -> dict[str, Tensor]:
        """Extract kinematic features from data.
        
        Args:
            data: Input tensor of shape (B, C, T) or (B, C, S, T)
            
        Returns:
            Dictionary of extracted features
        """
        features: dict[str, Tensor] = {}
        
        # TODO: Implement feature extraction
        features["mean"] = data.mean(dim=-1)
        features["std"] = data.std(dim=-1)
        
        return features
        
    def _compute_dtw_distance(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Compute DTW distance between sequences.
        
        Args:
            x: Tensor of shape (B, C, T)
            y: Tensor of shape (B, C, T)
            
        Returns:
            Distance tensor of shape (B,)
            
        Raises:
            AssertionError: If shapes don't match
        """
        assert x.shape == y.shape, f"Shape mismatch: x={x.shape}, y={y.shape}"
        
        # Use vectorized computation
        distances = torch.linalg.vector_norm(x - y, ord=2, dim=1)
        return torch.mean(distances, dim=1)