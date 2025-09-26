"""Finite Scalar Quantization (FSQ) layer with type hints."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor


class FSQLayer(nn.Module):
    """Finite Scalar Quantization layer for discrete representation learning.
    
    This layer quantizes continuous features into discrete codes using
    learnable scalar quantization levels.
    
    Attributes:
        levels: Number of quantization levels per dimension
        embedding_dim: Dimension of the quantized embeddings
        codebook_size: Total number of unique codes
    """

    def __init__(
        self,
        levels: list[int],
        embedding_dim: int | None = None,
    ) -> None:
        """Initialize FSQ layer.
        
        Args:
            levels: Number of quantization levels per dimension
            embedding_dim: Optional embedding dimension (auto-computed if None)
        """
        super().__init__()
        self.levels: list[int] = levels
        self.num_dimensions: int = len(levels)
        
        # Compute codebook size
        self.codebook_size: int = 1
        for level in levels:
            self.codebook_size *= level
            
        # Determine embedding dimension
        if embedding_dim is None:
            self.embedding_dim = self._compute_embedding_dim()
        else:
            self.embedding_dim = embedding_dim
            
        # Create quantization boundaries
        self.register_buffer(
            "boundaries",
            self._create_boundaries(),
            persistent=True,
        )
        
        # Projection layers
        self.project_in = nn.Linear(self.embedding_dim, self.num_dimensions)
        self.project_out = nn.Linear(self.num_dimensions, self.embedding_dim)
        
        # Statistics tracking
        self.register_buffer("code_usage", torch.zeros(self.codebook_size))
        self.register_buffer("total_samples", torch.tensor(0.0))
        
    def _compute_embedding_dim(self) -> int:
        """Compute appropriate embedding dimension based on codebook size."""
        if self.codebook_size <= 16:
            return 32
        elif self.codebook_size <= 64:
            return 64
        elif self.codebook_size <= 256:
            return 128
        else:
            return 256
            
    def _create_boundaries(self) -> Tensor:
        """Create quantization boundaries for each dimension."""
        boundaries_list = []
        for level in self.levels:
            # Create evenly spaced boundaries in [-1, 1]
            bounds = torch.linspace(-1, 1, level)
            boundaries_list.append(bounds)
        return torch.stack(boundaries_list)
        
    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize continuous values to discrete codes.
        
        Args:
            x: Input tensor of shape (B, D) where D is num_dimensions
            
        Returns:
            Tuple of (quantized values, discrete codes)
        """
        batch_size = x.shape[0]
        
        # Clamp to [-1, 1]
        x_clamped = torch.tanh(x)
        
        # Quantize each dimension
        codes = torch.zeros_like(x, dtype=torch.long)
        quantized = torch.zeros_like(x)
        
        for dim in range(self.num_dimensions):
            # Find closest boundary
            diffs = torch.abs(
                x_clamped[:, dim:dim+1] - self.boundaries[dim].unsqueeze(0)
            )
            indices = torch.argmin(diffs, dim=1)
            codes[:, dim] = indices
            quantized[:, dim] = self.boundaries[dim][indices]
            
        # Convert multi-dimensional codes to single index
        flat_codes = self._flatten_codes(codes)
        
        # Update statistics
        if self.training:
            self._update_statistics(flat_codes)
            
        return quantized, flat_codes
        
    def _flatten_codes(self, codes: Tensor) -> Tensor:
        """Convert multi-dimensional codes to flat indices."""
        flat_codes = codes[:, 0]
        multiplier = 1
        for dim in range(1, self.num_dimensions):
            multiplier *= self.levels[dim - 1]
            flat_codes = flat_codes + codes[:, dim] * multiplier
        return flat_codes
        
    def _update_statistics(self, codes: Tensor) -> None:
        """Update code usage statistics."""
        unique_codes, counts = torch.unique(codes, return_counts=True)
        self.code_usage[unique_codes] += counts.float()
        self.total_samples += codes.shape[0]
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Forward pass with quantization.
        
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, D)
            
        Returns:
            Tuple of (quantized output, codes, metrics dictionary)
        """
        # Flatten if needed
        original_shape = x.shape
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
            
        # Project to quantization space
        x_proj = self.project_in(x)
        
        # Quantize with straight-through estimator
        quantized, codes = self.quantize(x_proj)
        
        if self.training:
            # Straight-through estimator
            quantized = x_proj + (quantized - x_proj).detach()
            
        # Project back to original space
        output = self.project_out(quantized)
        
        # Reshape if needed
        if len(original_shape) > 2:
            output = output.view(original_shape)
            
        # Compute metrics
        metrics = {
            "perplexity": self.perplexity,
            "usage_rate": self.usage_rate,
        }
        
        return output, codes, metrics
        
    @property
    def perplexity(self) -> Tensor:
        """Calculate perplexity of code usage."""
        if self.total_samples == 0:
            return torch.tensor(0.0)
            
        probs = self.code_usage / self.total_samples
        probs = probs[probs > 0]  # Filter zero probabilities
        
        if len(probs) == 0:
            return torch.tensor(0.0)
            
        entropy = -torch.sum(probs * torch.log(probs))
        perplexity = torch.exp(entropy)
        
        return perplexity
        
    @property
    def usage_rate(self) -> Tensor:
        """Calculate percentage of codes being used."""
        used_codes = (self.code_usage > 0).sum()
        return used_codes.float() / self.codebook_size
        
    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.code_usage.zero_()
        self.total_samples.zero_()