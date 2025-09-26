"""FSQ encoding contract with clear interface and guarantees.

This module provides a single, well-defined interface for FSQ encoding
with explicit contracts about inputs, outputs, and behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class CodesAndFeatures:
    """Output contract for FSQ encoding.
    
    Attributes:
        codes: Discrete codes of shape [B, T] with dtype int32
        quantized: Quantized features of shape [B, D, T] with dtype float32
        perplexity: Code usage perplexity (higher = more codes used)
        usage_histogram: Histogram of code usage frequencies
        codebook_size: Total number of possible codes
    """
    codes: Tensor  # int32[B, T]
    quantized: Tensor  # float32[B, D, T]
    perplexity: float
    usage_histogram: np.ndarray
    codebook_size: int
    
    def __post_init__(self) -> None:
        """Validate output contract."""
        # Check dtypes
        assert self.codes.dtype == torch.int32, f"codes must be int32, got {self.codes.dtype}"
        assert self.quantized.dtype == torch.float32, f"quantized must be float32, got {self.quantized.dtype}"
        
        # Check shapes
        B_codes, T_codes = self.codes.shape
        B_quant, D_quant, T_quant = self.quantized.shape
        assert B_codes == B_quant, f"Batch size mismatch: codes {B_codes} vs quantized {B_quant}"
        assert T_codes == T_quant, f"Time dimension mismatch: codes {T_codes} vs quantized {T_quant}"
        
        # Check code range
        assert self.codes.min() >= 0, f"Negative code found: {self.codes.min()}"
        assert self.codes.max() < self.codebook_size, f"Code {self.codes.max()} >= codebook size {self.codebook_size}"
        
        # Log statistics
        logger.info(
            f"FSQ encoding: B={B_codes}, T={T_codes}, D={D_quant}, "
            f"codebook_size={self.codebook_size}, perplexity={self.perplexity:.2f}"
        )


class FSQEncoder(nn.Module):
    """FSQ encoder with deterministic behavior and clear contract.
    
    Guarantees:
    - Deterministic: Same input + same seed → identical codes
    - Type-safe: Enforces float32/int32 at boundaries
    - Logged: Tracks code usage and perplexity
    """
    
    def __init__(
        self,
        levels: List[int] = [8, 6, 5],
        input_channels: int = 9,
        input_sensors: int = 2,
        input_timesteps: int = 100,
        embedding_dim: int = 64,
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize FSQ encoder.
        
        Args:
            levels: Quantization levels per dimension
            input_channels: Expected number of input channels (9 for IMU)
            input_sensors: Expected number of sensors (2)
            input_timesteps: Expected number of timesteps (100)
            embedding_dim: Dimension of quantized features
            seed: Random seed for deterministic initialization
        """
        super().__init__()
        
        self.levels = levels
        self.input_channels = input_channels
        self.input_sensors = input_sensors
        self.input_timesteps = input_timesteps
        self.embedding_dim = embedding_dim
        self.num_codes_per_dim = len(levels)
        
        # Compute total codebook size
        self.codebook_size = 1
        for level in levels:
            self.codebook_size *= level
            
        # Set seed for deterministic initialization
        if seed is not None:
            torch.manual_seed(seed)
            
        # Encoder: (B, C, S, T) -> (B, D, T)
        self.encoder = nn.Sequential(
            # Flatten sensors: (B, C, S, T) -> (B, C*S, T)
            nn.Flatten(start_dim=1, end_dim=2),
            # Conv layers
            nn.Conv1d(input_channels * input_sensors, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, self.num_codes_per_dim, kernel_size=1),
        )
        
        # Projection to embedding dimension
        self.projection = nn.Conv1d(self.num_codes_per_dim, embedding_dim, kernel_size=1)
        
        # Create quantization boundaries (deterministic)
        self.boundaries = self._create_boundaries()
        
        # Statistics tracking
        self.register_buffer("code_counts", torch.zeros(self.codebook_size, dtype=torch.int64))
        self.register_buffer("total_samples", torch.tensor(0, dtype=torch.int64))
        
    def _create_boundaries(self) -> List[Tensor]:
        """Create deterministic quantization boundaries."""
        boundaries_list = []
        for i, level in enumerate(self.levels):
            # Uniform boundaries in [-1, 1]
            bounds = torch.linspace(-1.0, 1.0, level)
            boundaries_list.append(bounds)
        return boundaries_list
    
    def _quantize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize continuous values to discrete codes.
        
        Args:
            x: Continuous values of shape (B, num_codes_per_dim, T)
            
        Returns:
            (quantized values, discrete codes)
        """
        B, D, T = x.shape
        assert D == self.num_codes_per_dim, f"Expected {self.num_codes_per_dim} dims, got {D}"
        
        # Apply tanh for bounded range
        x_bounded = torch.tanh(x)
        
        # Quantize each dimension
        quantized = torch.zeros_like(x_bounded)
        indices = torch.zeros(B, D, T, dtype=torch.int64, device=x.device)
        
        for dim in range(D):
            # Get boundaries for this dimension
            bounds = self.boundaries[dim].to(x.device)
            
            # Find closest boundary (deterministic)
            x_dim = x_bounded[:, dim:dim+1, :]  # (B, 1, T)
            diffs = torch.abs(x_dim.unsqueeze(-1) - bounds.view(1, 1, 1, -1))
            idx = torch.argmin(diffs, dim=-1).squeeze(1)  # (B, T)
            
            # Store quantized values and indices
            quantized[:, dim, :] = bounds[idx]
            indices[:, dim, :] = idx
            
        # Convert multi-dimensional indices to flat codes
        codes = self._indices_to_codes(indices)
        
        return quantized, codes
    
    def _indices_to_codes(self, indices: Tensor) -> Tensor:
        """Convert multi-dimensional indices to flat codes.
        
        Args:
            indices: Indices of shape (B, D, T)
            
        Returns:
            Flat codes of shape (B, T)
        """
        B, D, T = indices.shape
        
        # Compute flat codes
        codes = indices[:, 0, :]  # Start with first dimension
        multiplier = 1
        
        for dim in range(1, D):
            multiplier *= self.levels[dim - 1]
            codes = codes + indices[:, dim, :] * multiplier
            
        return codes.to(torch.int32)
    
    def _update_statistics(self, codes: Tensor) -> None:
        """Update code usage statistics.
        
        Args:
            codes: Flat codes of shape (B, T)
        """
        # Count code occurrences
        unique_codes, counts = torch.unique(codes.flatten(), return_counts=True)
        
        # Update counts
        self.code_counts[unique_codes] += counts
        self.total_samples += codes.numel()
        
    @property
    def perplexity(self) -> float:
        """Calculate perplexity of code usage."""
        if self.total_samples == 0:
            return 1.0  # Return 1 for empty statistics (not 0)
            
        # Compute probabilities
        probs = self.code_counts.float() / self.total_samples
        
        # Filter out zero probabilities
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 1.0  # Return 1 if no codes used
            
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs))
        
        # Perplexity = exp(entropy)
        perplexity = torch.exp(entropy).item()
        
        return max(perplexity, 1.0)  # Ensure minimum of 1
    
    @property
    def usage_histogram(self) -> np.ndarray:
        """Get histogram of code usage."""
        if self.total_samples == 0:
            return np.zeros(self.codebook_size)
            
        histogram = (self.code_counts.float() / self.total_samples).cpu().numpy()
        return histogram
    
    def forward(self, x: Tensor) -> CodesAndFeatures:
        """Encode input to codes and features.
        
        Args:
            x: Input tensor of shape (B, C, S, T)
               where B=batch, C=channels=9, S=sensors=2, T=timesteps=100
               
        Returns:
            CodesAndFeatures with codes and quantized features
        """
        # Validate input shape
        B, C, S, T = x.shape
        assert C == self.input_channels, f"Expected {self.input_channels} channels, got {C}"
        assert S == self.input_sensors, f"Expected {self.input_sensors} sensors, got {S}"
        assert T == self.input_timesteps, f"Expected {self.input_timesteps} timesteps, got {T}"
        
        # Ensure float32
        x = x.to(torch.float32)
        
        # Encode to continuous features
        features = self.encoder(x)  # (B, num_codes_per_dim, T)
        
        # Quantize
        quantized, codes = self._quantize(features)
        
        # Straight-through estimator for gradients
        if self.training:
            quantized = features + (quantized - features).detach()
            
        # Project to embedding dimension
        quantized = self.projection(quantized)  # (B, embedding_dim, T)
        
        # Update statistics (always update, not just during training for testing)
        self._update_statistics(codes)
            
        # Create output
        output = CodesAndFeatures(
            codes=codes,
            quantized=quantized,
            perplexity=self.perplexity,
            usage_histogram=self.usage_histogram,
            codebook_size=self.codebook_size,
        )
        
        return output
    
    def reset_statistics(self) -> None:
        """Reset code usage statistics."""
        self.code_counts.zero_()
        self.total_samples.zero_()


# Global encoder instance for functional interface
_global_encoder: Optional[FSQEncoder] = None


def encode_fsq(
    x: Tensor,
    levels: List[int] = [8, 6, 5],
    embedding_dim: int = 64,
    reset_stats: bool = False,
) -> CodesAndFeatures:
    """Single function interface for FSQ encoding.
    
    This is the main API for FSQ encoding. It ensures:
    - Deterministic behavior (same input → same codes)
    - Proper shape validation
    - Type enforcement (float32/int32)
    - Statistics tracking
    
    Args:
        x: Input tensor of shape (B, 9, 2, 100)
        levels: Quantization levels per dimension
        embedding_dim: Dimension of output features
        reset_stats: Whether to reset statistics before encoding
        
    Returns:
        CodesAndFeatures with:
        - codes: int32[B, T=100]
        - quantized: float32[B, D=embedding_dim, T=100]
        - perplexity: float
        - usage_histogram: array of code frequencies
        - codebook_size: int
        
    Example:
        >>> x = torch.randn(32, 9, 2, 100)
        >>> result = encode_fsq(x)
        >>> assert result.codes.shape == (32, 100)
        >>> assert result.codes.dtype == torch.int32
        >>> assert result.quantized.shape == (32, 64, 100)
        >>> assert result.quantized.dtype == torch.float32
    """
    global _global_encoder
    
    # Create or reuse encoder
    if (_global_encoder is None or 
        _global_encoder.levels != levels or 
        _global_encoder.embedding_dim != embedding_dim):
        _global_encoder = FSQEncoder(
            levels=levels,
            embedding_dim=embedding_dim,
            seed=42,  # Deterministic
        )
        _global_encoder.eval()  # Default to eval mode
        
    if reset_stats:
        _global_encoder.reset_statistics()
        
    # Encode
    with torch.no_grad():
        output = _global_encoder(x)
        
    # Log statistics
    unique_codes = len(torch.unique(output.codes))
    logger.info(
        f"FSQ encoding complete: {unique_codes}/{output.codebook_size} codes used, "
        f"perplexity={output.perplexity:.2f}"
    )
    
    return output


def verify_fsq_invariants(
    encoder: FSQEncoder,
    x: Tensor,
    tolerance: float = 1e-6,
) -> bool:
    """Verify FSQ encoder invariants.
    
    Checks:
    1. Determinism: Same input → same codes
    2. Shape contracts: (B,9,2,100) → (B,T), (B,D,T)
    3. Type contracts: int32 codes, float32 features
    4. Code range: 0 <= codes < codebook_size
    
    Args:
        encoder: FSQ encoder to verify
        x: Test input of shape (B, 9, 2, 100)
        tolerance: Floating point tolerance for comparisons
        
    Returns:
        True if all invariants hold
    """
    try:
        # Check 1: Determinism
        encoder.eval()
        with torch.no_grad():
            result1 = encoder(x)
            result2 = encoder(x)
            
        if not torch.equal(result1.codes, result2.codes):
            logger.error("Determinism violated: different codes for same input")
            return False
            
        if not torch.allclose(result1.quantized, result2.quantized, atol=tolerance):
            logger.error("Determinism violated: different features for same input")
            return False
            
        # Check 2: Shape contracts
        B, C, S, T = x.shape
        if result1.codes.shape != (B, T):
            logger.error(f"Code shape violation: expected ({B}, {T}), got {result1.codes.shape}")
            return False
            
        if result1.quantized.shape[0] != B or result1.quantized.shape[2] != T:
            logger.error(f"Feature shape violation: batch or time dimension mismatch")
            return False
            
        # Check 3: Type contracts
        if result1.codes.dtype != torch.int32:
            logger.error(f"Code dtype violation: expected int32, got {result1.codes.dtype}")
            return False
            
        if result1.quantized.dtype != torch.float32:
            logger.error(f"Feature dtype violation: expected float32, got {result1.quantized.dtype}")
            return False
            
        # Check 4: Code range
        if result1.codes.min() < 0 or result1.codes.max() >= encoder.codebook_size:
            logger.error(f"Code range violation: codes not in [0, {encoder.codebook_size})")
            return False
            
        logger.info("All FSQ invariants verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"Invariant verification failed: {e}")
        return False