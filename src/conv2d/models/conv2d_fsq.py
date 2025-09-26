"""Conv2d-FSQ model implementation."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from conv2d.models.fsq_layer import FSQLayer


class Conv2dFSQModel(nn.Module):
    """Conv2d model with FSQ quantization."""
    
    def __init__(
        self,
        input_channels: int = 9,
        hidden_channels: int = 64,
        fsq_levels: list[int] | None = None,
        embedding_dim: int = 64,
    ) -> None:
        """Initialize Conv2d-FSQ model.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
            fsq_levels: FSQ quantization levels
            embedding_dim: Embedding dimension
        """
        super().__init__()
        
        if fsq_levels is None:
            fsq_levels = [8, 6, 5]  # Default: 240 codes
            
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, embedding_dim, 1),
        )
        
        # FSQ layer
        self.fsq = FSQLayer(fsq_levels, embedding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, input_channels, 3, padding=1),
        )
        
    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (output, codes, metrics)
        """
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_q, codes, metrics = self.fsq(z)
        
        # Decode
        output = self.decoder(z_q)
        
        return output, codes, metrics