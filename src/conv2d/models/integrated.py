"""Integrated Conv2d-FSQ-HSMM model."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from conv2d.models.conv2d_fsq import Conv2dFSQModel
from conv2d.models.hsmm import HSMMComponents


class Conv2dFSQHSMM(nn.Module):
    """Complete Conv2d-FSQ-HSMM model."""
    
    def __init__(
        self,
        input_channels: int = 9,
        hidden_channels: int = 64,
        fsq_levels: list[int] | None = None,
        embedding_dim: int = 64,
        num_states: int = 32,
    ) -> None:
        """Initialize integrated model.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
            fsq_levels: FSQ quantization levels
            embedding_dim: Embedding dimension
            num_states: Number of HSMM states
        """
        super().__init__()
        
        # Conv2d-FSQ encoder
        self.conv2d_fsq = Conv2dFSQModel(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            fsq_levels=fsq_levels,
            embedding_dim=embedding_dim,
        )
        
        # HSMM temporal model
        self.hsmm = HSMMComponents(
            num_states=num_states,
            observation_dim=embedding_dim,
        )
        
    def forward(
        self, x: Tensor
    ) -> dict[str, Tensor]:
        """Forward pass through complete model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary of outputs and metrics
        """
        # Conv2d-FSQ encoding
        reconstruction, codes, fsq_metrics = self.conv2d_fsq(x)
        
        # Reshape for temporal modeling
        B, C, H, W = x.shape
        features = reconstruction.flatten(start_dim=2).transpose(1, 2)
        
        # HSMM temporal modeling
        states, durations, hsmm_metrics = self.hsmm(features)
        
        # Combine outputs
        outputs = {
            "reconstruction": reconstruction,
            "codes": codes,
            "states": states,
            "durations": durations,
            "fsq_perplexity": fsq_metrics["perplexity"],
            "fsq_usage": fsq_metrics["usage_rate"],
            "hsmm_entropy": hsmm_metrics["entropy"],
        }
        
        return outputs