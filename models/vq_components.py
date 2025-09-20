"""
Vector Quantization Components for Conv2d-VQ-HDP-HSMM
Implements discrete representation learning for behavioral primitives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class VectorQuantizer(nn.Module):
    """
    Basic Vector Quantizer with straight-through estimator
    Converts continuous representations to discrete codes
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.0,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # Track codebook usage for perplexity calculation
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.codebook.weight.data.clone())
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            z: Tensor of shape (B, C, H, W) or (B, C, T) for Conv2d/Conv1d
        
        Returns:
            quantized: Quantized output with same shape as input
            indices: Codebook indices
            metrics: Dictionary containing loss and perplexity
        """
        # Store original shape and flatten
        original_shape = z.shape
        B, C = z.shape[:2]
        
        # Reshape to (B*H*W, C) or (B*T, C)
        z_flat = z.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)
        
        # Compute distances to codebook vectors
        distances = torch.cdist(z_flat, self.codebook.weight, p=2)
        
        # Find nearest codebook entries
        indices = torch.argmin(distances, dim=1)
        
        # Quantize
        quantized_flat = self.codebook(indices)
        quantized = quantized_flat.reshape(B, -1, C).permute(0, 2, 1).reshape(original_shape)
        
        # Compute losses
        if self.training:
            # Commitment loss - encourages encoder to commit to codebook entries
            commitment_loss = F.mse_loss(z, quantized.detach())
            
            # Codebook loss - moves codebook vectors towards encoder outputs
            codebook_loss = F.mse_loss(quantized, z.detach())
            
            # Combined VQ loss
            vq_loss = codebook_loss + self.commitment_cost * commitment_loss
            
            # Update usage statistics
            with torch.no_grad():
                self.usage_count.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
        else:
            vq_loss = torch.tensor(0.0, device=z.device)
        
        # Straight-through estimator: gradients pass through unchanged
        quantized = z + (quantized - z).detach()
        
        # Calculate perplexity (measure of codebook utilization)
        perplexity = self.calculate_perplexity(indices)
        
        metrics = {
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'active_codes': len(torch.unique(indices)),
            'commitment_loss': commitment_loss if self.training else torch.tensor(0.0),
            'codebook_loss': codebook_loss if self.training else torch.tensor(0.0)
        }
        
        return quantized, indices, metrics
    
    def calculate_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """Calculate perplexity to measure codebook utilization"""
        # Get unique indices and their counts
        unique_indices, counts = torch.unique(indices, return_counts=True)
        
        # Calculate probabilities
        probs = counts.float() / indices.shape[0]
        
        # Calculate perplexity
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        
        return perplexity
    
    def get_codebook_usage_stats(self) -> Dict:
        """Get statistics about codebook usage"""
        total_usage = self.usage_count.sum()
        if total_usage > 0:
            usage_probs = self.usage_count / total_usage
            active_codes = (self.usage_count > 0).sum().item()
            usage_entropy = -torch.sum(
                usage_probs * torch.log(usage_probs + 1e-10)
            ).item()
        else:
            active_codes = 0
            usage_entropy = 0.0
        
        return {
            'active_codes': active_codes,
            'usage_entropy': usage_entropy,
            'usage_fraction': active_codes / self.num_embeddings,
            'most_used_code': torch.argmax(self.usage_count).item(),
            'least_used_codes': torch.nonzero(self.usage_count == 0).squeeze(-1).tolist()[:10]
        }


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average codebook updates
    More stable training than standard VQ
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize embeddings
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', embedding.clone())
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        EMA-based vector quantization
        """
        original_shape = z.shape
        B, C = z.shape[:2]
        
        # Flatten input
        z_flat = z.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)
        
        # Calculate distances
        distances = torch.cdist(z_flat, self.embedding, p=2)
        
        # Get nearest neighbors
        indices = torch.argmin(distances, dim=1)
        indices_onehot = F.one_hot(indices, self.num_embeddings).float()
        
        # Quantize
        quantized_flat = torch.matmul(indices_onehot, self.embedding)
        quantized = quantized_flat.reshape(B, -1, C).permute(0, 2, 1).reshape(original_shape)
        
        # Update embeddings with EMA
        if self.training:
            # Update cluster sizes
            n = indices_onehot.sum(dim=0)
            self.cluster_size.data.mul_(self.decay).add_(n, alpha=1-self.decay)
            
            # Update embedding averages
            embed_sum = torch.matmul(indices_onehot.T, z_flat)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
            
            # Update embeddings
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)
            
            # Update usage statistics
            with torch.no_grad():
                self.usage_count.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
            
            # Commitment loss only (no codebook loss needed with EMA)
            commitment_loss = F.mse_loss(z, quantized.detach())
            vq_loss = self.commitment_cost * commitment_loss
        else:
            vq_loss = torch.tensor(0.0, device=z.device)
            commitment_loss = torch.tensor(0.0, device=z.device)
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        # Calculate perplexity
        perplexity = self.calculate_perplexity(indices)
        
        metrics = {
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'active_codes': len(torch.unique(indices)),
            'commitment_loss': commitment_loss,
            'cluster_size_mean': self.cluster_size.mean().item() if self.training else 0.0
        }
        
        return quantized, indices, metrics
    
    def calculate_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """Calculate perplexity to measure codebook utilization"""
        unique_indices, counts = torch.unique(indices, return_counts=True)
        probs = counts.float() / indices.shape[0]
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        return perplexity
    
    def get_codebook_usage_stats(self) -> Dict:
        """Get statistics about codebook usage"""
        total_usage = self.usage_count.sum()
        if total_usage > 0:
            usage_probs = self.usage_count / total_usage
            active_codes = (self.usage_count > 0).sum().item()
            usage_entropy = -torch.sum(
                usage_probs * torch.log(usage_probs + 1e-10)
            ).item()
        else:
            active_codes = 0
            usage_entropy = 0.0
        
        return {
            'active_codes': active_codes,
            'usage_entropy': usage_entropy,
            'usage_fraction': active_codes / self.num_embeddings,
            'cluster_sizes': self.cluster_size.tolist()[:20],  # First 20 for inspection
        }


class VQ2D(nn.Module):
    """
    2D Vector Quantizer optimized for Conv2d behavioral data
    Special handling for (Batch, Channels, Devices=2, Time) format
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.use_ema = use_ema
        
        if use_ema:
            self.vq = VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay
            )
        else:
            self.vq = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost
            )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """
        Args:
            z: (B, C, H=2, W=T) where H=2 represents human and dog devices
        
        Returns:
            quantized: Quantized tensor with same shape
            tokens: Dictionary with 'human' and 'dog' token sequences
            metrics: VQ metrics including per-device statistics
        """
        B, C, H, W = z.shape
        assert H == 2, f"Expected H=2 for human+dog, got H={H}"
        
        # Quantize entire tensor
        quantized, indices, metrics = self.vq(z)
        
        # Reshape indices to separate devices
        indices_reshaped = indices.reshape(B, H, W)
        
        # Extract per-device tokens
        tokens = {
            'human': indices_reshaped[:, 0, :],  # (B, T)
            'dog': indices_reshaped[:, 1, :],     # (B, T)
            'combined': indices  # Flattened version
        }
        
        # Calculate per-device perplexity
        with torch.no_grad():
            human_perplexity = self.vq.calculate_perplexity(tokens['human'].flatten())
            dog_perplexity = self.vq.calculate_perplexity(tokens['dog'].flatten())
            
            # Measure token overlap between human and dog
            human_unique = set(tokens['human'].flatten().tolist())
            dog_unique = set(tokens['dog'].flatten().tolist())
            overlap = len(human_unique & dog_unique)
            
            metrics['human_perplexity'] = human_perplexity
            metrics['dog_perplexity'] = dog_perplexity
            metrics['token_overlap'] = overlap
            metrics['token_overlap_ratio'] = overlap / max(len(human_unique), len(dog_unique), 1)
        
        return quantized, tokens, metrics


def test_vq_components():
    """Test VQ components with dummy data"""
    print("Testing VectorQuantizer components...")
    
    # Test dimensions matching Conv2d output
    B, C, H, W = 4, 256, 2, 100  # Batch, Channels, Devices, Time
    
    # Test basic VQ
    vq = VectorQuantizer(num_embeddings=512, embedding_dim=256)
    z = torch.randn(B, C, H, W)
    quantized, indices, metrics = vq(z)
    
    assert quantized.shape == z.shape, f"Shape mismatch: {quantized.shape} vs {z.shape}"
    assert indices.shape[0] == B * H * W, f"Indices shape error: {indices.shape}"
    print(f"✓ Basic VQ: perplexity={metrics['perplexity']:.2f}, active={metrics['active_codes']}")
    
    # Test VQ-EMA
    vq_ema = VectorQuantizerEMA(num_embeddings=512, embedding_dim=256)
    quantized_ema, indices_ema, metrics_ema = vq_ema(z)
    
    assert quantized_ema.shape == z.shape
    print(f"✓ VQ-EMA: perplexity={metrics_ema['perplexity']:.2f}, active={metrics_ema['active_codes']}")
    
    # Test VQ2D
    vq2d = VQ2D(num_embeddings=512, embedding_dim=256, use_ema=True)
    quantized_2d, tokens, metrics_2d = vq2d(z)
    
    assert quantized_2d.shape == z.shape
    assert tokens['human'].shape == (B, W)
    assert tokens['dog'].shape == (B, W)
    print(f"✓ VQ2D: human_perplexity={metrics_2d['human_perplexity']:.2f}, "
          f"dog_perplexity={metrics_2d['dog_perplexity']:.2f}, "
          f"overlap_ratio={metrics_2d['token_overlap_ratio']:.2f}")
    
    # Test gradient flow
    loss = quantized.mean() + metrics['vq_loss']
    loss.backward()
    print("✓ Gradient flow verified")
    
    print("\nAll VQ component tests passed!")
    return True


if __name__ == "__main__":
    test_vq_components()