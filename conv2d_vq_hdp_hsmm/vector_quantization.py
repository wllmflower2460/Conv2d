"""
Vector Quantization layer with straight-through estimator.
Uses Conv2d operations only for edge deployment compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantization(nn.Module):
    """
    Vector Quantization layer with straight-through estimator.
    
    Args:
        num_embeddings: Number of codebook vectors (512)
        embedding_dim: Dimension of each codebook vector (64)
        commitment_cost: Weight for commitment loss (0.25)
    """
    
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(VectorQuantization, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook - using Conv2d with 1x1 kernel for edge compatibility
        self.codebook = nn.Conv2d(embedding_dim, num_embeddings, kernel_size=1, bias=False)
        
        # Initialize codebook vectors
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # For perplexity calculation
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', torch.zeros(embedding_dim, num_embeddings, 1, 1))
    
    def forward(self, inputs):
        """
        Forward pass with straight-through estimator.
        
        Args:
            inputs: Input tensor of shape (B, embedding_dim, H, W)
            
        Returns:
            quantized: Quantized tensor
            vq_loss: VQ loss (commitment + codebook)
            perplexity: Perplexity of the codebook usage
            encodings: One-hot encodings of selected codes
        """
        # Get input shape
        B, C, H, W = inputs.shape
        
        # Flatten spatial dimensions for distance calculation
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Get codebook weights - transpose for compatibility
        codebook_weights = self.codebook.weight.data.squeeze(-1).squeeze(-1).t()  # (embedding_dim, num_embeddings)
        
        # Calculate distances using Conv2d approach
        distances = torch.cdist(flat_inputs, codebook_weights.t())  # (B*H*W, num_embeddings)
        
        # Find closest codes
        encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # (B*H*W, num_embeddings)
        
        # Quantize using codebook lookup
        quantized_flat = torch.matmul(encodings, codebook_weights.t())  # (B*H*W, embedding_dim)
        quantized = quantized_flat.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2)  # (B, embedding_dim, H, W)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Update moving averages for codebook learning (EMA)
        if self.training:
            with torch.no_grad():  # Prevent gradient tracking for EMA updates
                encodings_sum = encodings.sum(0)
                self.cluster_size.data.mul_(0.99).add_(encodings_sum, alpha=0.01)
                
                embed_sum = torch.matmul(encodings.t(), flat_inputs)  # (num_embeddings, embedding_dim)
                embed_sum_reshaped = embed_sum.t().unsqueeze(-1).unsqueeze(-1)  # (embedding_dim, num_embeddings, 1, 1)
                self.embed_avg.data.mul_(0.99).add_(embed_sum_reshaped, alpha=0.01)
                
                # Update codebook weights
                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                cluster_size_reshaped = cluster_size.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, num_embeddings, 1, 1)
                embed_normalized = self.embed_avg / cluster_size_reshaped
                self.codebook.weight.data.copy_(embed_normalized.permute(1, 0, 2, 3))
        
        # Calculate perplexity
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape encodings back to spatial format
        encodings = encodings.view(B, H, W, self.num_embeddings).permute(0, 3, 1, 2)
        
        return quantized, vq_loss, perplexity, encodings
    
    def get_codebook_usage(self):
        """Get codebook usage statistics."""
        return {
            'cluster_size': self.cluster_size.clone(),
            'active_codes': (self.cluster_size > 0).sum().item(),
            'utilization': (self.cluster_size > 0).float().mean().item()
        }