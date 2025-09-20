# models/vq_ema_2d.py
# Vector Quantizer (EMA + Straight-Through) for Conv2d feature maps
# Hailo-safe: no Conv1d, no GroupNorm/LayerNorm, no Softmax
# Expected encoder output shape: (B, D, H, T)  e.g., (32, 64, 1, 100)

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA2D(nn.Module):
    """
    EMA Vector Quantization for 2D feature maps (H,T), with straight-through estimator.

    Args:
        num_codes: size of codebook (e.g., 512)
        code_dim: embedding dimension (must match encoder output channel D)
        decay: EMA decay for codebook updates (e.g., 0.99–0.999)
        epsilon: small constant to avoid division by zero
        commitment_cost: beta for commitment loss (e.g., 0.25–0.4)
        init_scale: scale for codebook init (kaiming-uniform-ish)
    Input:
        z_e: (B, D, H, T) continuous encoder features
    Output:
        z_q: (B, D, H, T) quantized features (straight-through)
        loss_dict: {"commitment": ..., "codebook": ... (0 here, EMA updates), "vq": total}
        info: {"indices": (B, H, T), "perplexity": float, "usage": float[0-1]}
    """
    def __init__(
        self,
        num_codes: int = 512,
        code_dim: int = 64,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        commitment_cost: float = 0.25,
        init_scale: float = 1.0,
        dead_code_threshold: int = 100,
        enable_dead_code_refresh: bool = False
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.epsilon = epsilon
        self.beta = commitment_cost
        self.dead_code_threshold = dead_code_threshold
        self.enable_dead_code_refresh = enable_dead_code_refresh

        # Codebook: (num_codes, code_dim)
        embed = torch.randn(num_codes, code_dim)
        embed = F.normalize(embed, dim=1) * init_scale
        self.register_buffer("embedding", embed)  # updated via EMA (not a Parameter)

        # EMA state
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embed_sum", torch.zeros(num_codes, code_dim))

        # For numerical stability in cluster size debiasing
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
        
        # Track code usage for dead code detection
        self.register_buffer("code_usage_count", torch.zeros(num_codes))
        self.register_buffer("total_usage_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _ema_update(self, flat_z_e: torch.Tensor, onehot: torch.Tensor):
        """
        EMA update of codebook using current minibatch assignments.
        flat_z_e: (N, D)
        onehot:   (N, K)
        """
        self.steps += 1

        # Batch counts per code: (K,)
        cluster_size = onehot.sum(dim=0)  # counts
        # Sum of vectors assigned to each code: (K, D)
        embed_sum = onehot.t() @ flat_z_e

        # Momentum updates
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
        self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

        # Laplace smoothing for cluster sizes
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon) /
            (n + self.num_codes * self.epsilon) * n
        )

        # Normalize embeddings by cluster size
        embed_normalized = self.ema_embed_sum / cluster_size.unsqueeze(1)
        # Handle empty codes by preserving previous embeddings
        cluster_mask = (self.ema_cluster_size > 0).float().unsqueeze(1)
        self.embedding = (
            cluster_mask * embed_normalized +
            (1.0 - cluster_mask) * self.embedding
        )

        # Normalize embeddings to keep scale bounded (optional but stabilizing)
        self.embedding = F.normalize(self.embedding, dim=1)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """
        z_e: (B, D, H, T)
        """
        assert z_e.dim() == 4, f"Expected 4D (B, D, H, T), got {z_e.shape}"
        B, D, H, T = z_e.shape
        assert D == self.code_dim, f"Encoder channel dim D={D} must equal code_dim={self.code_dim}"

        # Flatten spatial dims and move channels last for distance calc
        # z_e_perm: (B, H, T, D) → flat_z_e: (N, D) where N = B*H*T
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()       # (B, H, T, D)
        flat_z_e = z_e_perm.view(-1, D)                       # (N, D)

        # Compute L2 distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2⟨z,e⟩
        # Shape: (N, 1) + (1, K) - 2(N, D)(D, K) = (N, K)
        with torch.no_grad():
            z_squared = (flat_z_e ** 2).sum(dim=1, keepdim=True)  # (N, 1)
            e_squared = (self.embedding ** 2).sum(dim=1, keepdim=True).t()  # (1, K)
            dot_product = flat_z_e @ self.embedding.t()  # (N, K)

            distances = z_squared + e_squared - 2.0 * dot_product
            indices = torch.argmin(distances, dim=1)  # (N,)

        # One-hot assignments (no softmax; Hailo-safe)
        onehot = F.one_hot(indices, num_classes=self.num_codes).type_as(flat_z_e)  # (N, K)

        # Quantized vectors via codebook lookup: (N, D)
        z_q_flat = onehot @ self.embedding   # (N, D)

        # Straight-through estimator: z_q + (z_e - z_q).detach()
        # Grad flows to z_e; codebook grads managed by EMA.
        z_q = z_q_flat.view(B, H, T, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, T)
        z_q_st = z_e + (z_q - z_e).detach()

        # Commitment loss: encourage encoder outputs to commit to embeddings
        # Note: codebook loss is handled by EMA; no gradient to embedding.
        commitment_loss = self.beta * F.mse_loss(z_e.detach(), z_q)  # ||sg[z_e] - e||^2
        codebook_loss = torch.tensor(0.0, device=z_e.device)

        # Compute codebook statistics
        with torch.no_grad():
            # Average assignment probabilities across all spatial positions
            avg_probs = onehot.float().mean(dim=0)  # (K,)

            # Perplexity: exp(-sum(p * log(p)))
            eps = 1e-12
            log_probs = (avg_probs + eps).log()
            entropy = -(avg_probs * log_probs).sum()
            perplexity = torch.exp(entropy)

            # Code utilization: fraction of codes used in this batch
            usage = (avg_probs > 0).float().mean()
            
            # Track which codes are used
            unique_codes = torch.unique(indices)
            active_codes = len(unique_codes)
            
            # Update usage counts
            if self.training:
                self.total_usage_updates += 1
                for code_idx in unique_codes:
                    self.code_usage_count[code_idx] += 1
                
                # Dead code refresh (optional)
                if self.enable_dead_code_refresh and self.total_usage_updates % self.dead_code_threshold == 0:
                    # Find codes that haven't been used recently
                    dead_codes = (self.code_usage_count == 0).nonzero(as_tuple=True)[0]
                    if len(dead_codes) > 0:
                        # Refresh dead codes with random samples from current batch
                        n_refresh = min(len(dead_codes), flat_z_e.shape[0])
                        random_indices = torch.randperm(flat_z_e.shape[0])[:n_refresh]
                        self.embedding[dead_codes[:n_refresh]] = flat_z_e[random_indices].detach()
                        # Reset their usage counts
                        self.code_usage_count[dead_codes[:n_refresh]] = 1
                    
                    # Reset usage counts every threshold steps to track recent usage
                    if self.total_usage_updates % (self.dead_code_threshold * 10) == 0:
                        self.code_usage_count.fill_(0)

            # EMA update during training only
            if self.training:
                self._ema_update(flat_z_e, onehot)

        loss = commitment_loss + codebook_loss

        info = {
            "indices": indices.view(B, H, T),
            "perplexity": perplexity.detach(),
            "usage": usage.detach(),
            "active_codes": active_codes,
            "code_histogram": avg_probs.detach(),
            "dead_codes": (self.code_usage_count == 0).sum().item(),
            "entropy": entropy.detach()
        }
        loss_dict = {"commitment": commitment_loss, "codebook": codebook_loss, "vq": loss}
        return z_q_st, loss_dict, info


class VQHead2D(nn.Module):
    """
    Optional 1x1 Conv projection to code_dim before VQ.
    Keeps everything Conv2d-compatible and static-shaped.

    Use only if your encoder's channel D != code_dim.
    """
    def __init__(self, in_dim: int, code_dim: int, bias: bool = True):
        super().__init__()
        # 1x1 conv projection: (B, in_dim, H, T) -> (B, code_dim, H, T)
        self.proj = nn.Conv2d(in_dim, code_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, T)
        return self.proj(x)


def _test_vq_model():
    """Self-test with typical Conv2d shapes for Hailo deployment."""
    torch.manual_seed(42)
    B, D, H, T = 32, 64, 1, 100  # Batch, Channels, Height, Time
    num_codes = 512

    # Test input tensor
    z = torch.randn(B, D, H, T)

    # Initialize VQ model
    vq = VectorQuantizerEMA2D(
        num_codes=num_codes,
        code_dim=D,
        decay=0.99,
        commitment_cost=0.25
    )
    vq.train()

    # Forward pass
    z_q, loss_dict, info = vq(z)

    # Verify outputs
    assert z_q.shape == z.shape, f"Shape mismatch: {z_q.shape} != {z.shape}"
    assert info["indices"].shape == (B, H, T), f"Indices shape: {info['indices'].shape}"

    print(f"✓ VQ output shape: {z_q.shape}")
    print(f"✓ VQ loss: {loss_dict['vq']:.4f}")
    print(f"✓ Perplexity: {info['perplexity']:.2f}")
    print(f"✓ Code usage: {info['usage']:.2%}")

    # Test VQHead2D
    head = VQHead2D(in_dim=128, code_dim=D)
    x = torch.randn(B, 128, H, T)
    y = head(x)
    assert y.shape == (B, D, H, T), f"Head output shape: {y.shape}"
    print(f"✓ VQHead2D: {x.shape} -> {y.shape}")


if __name__ == "__main__":
    _test_vq_model()
