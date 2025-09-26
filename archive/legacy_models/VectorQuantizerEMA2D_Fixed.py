# VectorQuantizerEMA2D_Fixed.py
# Fixed VQ-EMA with proper gradient detachment and loss clamping
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA2D_Fixed(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, decay: float = 0.99, commitment_cost: float = 0.25,
                 eps: float = 1e-5, l2_normalize_input: bool = False, restart_dead_codes: bool = True,
                 dead_code_threshold: float = 0.01):
        super().__init__()
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.decay = float(decay)
        self.beta = float(commitment_cost)
        self.eps = float(eps)
        self.l2_normalize_input = bool(l2_normalize_input)
        self.restart_dead_codes = bool(restart_dead_codes)
        self.dead_code_threshold = float(dead_code_threshold)
        
        # Codebook (buffer; EMA updates)
        embedding = torch.randn(self.num_codes, self.code_dim) / (self.code_dim ** 0.5)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_codes))
        self.register_buffer("ema_cluster_sum", torch.zeros(self.num_codes, self.code_dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _assign(self, z_flat):
        # Compute L2 distances to codebook
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)              # (N, 1)
        e_sq = (self.embedding ** 2).sum(dim=1).unsqueeze(0)       # (1, K)
        z_e = z_flat @ self.embedding.t()                          # (N, K)
        dists = z_sq - 2 * z_e + e_sq                              # (N, K)
        indices = torch.argmin(dists, dim=1)                       # (N,)
        one_hot = F.one_hot(indices, num_classes=self.num_codes).float()  # (N, K)
        return indices, one_hot

    @torch.no_grad()
    def _ema_update(self, z_flat, one_hot):
        # CRITICAL: Use detached inputs for EMA updates
        z_flat = z_flat.detach()
        
        cluster_size = one_hot.sum(dim=0)                          # (K,)
        cluster_sum = one_hot.t() @ z_flat                         # (K, D)
        
        # Update with decay
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size * (1.0 - self.decay))
        self.ema_cluster_sum.mul_(self.decay).add_(cluster_sum * (1.0 - self.decay))
        
        # Normalize to update embedding
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_codes * self.eps)
        new_embed = self.ema_cluster_sum / cluster_size.unsqueeze(1).clamp(min=self.eps)
        self.embedding.copy_(new_embed)
        
        # Dead code reinit
        if self.restart_dead_codes:
            dead = self.ema_cluster_size < (self.dead_code_threshold * n / self.num_codes)
            if dead.any():
                n_dead = int(dead.sum().item())
                # Sample random vectors from current batch
                idx = torch.randint(0, z_flat.shape[0], (n_dead,), device=z_flat.device)
                noise = torch.randn(n_dead, self.code_dim, device=z_flat.device) * 0.01
                self.embedding[dead] = z_flat[idx] + noise
                # Reset EMA stats for reinitialized codes
                self.ema_cluster_size[dead] = 1.0
                self.ema_cluster_sum[dead] = self.embedding[dead].clone()

    def forward(self, z_e):
        # z_e: (B, D, 1, T) or (B, D, T) or (B, D, H, W)
        input_shape = z_e.shape
        
        # Handle different input dimensions
        if z_e.dim() == 3:
            # (B, D, T)
            B, D, T = z_e.shape
            z = z_e.permute(0, 2, 1).contiguous()       # (B, T, D)
            z_flat = z.view(-1, D)
        elif z_e.dim() == 4:
            if z_e.shape[2] == 1:
                # (B, D, 1, T)
                B, D, _, T = z_e.shape
                z = z_e.squeeze(2).permute(0, 2, 1).contiguous()  # (B, T, D)
                z_flat = z.view(-1, D)
            else:
                # (B, D, H, W)
                B, D, H, W = z_e.shape
                z = z_e.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
                z_flat = z.view(-1, D)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {z_e.dim()}D")
        
        # Optional L2 normalization
        if self.l2_normalize_input:
            z_flat = F.normalize(z_flat, dim=1)
        
        # Get assignments
        with torch.no_grad():
            indices, one_hot = self._assign(z_flat)
        
        # Quantized vectors from codebook
        z_q_flat = one_hot @ self.embedding                         # (N, D)
        
        # EMA update (only during training)
        if self.training:
            self._ema_update(z_flat, one_hot)
        
        # Straight-through estimator with proper stop-gradient
        z_q_flat_st = z_flat + (z_q_flat - z_flat).detach()
        
        # CRITICAL FIX: Clamp losses to prevent explosion
        commitment = self.beta * torch.mean((z_flat.detach() - z_q_flat) ** 2)
        codebook = torch.mean((z_flat - z_q_flat.detach()) ** 2)
        
        # Clamp individual losses before combining
        commitment = torch.clamp(commitment, max=100.0)
        codebook = torch.clamp(codebook, max=100.0)
        
        vq_loss = commitment + codebook
        
        # Reshape back to original dimensions
        if z_e.dim() == 3:
            z_q = z_q_flat_st.view(B, T, D).permute(0, 2, 1).contiguous()
        elif z_e.dim() == 4:
            if input_shape[2] == 1:
                z_q_bt = z_q_flat_st.view(B, T, D).permute(0, 2, 1).contiguous()
                z_q = z_q_bt.unsqueeze(2)
            else:
                z_q_bhw = z_q_flat_st.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
                z_q = z_q_bhw
        
        # Calculate statistics
        with torch.no_grad():
            counts = one_hot.sum(dim=0)                            # (K,)
            probs = counts / max(float(z_flat.shape[0]), 1.0)
            probs = torch.clamp(probs, min=self.eps)
            entropy = -torch.sum(probs * torch.log(probs))
            perplexity = torch.exp(entropy)
            usage = (counts > 0).float().mean()
            
            # Track actual unique codes used
            unique_codes = len(torch.unique(indices))
        
        # Prepare output info
        if z_e.dim() == 3:
            idx_bt = indices.view(B, T).unsqueeze(1)
        elif z_e.dim() == 4 and input_shape[2] == 1:
            idx_bt = indices.view(B, T).unsqueeze(1)
        else:
            idx_bt = indices.view(B, H * W).unsqueeze(1)
        
        info = {
            "indices": idx_bt, 
            "perplexity": float(perplexity.item()), 
            "usage": float(usage.item()),
            "unique_codes": unique_codes
        }
        loss_dict = {
            "vq": vq_loss, 
            "commitment": commitment, 
            "codebook": codebook
        }
        
        return z_q, loss_dict, info