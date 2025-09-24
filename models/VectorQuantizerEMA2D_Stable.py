# VectorQuantizerEMA2D_Stable.py
# Drop-in VQ-EMA module with robust EMA updates, dead-code reinit, and proper perplexity.
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA2D_Stable(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, decay: float = 0.99, commitment_cost: float = 0.4,
                 eps: float = 1e-5, l2_normalize_input: bool = True, restart_dead_codes: bool = True,
                 dead_code_threshold: float = 1e-3):
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

    @torch.no_grad()
    def _assign(self, z_flat):
        # Compute L2 distances to codebook: ||z||^2 - 2 z e^T + ||e||^2
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)              # (N, 1)
        e_sq = (self.embedding ** 2).sum(dim=1).unsqueeze(0)       # (1, K)
        z_e = z_flat @ self.embedding.t()                          # (N, K)
        dists = z_sq - 2 * z_e + e_sq                              # (N, K)
        indices = torch.argmin(dists, dim=1)                       # (N,)
        one_hot = F.one_hot(indices, num_classes=self.num_codes).float()  # (N, K)
        return indices, one_hot

    @torch.no_grad()
    def _ema_update(self, z_flat, one_hot):
        cluster_size = one_hot.sum(dim=0)                          # (K,)
        cluster_sum = one_hot.t() @ z_flat                         # (K, D)
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size * (1.0 - self.decay))
        self.ema_cluster_sum.mul_(self.decay).add_(cluster_sum * (1.0 - self.decay))
        # Normalize to avoid zeros
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_codes * self.eps)
        new_embed = self.ema_cluster_sum / cluster_size.unsqueeze(1)
        self.embedding.copy_(new_embed)
        # Dead code reinit
        if self.restart_dead_codes:
            dead = self.ema_cluster_size < self.dead_code_threshold
            if dead.any():
                idx = torch.randint(0, z_flat.shape[0], (int(dead.sum().item()),), device=z_flat.device)
                self.embedding[dead] = z_flat[idx]

    def forward(self, z_e):
        # z_e: (B, D, 1, T) or (B, D, T)
        assert z_e.dim() in (3, 4), "Expected (B, D, T) or (B, D, 1, T)"
        if z_e.dim() == 4:
            B, D, _, T = z_e.shape
            z = z_e.squeeze(2).permute(0, 2, 1).contiguous()       # (B, T, D)
            z_flat = z.view(B * T, D)
        else:
            B, D, T = z_e.shape
            z = z_e.permute(0, 2, 1).contiguous()
            z_flat = z.view(B * T, D)
        if self.l2_normalize_input:
            z_flat = F.normalize(z_flat, dim=1)
        with torch.no_grad():
            indices, one_hot = self._assign(z_flat)
        z_q_flat = one_hot @ self.embedding                         # (N, D)
        if self.training:
            self._ema_update(z_flat.detach(), one_hot)
        # Straight-through
        z_q_flat_st = z_flat + (z_q_flat - z_flat).detach()
        commitment = self.beta * torch.mean((z_flat - z_q_flat.detach()) ** 2)
        codebook = torch.mean((z_flat.detach() - z_q_flat) ** 2)
        vq_loss = commitment + codebook
        # Reshape back
        z_q_bt = z_q_flat_st.view(B, T, D).permute(0, 2, 1).contiguous()  # (B, D, T)
        if z_e.dim() == 4:
            z_q = z_q_bt.unsqueeze(2)                                     # (B, D, 1, T)
        else:
            z_q = z_q_bt
        # Stats
        counts = one_hot.sum(dim=0)                                        # (K,)
        probs = counts / max(float(B * T), 1.0)
        probs = torch.clamp(probs, min=self.eps)
        entropy = -torch.sum(probs * torch.log(probs))
        perplexity = torch.exp(entropy)
        usage = (counts > 0).float().mean()
        idx_bt = indices.view(B, T).unsqueeze(1)
        info = {"indices": idx_bt, "perplexity": float(perplexity.item()), "usage": float(usage.item())}
        loss_dict = {"vq": vq_loss, "commitment": commitment, "codebook": codebook}
        return z_q, loss_dict, info
