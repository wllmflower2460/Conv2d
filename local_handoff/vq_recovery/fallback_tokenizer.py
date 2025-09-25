# fallback_tokenizer.py
# Tiny online k-means tokenizer to avoid halting when VQ collapses.
import torch
import torch.nn as nn

class KMeansTokenizer(nn.Module):
    def __init__(self, num_codes: int = 128, code_dim: int = 64, iters: int = 10, seed: int = 42):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.iters = iters
        self.seed = seed
        self.register_buffer("centroids", torch.empty(0))

    @torch.no_grad()
    def fit(self, z_e):
        # z_e: (B, D, 1, T) or (B, D, T)
        if z_e.dim() == 4:
            B, D, _, T = z_e.shape
            z = z_e.squeeze(2).permute(0, 2, 1).contiguous().view(B*T, D)
        else:
            B, D, T = z_e.shape
            z = z_e.permute(0, 2, 1).contiguous().view(B*T, D)
        torch.manual_seed(self.seed)
        idx = torch.randperm(z.shape[0], device=z.device)[:self.num_codes]
        C = z[idx].clone()
        for _ in range(self.iters):
            d = (z**2).sum(1, keepdim=True) - 2*z@C.t() + (C**2).sum(1).unsqueeze(0)
            a = torch.argmin(d, dim=1)
            for k in range(self.num_codes):
                m = (a == k)
                if m.any():
                    C[k] = z[m].mean(0)
        self.centroids = C

    @torch.no_grad()
    def tokenize(self, z_e):
        if self.centroids.numel() == 0:
            raise RuntimeError("Call fit() before tokenize().")
        if z_e.dim() == 4:
            B, D, _, T = z_e.shape
            z = z_e.squeeze(2).permute(0, 2, 1).contiguous().view(B*T, D)
        else:
            B, D, T = z_e.shape
            z = z_e.permute(0, 2, 1).contiguous().view(B*T, D)
        d = (z**2).sum(1, keepdim=True) - 2*z@self.centroids.t() + (self.centroids**2).sum(1).unsqueeze(0)
        idx = torch.argmin(d, dim=1)  # (B*T,)
        idx_bt = idx.view(B, T).unsqueeze(1)
        counts = torch.bincount(idx, minlength=self.num_codes).float()
        p = counts / max(float(idx.numel()), 1.0)
        p = torch.clamp(p, min=1e-12)
        H = -torch.sum(p * torch.log(p)); perp = torch.exp(H)
        usage = (counts > 0).float().mean()
        return idx_bt, float(perp.item()), float(usage.item())
