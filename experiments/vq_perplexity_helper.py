# vq_perplexity_helper.py
# Drop into your repo (e.g., models/ or utils/) and call from VectorQuantizerEMA2D.forward()

import torch
import numpy as np

def vq_perplexity_from_indices(code_indices: torch.Tensor, K: int):
    """
    Compute codebook perplexity and usage given code indices.
    Args:
      code_indices: LongTensor of shape (...,) with values in [0, K-1]
      K: number of codes
    Returns: (perplexity, usage_fraction)
    """
    flat = code_indices.view(-1)
    counts = torch.bincount(flat, minlength=K).float()
    probs = counts / max(float(flat.numel()), 1.0)
    probs = torch.clamp(probs, min=1e-12)
    entropy = -torch.sum(probs * torch.log(probs))  # nats
    perplexity = float(torch.exp(entropy).item())
    usage = float((counts > 0).sum().item()) / float(K)
    return perplexity, usage


def vq_perplexity_from_onehot(one_hot_assign: torch.Tensor):
    """
    one_hot_assign: FloatTensor (B*T, K) or (N, K) with rows summing to 1
    Returns: (perplexity, usage_fraction)
    """
    probs = one_hot_assign.mean(dim=0)
    probs = torch.clamp(probs, min=1e-12)
    entropy = -torch.sum(probs * torch.log(probs))
    perplexity = float(torch.exp(entropy).item())
    usage = float((probs > 0).sum().item()) / float(probs.numel())
    return perplexity, usage