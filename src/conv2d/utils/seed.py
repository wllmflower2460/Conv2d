"""Seed utilities for reproducibility."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None) -> int:
    """Set random seed for reproducibility.
    
    Sets the seed for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed to use. If None, generates a random seed.
        
    Returns:
        The seed that was set
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return seed