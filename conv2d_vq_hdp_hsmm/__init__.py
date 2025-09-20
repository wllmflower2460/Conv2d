"""
Conv2d-VQ-HDP-HSMM: A PyTorch implementation combining Vector Quantization,
Hierarchical Dirichlet Process clustering, and Hidden Semi-Markov Models
for behavioral analysis on edge devices.
"""

from .vector_quantization import VectorQuantization
from .hdp_clustering import HDPClustering
from .hsmm import HSMM
from .model import Conv2dVQHDPHSMM

__version__ = "0.1.0"
__all__ = ["VectorQuantization", "HDPClustering", "HSMM", "Conv2dVQHDPHSMM"]