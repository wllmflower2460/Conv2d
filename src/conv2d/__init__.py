"""Conv2d-FSQ-HSMM: Behavioral synchrony analysis framework.

A comprehensive framework for behavioral synchrony analysis using:
- Finite Scalar Quantization (FSQ) for discrete representation
- Hidden Semi-Markov Models (HSMM) for temporal dynamics
- Entropy-based uncertainty quantification
- Edge device optimization (Hailo-8, CoreML)
"""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "Conv2d Team"
__email__ = "flower.mobile@gmail.com"

# Public API exports
from conv2d.models import (
    Conv2dFSQHSMM,
    Conv2dFSQModel,
    FSQLayer,
    HSMMComponents,
)
from conv2d.preprocessing import (
    DataQualityHandler,
    KinematicFeatureExtractor,
    VectorizedOperations,
)
from conv2d.training import Trainer, TrainingConfig
from conv2d.utils import set_seed, setup_logger

__all__ = [
    # Version info
    "__version__",
    # Models
    "Conv2dFSQHSMM",
    "Conv2dFSQModel",
    "FSQLayer",
    "HSMMComponents",
    # Preprocessing
    "DataQualityHandler",
    "KinematicFeatureExtractor",
    "VectorizedOperations",
    # Training
    "Trainer",
    "TrainingConfig",
    # Utils
    "set_seed",
    "setup_logger",
]