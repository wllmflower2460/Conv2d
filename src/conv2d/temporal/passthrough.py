"""Passthrough policy that applies no temporal smoothing.

Used as baseline and when temporal smoothing is disabled.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from conv2d.temporal.interface import TemporalConfig, TemporalPolicy

logger = logging.getLogger(__name__)


class PassthroughPolicy(TemporalPolicy):
    """No-op temporal policy that returns input unchanged.
    
    Useful for:
    - Baseline comparisons
    - Disabling smoothing at runtime
    - Testing upstream components
    """
    
    def __init__(self, config: TemporalConfig):
        """Initialize passthrough policy.
        
        Args:
            config: Temporal configuration (ignored)
        """
        super().__init__(config)
        logger.info("PassthroughPolicy: temporal smoothing disabled")
    
    def smooth_sequence(
        self, 
        motifs: NDArray[np.int32],
        confidences: Optional[NDArray[np.float32]] = None,
    ) -> NDArray[np.int32]:
        """Return motifs unchanged.
        
        Args:
            motifs: Raw motif predictions
            confidences: Ignored
            
        Returns:
            Original motifs without modification
        """
        return motifs.copy()