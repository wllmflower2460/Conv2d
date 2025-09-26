"""Temporal smoothing policies for motif sequences.

This module provides swappable temporal policies to prevent
flickering and enforce realistic behavioral durations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class TemporalConfig:
    """Configuration for temporal smoothing.
    
    Attributes:
        min_dwell: Minimum frames to stay in a motif
        enter_threshold: Confidence/count to enter new motif
        exit_threshold: Confidence/count to exit current motif
        window_size: Sliding window for median/voting
        enable_smoothing: Global on/off switch
        policy_type: Which policy to use ('median', 'hsmm', 'none')
        extra_params: Policy-specific parameters
    """
    min_dwell: int = 5
    enter_threshold: float = 0.6
    exit_threshold: float = 0.4
    window_size: int = 5
    enable_smoothing: bool = True
    policy_type: str = "median"
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.min_dwell >= 1, f"min_dwell must be >= 1, got {self.min_dwell}"
        assert 0 <= self.enter_threshold <= 1, f"enter_threshold must be in [0,1], got {self.enter_threshold}"
        assert 0 <= self.exit_threshold <= 1, f"exit_threshold must be in [0,1], got {self.exit_threshold}"
        assert self.exit_threshold <= self.enter_threshold, "exit_threshold must be <= enter_threshold"
        assert self.window_size >= 1, f"window_size must be >= 1, got {self.window_size}"
        assert self.policy_type in ["median", "hsmm", "none"], f"Unknown policy: {self.policy_type}"
        
        if self.extra_params is None:
            self.extra_params = {}
        
        logger.info(
            f"TemporalConfig: policy={self.policy_type}, min_dwell={self.min_dwell}, "
            f"enter={self.enter_threshold:.2f}, exit={self.exit_threshold:.2f}, "
            f"window={self.window_size}, enabled={self.enable_smoothing}"
        )


class TemporalPolicy(ABC):
    """Abstract base class for temporal smoothing policies.
    
    All policies take raw motif predictions and produce
    temporally coherent motif sequences respecting minimum
    dwell times and transition hysteresis.
    """
    
    def __init__(self, config: TemporalConfig):
        """Initialize temporal policy.
        
        Args:
            config: Temporal smoothing configuration
        """
        self.config = config
        self.name = self.__class__.__name__
        
        # State tracking
        self.current_motif_: Optional[int] = None
        self.dwell_counter_: int = 0
        self.confidence_buffer_: Optional[NDArray[np.float32]] = None
        
        logger.info(f"Initialized {self.name} with config: {config}")
    
    @abstractmethod
    def smooth_sequence(
        self, 
        motifs: NDArray[np.int32],
        confidences: Optional[NDArray[np.float32]] = None,
    ) -> NDArray[np.int32]:
        """Apply temporal smoothing to motif sequence.
        
        Args:
            motifs: Raw motif predictions (B, T) or (T,)
            confidences: Optional confidence scores (B, T) or (T,)
            
        Returns:
            Smoothed motif sequence with same shape
        """
        pass
    
    def reset(self) -> None:
        """Reset internal state for new sequence."""
        self.current_motif_ = None
        self.dwell_counter_ = 0
        self.confidence_buffer_ = None
        logger.debug(f"{self.name}: State reset")
    
    def _enforce_min_dwell(
        self, 
        motifs: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        """Enforce minimum dwell time constraint.
        
        Args:
            motifs: Motif sequence (T,)
            
        Returns:
            Motifs with minimum dwell enforced
        """
        if self.config.min_dwell <= 1:
            return motifs
        
        smoothed = motifs.copy()
        T = len(motifs)
        
        i = 0
        while i < T:
            # Find run length
            current = motifs[i]
            run_end = i + 1
            while run_end < T and motifs[run_end] == current:
                run_end += 1
            run_length = run_end - i
            
            # If run too short, try to extend or merge
            if run_length < self.config.min_dwell:
                # Look at neighbors
                prev_motif = motifs[i-1] if i > 0 else -1
                next_motif = motifs[run_end] if run_end < T else -1
                
                # Merge with larger neighbor
                if i > 0 and run_end < T:
                    # Check which neighbor is more prevalent
                    prev_count = np.sum(motifs[max(0, i-10):i] == prev_motif)
                    next_count = np.sum(motifs[run_end:min(T, run_end+10)] == next_motif)
                    
                    if prev_count >= next_count:
                        smoothed[i:run_end] = prev_motif
                    else:
                        smoothed[i:run_end] = next_motif
                elif i > 0:
                    # Only previous neighbor exists
                    smoothed[i:run_end] = prev_motif
                elif run_end < T:
                    # Only next neighbor exists
                    smoothed[i:run_end] = next_motif
                # else: single short segment, leave as is
            
            i = run_end
        
        return smoothed
    
    def _apply_hysteresis(
        self,
        motifs: NDArray[np.int32],
        confidences: NDArray[np.float32],
    ) -> NDArray[np.int32]:
        """Apply hysteresis thresholds for transitions.
        
        Args:
            motifs: Raw motif predictions (T,)
            confidences: Confidence scores (T, K) where K is num motifs
            
        Returns:
            Motifs with hysteresis applied
        """
        T = len(motifs)
        smoothed = np.zeros(T, dtype=np.int32)
        
        for t in range(T):
            if self.current_motif_ is None:
                # First frame, just take the prediction
                self.current_motif_ = motifs[t]
                self.dwell_counter_ = 1
            else:
                proposed = motifs[t]
                
                if proposed == self.current_motif_:
                    # Staying in same motif
                    self.dwell_counter_ += 1
                else:
                    # Check if we should transition
                    if self.dwell_counter_ < self.config.min_dwell:
                        # Haven't met minimum dwell, stay
                        proposed = self.current_motif_
                    else:
                        # Check confidence thresholds
                        if confidences is not None and confidences.ndim == 2:
                            current_conf = confidences[t, self.current_motif_]
                            proposed_conf = confidences[t, proposed]
                            
                            # Need high confidence to enter new motif
                            if proposed_conf < self.config.enter_threshold:
                                proposed = self.current_motif_
                            # Need low confidence to exit current motif
                            elif current_conf > self.config.exit_threshold:
                                proposed = self.current_motif_
                        
                        # Update state if transitioning
                        if proposed != self.current_motif_:
                            self.current_motif_ = proposed
                            self.dwell_counter_ = 1
                        else:
                            self.dwell_counter_ += 1
            
            smoothed[t] = self.current_motif_
        
        return smoothed
    
    @staticmethod
    def create(config: Union[TemporalConfig, Dict[str, Any], str]) -> TemporalPolicy:
        """Factory method to create appropriate policy.
        
        Args:
            config: TemporalConfig, dict, or policy name string
            
        Returns:
            Configured temporal policy instance
        """
        # Handle different input types
        if isinstance(config, str):
            config = TemporalConfig(policy_type=config)
        elif isinstance(config, dict):
            config = TemporalConfig(**config)
        elif not isinstance(config, TemporalConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        
        # Create appropriate policy
        if config.policy_type == "median":
            from conv2d.temporal.median import MedianHysteresisPolicy
            return MedianHysteresisPolicy(config)
        elif config.policy_type == "hsmm":
            from conv2d.temporal.hsmm import HSMMPolicy
            return HSMMPolicy(config)
        elif config.policy_type == "none":
            from conv2d.temporal.passthrough import PassthroughPolicy
            return PassthroughPolicy(config)
        else:
            raise ValueError(f"Unknown policy type: {config.policy_type}")