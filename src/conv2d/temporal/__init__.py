"""Temporal smoothing policies for motif sequences.

Provides swappable policies to prevent flickering and enforce
realistic behavioral durations without modifying upstream components.
"""

from __future__ import annotations

from conv2d.temporal.hsmm import HSMMPolicy
from conv2d.temporal.interface import TemporalConfig, TemporalPolicy
from conv2d.temporal.median import MedianHysteresisPolicy
from conv2d.temporal.passthrough import PassthroughPolicy

__all__ = [
    "TemporalPolicy",
    "TemporalConfig",
    "MedianHysteresisPolicy",
    "HSMMPolicy",
    "PassthroughPolicy",
]