"""Metrics, calibration, and reporting for model evaluation.

Provides comprehensive evaluation tools including standard metrics,
calibration analysis, and reproducible evaluation bundles.
"""

from __future__ import annotations

from conv2d.metrics.bundle import BundleGenerator, EvaluationBundle
from conv2d.metrics.calibration import CalibrationAnalyzer, CalibrationResult
from conv2d.metrics.core import MetricsCalculator, MetricsResult, QATracker

__all__ = [
    "MetricsCalculator",
    "MetricsResult",
    "QATracker",
    "CalibrationAnalyzer",
    "CalibrationResult",
    "BundleGenerator",
    "EvaluationBundle",
]