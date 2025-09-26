"""Core metrics for behavioral analysis evaluation.

Provides standard metrics for model evaluation including
accuracy, F1, calibration, and behavioral-specific metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for evaluation metrics.
    
    Attributes:
        accuracy: Overall accuracy
        macro_f1: Macro-averaged F1 score
        per_class_f1: F1 score per class
        precision: Per-class precision
        recall: Per-class recall
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        coverage: Fraction of classes predicted
        motif_count: Number of unique motifs
        code_usage_percent: Percentage of codes used
        perplexity: Code usage perplexity
        confusion_matrix: Confusion matrix
        classification_report: Detailed classification report
        qa_stats: Quality assurance statistics
        metadata: Additional metadata
    """
    accuracy: float
    macro_f1: float
    per_class_f1: NDArray[np.float32]
    precision: NDArray[np.float32]
    recall: NDArray[np.float32]
    ece: float
    mce: float
    coverage: float
    motif_count: int
    code_usage_percent: float
    perplexity: float
    confusion_matrix: NDArray[np.int32]
    classification_report: Dict[str, Any]
    qa_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self, path: Optional[Path] = None) -> str:
        """Export metrics to JSON.
        
        Args:
            path: Optional path to save JSON
            
        Returns:
            JSON string representation
        """
        # Convert numpy arrays to lists
        data = {
            "accuracy": float(self.accuracy),
            "macro_f1": float(self.macro_f1),
            "per_class_f1": self.per_class_f1.tolist(),
            "precision": self.precision.tolist(),
            "recall": self.recall.tolist(),
            "ece": float(self.ece),
            "mce": float(self.mce),
            "coverage": float(self.coverage),
            "motif_count": int(self.motif_count),
            "code_usage_percent": float(self.code_usage_percent),
            "perplexity": float(self.perplexity),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
            "qa_stats": self.qa_stats,
            "metadata": self.metadata,
        }
        
        json_str = json.dumps(data, indent=2)
        
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str)
            logger.info(f"Saved metrics to {path}")
        
        return json_str
    
    def summary(self) -> str:
        """Get summary string of key metrics."""
        return (
            f"Accuracy: {self.accuracy:.3f} | "
            f"Macro-F1: {self.macro_f1:.3f} | "
            f"ECE: {self.ece:.3f} | "
            f"Coverage: {self.coverage:.2f} | "
            f"Motifs: {self.motif_count} | "
            f"Perplexity: {self.perplexity:.2f}"
        )


class MetricsCalculator:
    """Calculate standard and custom metrics."""
    
    def __init__(self, n_bins: int = 10):
        """Initialize metrics calculator.
        
        Args:
            n_bins: Number of bins for calibration metrics
        """
        self.n_bins = n_bins
    
    def compute_all(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        y_prob: Optional[NDArray[np.float32]] = None,
        codes: Optional[NDArray[np.int32]] = None,
        qa_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricsResult:
        """Compute all metrics.
        
        Args:
            y_true: True labels (N,)
            y_pred: Predicted labels (N,)
            y_prob: Optional predicted probabilities (N, K)
            codes: Optional discrete codes for perplexity
            qa_data: Optional QA statistics
            metadata: Optional metadata
            
        Returns:
            MetricsResult with all metrics
        """
        # Flatten if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Calibration metrics
        if y_prob is not None:
            ece, mce = self.compute_calibration_error(y_true, y_pred, y_prob)
        else:
            ece, mce = 0.0, 0.0
        
        # Coverage (fraction of classes predicted)
        n_classes = len(np.unique(y_true))
        n_predicted = len(np.unique(y_pred))
        coverage = n_predicted / n_classes if n_classes > 0 else 0.0
        
        # Motif count
        motif_count = n_predicted
        
        # Code usage and perplexity
        if codes is not None:
            code_usage_percent, perplexity = self.compute_code_metrics(codes)
        else:
            code_usage_percent, perplexity = 0.0, 0.0
        
        # QA stats
        if qa_data is None:
            qa_data = {}
        
        return MetricsResult(
            accuracy=accuracy,
            macro_f1=macro_f1,
            per_class_f1=f1.astype(np.float32),
            precision=precision.astype(np.float32),
            recall=recall.astype(np.float32),
            ece=ece,
            mce=mce,
            coverage=coverage,
            motif_count=motif_count,
            code_usage_percent=code_usage_percent,
            perplexity=perplexity,
            confusion_matrix=cm.astype(np.int32),
            classification_report=report,
            qa_stats=qa_data,
            metadata=metadata or {},
        )
    
    def compute_calibration_error(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        y_prob: NDArray[np.float32],
    ) -> Tuple[float, float]:
        """Compute Expected and Maximum Calibration Error.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (N, K)
            
        Returns:
            Tuple of (ECE, MCE)
        """
        # Get confidence (max probability)
        if y_prob.ndim == 2:
            confidence = np.max(y_prob, axis=1)
        else:
            confidence = y_prob
        
        # Bin predictions by confidence
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
                # Average confidence in this bin
                avg_confidence_in_bin = confidence[in_bin].mean()
                
                # Calibration error for this bin
                calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                
                # Weighted by proportion of samples
                ece += prop_in_bin * calibration_error
                
                # Track maximum
                mce = max(mce, calibration_error)
        
        return float(ece), float(mce)
    
    def compute_code_metrics(
        self,
        codes: NDArray[np.int32],
        codebook_size: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute code usage and perplexity.
        
        Args:
            codes: Discrete codes
            codebook_size: Total number of possible codes
            
        Returns:
            Tuple of (usage_percent, perplexity)
        """
        # Flatten codes
        codes_flat = codes.flatten()
        
        # Count unique codes
        unique_codes = np.unique(codes_flat)
        n_used = len(unique_codes)
        
        # Estimate codebook size if not provided
        if codebook_size is None:
            codebook_size = int(codes_flat.max() + 1)
        
        # Usage percentage
        usage_percent = (n_used / codebook_size) * 100 if codebook_size > 0 else 0.0
        
        # Compute perplexity
        counts = np.bincount(codes_flat, minlength=codebook_size)
        probs = counts / counts.sum() if counts.sum() > 0 else np.ones(codebook_size) / codebook_size
        
        # Filter out zeros for perplexity
        probs_nonzero = probs[probs > 0]
        
        if len(probs_nonzero) > 0:
            entropy = -np.sum(probs_nonzero * np.log(probs_nonzero))
            perplexity = np.exp(entropy)
        else:
            perplexity = 1.0
        
        return float(usage_percent), float(perplexity)
    
    def compute_behavioral_metrics(
        self,
        motif_sequence: NDArray[np.int32],
        timestamps: Optional[NDArray[np.float32]] = None,
    ) -> Dict[str, float]:
        """Compute behavioral-specific metrics.
        
        Args:
            motif_sequence: Sequence of motif IDs (T,)
            timestamps: Optional timestamps for each frame
            
        Returns:
            Dictionary of behavioral metrics
        """
        metrics = {}
        
        # Transition rate
        transitions = np.sum(np.diff(motif_sequence) != 0)
        metrics["transition_rate"] = transitions / len(motif_sequence)
        
        # Dwell time statistics
        dwell_times = self._compute_dwell_times(motif_sequence)
        if len(dwell_times) > 0:
            metrics["mean_dwell"] = float(np.mean(dwell_times))
            metrics["median_dwell"] = float(np.median(dwell_times))
            metrics["min_dwell"] = float(np.min(dwell_times))
            metrics["max_dwell"] = float(np.max(dwell_times))
        else:
            metrics["mean_dwell"] = 0.0
            metrics["median_dwell"] = 0.0
            metrics["min_dwell"] = 0.0
            metrics["max_dwell"] = 0.0
        
        # Motif distribution entropy
        motif_counts = np.bincount(motif_sequence)
        motif_probs = motif_counts / motif_counts.sum()
        motif_probs_nonzero = motif_probs[motif_probs > 0]
        
        if len(motif_probs_nonzero) > 0:
            motif_entropy = -np.sum(motif_probs_nonzero * np.log(motif_probs_nonzero))
            metrics["motif_entropy"] = float(motif_entropy)
        else:
            metrics["motif_entropy"] = 0.0
        
        return metrics
    
    def _compute_dwell_times(self, sequence: NDArray[np.int32]) -> NDArray[np.int32]:
        """Compute dwell times for each segment."""
        if len(sequence) == 0:
            return np.array([], dtype=np.int32)
        
        # Find transition points
        transitions = np.where(np.diff(sequence) != 0)[0] + 1
        transitions = np.concatenate([[0], transitions, [len(sequence)]])
        
        # Compute segment lengths
        dwell_times = np.diff(transitions)
        
        return dwell_times


class QATracker:
    """Track quality assurance metrics during evaluation."""
    
    def __init__(self):
        """Initialize QA tracker."""
        self.nan_events = 0
        self.inf_events = 0
        self.mean_fallbacks = 0
        self.zero_variance_features = 0
        self.outlier_samples = 0
        self.failed_windows = 0
        self.total_samples = 0
        
    def check_data(self, X: NDArray) -> Dict[str, Any]:
        """Check data quality and track issues.
        
        Args:
            X: Input data array
            
        Returns:
            Dictionary of QA statistics
        """
        self.total_samples += X.shape[0] if X.ndim > 1 else 1
        
        # Check for NaN
        nan_mask = np.isnan(X)
        self.nan_events += np.sum(nan_mask)
        
        # Check for Inf
        inf_mask = np.isinf(X)
        self.inf_events += np.sum(inf_mask)
        
        # Check for zero variance features
        if X.ndim >= 2:
            variances = np.var(X, axis=0)
            self.zero_variance_features += np.sum(variances < 1e-10)
        
        # Check for outliers (> 5 std)
        if not np.any(nan_mask) and not np.any(inf_mask):
            mean = np.mean(X)
            std = np.std(X)
            if std > 0:
                z_scores = np.abs((X - mean) / std)
                self.outlier_samples += np.sum(z_scores > 5)
        
        return self.get_stats()
    
    def record_fallback(self, reason: str = "unknown") -> None:
        """Record a fallback event."""
        self.mean_fallbacks += 1
        logger.warning(f"Fallback recorded: {reason}")
    
    def record_failed_window(self, reason: str = "unknown") -> None:
        """Record a failed window."""
        self.failed_windows += 1
        logger.warning(f"Failed window: {reason}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get QA statistics summary."""
        return {
            "nan_events": self.nan_events,
            "inf_events": self.inf_events,
            "mean_fallbacks": self.mean_fallbacks,
            "zero_variance_features": self.zero_variance_features,
            "outlier_samples": self.outlier_samples,
            "failed_windows": self.failed_windows,
            "total_samples": self.total_samples,
            "nan_rate": self.nan_events / max(self.total_samples, 1),
            "outlier_rate": self.outlier_samples / max(self.total_samples, 1),
        }
    
    def reset(self) -> None:
        """Reset all counters."""
        self.nan_events = 0
        self.inf_events = 0
        self.mean_fallbacks = 0
        self.zero_variance_features = 0
        self.outlier_samples = 0
        self.failed_windows = 0
        self.total_samples = 0