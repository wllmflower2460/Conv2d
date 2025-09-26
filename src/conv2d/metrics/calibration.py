"""Calibration analysis and reliability diagrams.

Provides tools for assessing and visualizing model calibration
including reliability diagrams and calibration plots.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Results from calibration analysis.
    
    Attributes:
        bin_edges: Confidence bin boundaries
        bin_accuracy: Actual accuracy per bin
        bin_confidence: Mean confidence per bin
        bin_counts: Number of samples per bin
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        reliability_data: Data for reliability diagram
    """
    bin_edges: NDArray[np.float32]
    bin_accuracy: NDArray[np.float32]
    bin_confidence: NDArray[np.float32]
    bin_counts: NDArray[np.int32]
    ece: float
    mce: float
    reliability_data: Dict[str, List[float]]
    
    def to_json(self, path: Optional[Path] = None) -> str:
        """Export calibration data to JSON."""
        data = {
            "bin_edges": self.bin_edges.tolist(),
            "bin_accuracy": self.bin_accuracy.tolist(),
            "bin_confidence": self.bin_confidence.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "ece": float(self.ece),
            "mce": float(self.mce),
            "reliability_data": self.reliability_data,
        }
        
        json_str = json.dumps(data, indent=2)
        
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str)
            
        return json_str


class CalibrationAnalyzer:
    """Analyze and visualize model calibration."""
    
    def __init__(self, n_bins: int = 10):
        """Initialize calibration analyzer.
        
        Args:
            n_bins: Number of confidence bins
        """
        self.n_bins = n_bins
    
    def analyze(
        self,
        y_true: NDArray[np.int32],
        y_prob: NDArray[np.float32],
        y_pred: Optional[NDArray[np.int32]] = None,
    ) -> CalibrationResult:
        """Analyze calibration of predictions.
        
        Args:
            y_true: True labels (N,)
            y_prob: Predicted probabilities (N, K) or (N,)
            y_pred: Optional predicted labels
            
        Returns:
            CalibrationResult with analysis
        """
        # Get predicted labels if not provided
        if y_pred is None:
            if y_prob.ndim == 2:
                y_pred = np.argmax(y_prob, axis=1)
            else:
                y_pred = (y_prob > 0.5).astype(np.int32)
        
        # Get confidence scores
        if y_prob.ndim == 2:
            confidence = np.max(y_prob, axis=1)
        else:
            confidence = y_prob
        
        # Create bins
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_accuracy = np.zeros(self.n_bins, dtype=np.float32)
        bin_confidence = np.zeros(self.n_bins, dtype=np.float32)
        bin_counts = np.zeros(self.n_bins, dtype=np.int32)
        
        # Compute statistics per bin
        for i in range(self.n_bins):
            bin_mask = (confidence > bin_edges[i]) & (confidence <= bin_edges[i + 1])
            bin_counts[i] = np.sum(bin_mask)
            
            if bin_counts[i] > 0:
                bin_accuracy[i] = np.mean(y_true[bin_mask] == y_pred[bin_mask])
                bin_confidence[i] = np.mean(confidence[bin_mask])
            else:
                # Empty bin - use bin center
                bin_accuracy[i] = 0.0
                bin_confidence[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        
        # Compute ECE and MCE
        ece = 0.0
        mce = 0.0
        total_samples = np.sum(bin_counts)
        
        for i in range(self.n_bins):
            if bin_counts[i] > 0:
                bin_weight = bin_counts[i] / total_samples
                calibration_error = np.abs(bin_confidence[i] - bin_accuracy[i])
                ece += bin_weight * calibration_error
                mce = max(mce, calibration_error)
        
        # Prepare reliability diagram data
        reliability_data = {
            "confidence": bin_confidence.tolist(),
            "accuracy": bin_accuracy.tolist(),
            "counts": bin_counts.tolist(),
            "gap": (bin_confidence - bin_accuracy).tolist(),
        }
        
        return CalibrationResult(
            bin_edges=bin_edges.astype(np.float32),
            bin_accuracy=bin_accuracy,
            bin_confidence=bin_confidence,
            bin_counts=bin_counts,
            ece=float(ece),
            mce=float(mce),
            reliability_data=reliability_data,
        )
    
    def plot_reliability_diagram(
        self,
        calibration_result: CalibrationResult,
        save_path: Optional[Path] = None,
        title: str = "Reliability Diagram",
        show_gap: bool = True,
        show_histogram: bool = True,
    ) -> None:
        """Plot reliability diagram.
        
        Args:
            calibration_result: Calibration analysis results
            save_path: Optional path to save plot
            title: Plot title
            show_gap: Whether to show calibration gap
            show_histogram: Whether to show confidence histogram
        """
        fig = plt.figure(figsize=(8, 6))
        
        if show_histogram:
            # Create subplot with histogram
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
        else:
            ax1 = plt.gca()
            ax2 = None
        
        # Plot perfect calibration line
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
        
        # Plot actual calibration
        bin_centers = calibration_result.bin_confidence
        bin_accuracy = calibration_result.bin_accuracy
        
        ax1.plot(bin_centers, bin_accuracy, 'o-', color='blue', 
                label=f'Model (ECE={calibration_result.ece:.3f})', 
                markersize=8, linewidth=2)
        
        # Show calibration gap
        if show_gap:
            for i in range(len(bin_centers)):
                if calibration_result.bin_counts[i] > 0:
                    ax1.plot([bin_centers[i], bin_centers[i]], 
                            [bin_centers[i], bin_accuracy[i]], 
                            'r-', alpha=0.3, linewidth=1)
        
        ax1.set_xlabel('Confidence' if not show_histogram else '')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1.set_title(title)
        
        # Add text with metrics
        text_str = f'ECE: {calibration_result.ece:.3f}\nMCE: {calibration_result.mce:.3f}'
        ax1.text(0.95, 0.05, text_str, transform=ax1.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot histogram of confidences
        if show_histogram and ax2 is not None:
            bin_edges = calibration_result.bin_edges
            bin_counts = calibration_result.bin_counts
            
            # Plot histogram
            ax2.bar(bin_centers, bin_counts, width=1.0/self.n_bins, 
                   alpha=0.5, color='gray', edgecolor='black')
            
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Count')
            ax2.set_xlim([0, 1])
            ax2.grid(True, alpha=0.3)
            
            # Hide x-axis labels on main plot
            plt.setp(ax1.get_xticklabels(), visible=False)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved reliability diagram to {save_path}")
        
        plt.close()
    
    def plot_confidence_histogram(
        self,
        y_prob: NDArray[np.float32],
        y_true: Optional[NDArray[np.int32]] = None,
        y_pred: Optional[NDArray[np.int32]] = None,
        save_path: Optional[Path] = None,
        title: str = "Confidence Distribution",
    ) -> None:
        """Plot histogram of prediction confidences.
        
        Args:
            y_prob: Predicted probabilities
            y_true: Optional true labels for correct/incorrect split
            y_pred: Optional predicted labels
            save_path: Optional path to save plot
            title: Plot title
        """
        # Get confidence scores
        if y_prob.ndim == 2:
            confidence = np.max(y_prob, axis=1)
            if y_pred is None:
                y_pred = np.argmax(y_prob, axis=1)
        else:
            confidence = y_prob
            if y_pred is None:
                y_pred = (y_prob > 0.5).astype(np.int32)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if y_true is not None and y_pred is not None:
            # Split by correct/incorrect
            correct_mask = y_true == y_pred
            
            ax.hist(confidence[correct_mask], bins=self.n_bins, 
                   alpha=0.5, label='Correct', color='green', 
                   range=(0, 1), density=True)
            ax.hist(confidence[~correct_mask], bins=self.n_bins, 
                   alpha=0.5, label='Incorrect', color='red', 
                   range=(0, 1), density=True)
            ax.legend()
        else:
            # Single histogram
            ax.hist(confidence, bins=self.n_bins, alpha=0.7, 
                   color='blue', range=(0, 1), density=True)
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Density')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        
        # Add mean confidence
        mean_conf = np.mean(confidence)
        ax.axvline(mean_conf, color='black', linestyle='--', 
                  alpha=0.5, label=f'Mean: {mean_conf:.3f}')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved confidence histogram to {save_path}")
        
        plt.close()
    
    def plot_calibration_comparison(
        self,
        calibration_results: Dict[str, CalibrationResult],
        save_path: Optional[Path] = None,
        title: str = "Calibration Comparison",
    ) -> None:
        """Plot comparison of multiple models' calibration.
        
        Args:
            calibration_results: Dict mapping model names to results
            save_path: Optional path to save plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect')
        
        # Plot each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(calibration_results)))
        
        for (name, result), color in zip(calibration_results.items(), colors):
            ax.plot(result.bin_confidence, result.bin_accuracy, 
                   'o-', color=color, 
                   label=f'{name} (ECE={result.ece:.3f})',
                   markersize=6, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved calibration comparison to {save_path}")
        
        plt.close()