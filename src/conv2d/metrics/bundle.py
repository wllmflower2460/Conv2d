"""Evaluation bundle generator for comprehensive model assessment.

Creates standardized evaluation packages with metrics, plots,
and quality assurance tracking for reproducible review.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray

from conv2d.metrics.calibration import CalibrationAnalyzer
from conv2d.metrics.core import MetricsCalculator, QATracker

logger = logging.getLogger(__name__)


@dataclass
class EvaluationBundle:
    """Complete evaluation package for a model/experiment.
    
    Attributes:
        exp_hash: Unique experiment hash
        timestamp: Evaluation timestamp
        config: Experiment configuration
        metrics: Core metrics results
        calibration: Calibration analysis results
        plots: Dictionary of plot paths
        qa_stats: Quality assurance statistics
        output_dir: Bundle output directory
    """
    exp_hash: str
    timestamp: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    calibration: Dict[str, Any]
    plots: Dict[str, Path]
    qa_stats: Dict[str, Any]
    output_dir: Path
    
    def summary(self) -> str:
        """Get summary of key results."""
        return (
            f"Evaluation Bundle: {self.exp_hash[:8]}\n"
            f"Timestamp: {self.timestamp}\n"
            f"Accuracy: {self.metrics.get('accuracy', 0):.3f}\n"
            f"Macro-F1: {self.metrics.get('macro_f1', 0):.3f}\n"
            f"ECE: {self.calibration.get('ece', 0):.3f}\n"
            f"Output: {self.output_dir}"
        )


class BundleGenerator:
    """Generate comprehensive evaluation bundles."""
    
    def __init__(
        self,
        output_base: Path = Path("reports"),
        save_raw_predictions: bool = False,
    ):
        """Initialize bundle generator.
        
        Args:
            output_base: Base directory for evaluation bundles
            save_raw_predictions: Whether to save raw prediction arrays
        """
        self.output_base = Path(output_base)
        self.save_raw_predictions = save_raw_predictions
        self.metrics_calc = MetricsCalculator()
        self.calibration_analyzer = CalibrationAnalyzer()
        self.qa_tracker = QATracker()
    
    def generate(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        y_prob: Optional[NDArray[np.float32]] = None,
        codes: Optional[NDArray[np.int32]] = None,
        config: Optional[Dict[str, Any]] = None,
        exp_name: Optional[str] = None,
        X_data: Optional[NDArray] = None,
    ) -> EvaluationBundle:
        """Generate complete evaluation bundle.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Optional predicted probabilities
            codes: Optional discrete codes for perplexity
            config: Experiment configuration
            exp_name: Optional experiment name
            X_data: Optional input data for QA checks
            
        Returns:
            EvaluationBundle with all results
        """
        # Generate experiment hash
        exp_hash = self._generate_exp_hash(config, exp_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = self.output_base / f"{exp_hash}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating evaluation bundle: {output_dir}")
        
        # QA checks on input data
        qa_stats = {}
        if X_data is not None:
            qa_stats = self.qa_tracker.check_data(X_data)
        
        # Compute metrics
        metrics_result = self.metrics_calc.compute_all(
            y_true, y_pred, y_prob, codes, qa_stats
        )
        
        # Save metrics JSON
        metrics_path = output_dir / "metrics.json"
        metrics_result.to_json(metrics_path)
        
        # Calibration analysis
        calibration_result = None
        if y_prob is not None:
            calibration_result = self.calibration_analyzer.analyze(
                y_true, y_prob, y_pred
            )
            
            # Save calibration JSON
            calib_path = output_dir / "calibration.json"
            calibration_result.to_json(calib_path)
        
        # Generate plots
        plots = {}
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Confusion matrix
        cm_path = plots_dir / "confusion_matrix.png"
        self._plot_confusion_matrix(
            metrics_result.confusion_matrix,
            save_path=cm_path
        )
        plots["confusion_matrix"] = cm_path
        
        # Reliability diagram
        if calibration_result:
            rel_path = plots_dir / "reliability_diagram.png"
            self.calibration_analyzer.plot_reliability_diagram(
                calibration_result,
                save_path=rel_path
            )
            plots["reliability_diagram"] = rel_path
            
            # Confidence histogram
            conf_path = plots_dir / "confidence_histogram.png"
            self.calibration_analyzer.plot_confidence_histogram(
                y_prob, y_true, y_pred,
                save_path=conf_path
            )
            plots["confidence_histogram"] = conf_path
        
        # Per-class metrics plot
        class_metrics_path = plots_dir / "per_class_metrics.png"
        self._plot_per_class_metrics(
            metrics_result,
            save_path=class_metrics_path
        )
        plots["per_class_metrics"] = class_metrics_path
        
        # Code usage plot if available
        if codes is not None:
            code_usage_path = plots_dir / "code_usage.png"
            self._plot_code_usage(codes, save_path=code_usage_path)
            plots["code_usage"] = code_usage_path
        
        # Save QA report
        qa_report = self._generate_qa_report(qa_stats, metrics_result)
        qa_path = output_dir / "qa_report.json"
        with open(qa_path, 'w') as f:
            json.dump(qa_report, f, indent=2)
        
        # Save configuration
        if config:
            config_path = output_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Save raw predictions if requested
        if self.save_raw_predictions:
            preds_dir = output_dir / "predictions"
            preds_dir.mkdir(exist_ok=True)
            
            np.save(preds_dir / "y_true.npy", y_true)
            np.save(preds_dir / "y_pred.npy", y_pred)
            if y_prob is not None:
                np.save(preds_dir / "y_prob.npy", y_prob)
            if codes is not None:
                np.save(preds_dir / "codes.npy", codes)
        
        # Generate summary report
        summary_path = output_dir / "summary.md"
        self._generate_summary_report(
            metrics_result,
            calibration_result,
            qa_report,
            config,
            exp_hash,
            timestamp,
            summary_path
        )
        
        # Create bundle
        bundle = EvaluationBundle(
            exp_hash=exp_hash,
            timestamp=timestamp,
            config=config or {},
            metrics=json.loads(metrics_result.to_json()),
            calibration=json.loads(calibration_result.to_json()) if calibration_result else {},
            plots=plots,
            qa_stats=qa_report,
            output_dir=output_dir,
        )
        
        logger.info(f"Bundle complete: {bundle.summary()}")
        
        return bundle
    
    def _generate_exp_hash(
        self,
        config: Optional[Dict[str, Any]],
        exp_name: Optional[str],
    ) -> str:
        """Generate unique experiment hash."""
        hasher = hashlib.sha256()
        
        if exp_name:
            hasher.update(exp_name.encode())
        
        if config:
            config_str = json.dumps(config, sort_keys=True)
            hasher.update(config_str.encode())
        
        # Add timestamp for uniqueness
        hasher.update(str(datetime.now()).encode())
        
        return hasher.hexdigest()[:12]
    
    def _plot_confusion_matrix(
        self,
        cm: NDArray[np.int32],
        save_path: Path,
        normalize: bool = True,
    ) -> None:
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
        else:
            cm_norm = cm
        
        # Use seaborn heatmap
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'},
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(
        self,
        metrics_result,
        save_path: Path,
    ) -> None:
        """Plot per-class F1, precision, recall."""
        n_classes = len(metrics_result.per_class_f1)
        x = np.arange(n_classes)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        width = 0.25
        ax.bar(x - width, metrics_result.precision, width, label='Precision', alpha=0.8)
        ax.bar(x, metrics_result.recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics_result.per_class_f1, width, label='F1', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in range(n_classes)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add macro-F1 line
        ax.axhline(metrics_result.macro_f1, color='red', linestyle='--', 
                  alpha=0.5, label=f'Macro-F1: {metrics_result.macro_f1:.3f}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_code_usage(
        self,
        codes: NDArray[np.int32],
        save_path: Path,
        max_codes: int = 50,
    ) -> None:
        """Plot code usage distribution."""
        # Count code occurrences
        unique_codes, counts = np.unique(codes.flatten(), return_counts=True)
        
        # Sort by frequency
        sort_idx = np.argsort(counts)[::-1]
        unique_codes = unique_codes[sort_idx][:max_codes]
        counts = counts[sort_idx][:max_codes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of top codes
        ax1.bar(range(len(counts)), counts, alpha=0.7)
        ax1.set_xlabel('Code Rank')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Top {len(counts)} Code Usage')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative distribution
        cumsum = np.cumsum(counts) / np.sum(counts)
        ax2.plot(cumsum, 'o-', alpha=0.7)
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50%')
        ax2.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90%')
        ax2.set_xlabel('Number of Codes')
        ax2.set_ylabel('Cumulative Coverage')
        ax2.set_title('Code Coverage')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _generate_qa_report(
        self,
        qa_stats: Dict[str, Any],
        metrics_result,
    ) -> Dict[str, Any]:
        """Generate comprehensive QA report."""
        report = {
            "data_quality": qa_stats,
            "model_quality": {
                "accuracy": float(metrics_result.accuracy),
                "macro_f1": float(metrics_result.macro_f1),
                "ece": float(metrics_result.ece),
                "coverage": float(metrics_result.coverage),
            },
            "issues": [],
            "warnings": [],
        }
        
        # Check for issues
        if qa_stats.get("nan_events", 0) > 0:
            report["issues"].append(f"Found {qa_stats['nan_events']} NaN values")
        
        if qa_stats.get("inf_events", 0) > 0:
            report["issues"].append(f"Found {qa_stats['inf_events']} Inf values")
        
        if metrics_result.ece > 0.1:
            report["warnings"].append(f"High calibration error: ECE={metrics_result.ece:.3f}")
        
        if metrics_result.coverage < 0.5:
            report["warnings"].append(f"Low class coverage: {metrics_result.coverage:.2f}")
        
        # Add summary
        report["summary"] = {
            "total_issues": len(report["issues"]),
            "total_warnings": len(report["warnings"]),
            "qa_pass": len(report["issues"]) == 0,
        }
        
        return report
    
    def _generate_summary_report(
        self,
        metrics_result,
        calibration_result,
        qa_report: Dict[str, Any],
        config: Optional[Dict[str, Any]],
        exp_hash: str,
        timestamp: str,
        save_path: Path,
    ) -> None:
        """Generate markdown summary report."""
        lines = [
            f"# Evaluation Report",
            f"",
            f"**Experiment Hash**: `{exp_hash}`",
            f"**Timestamp**: {timestamp}",
            f"",
            f"## Metrics Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy | {metrics_result.accuracy:.3f} |",
            f"| Macro-F1 | {metrics_result.macro_f1:.3f} |",
            f"| ECE | {metrics_result.ece:.3f} |",
            f"| MCE | {metrics_result.mce:.3f} |",
            f"| Coverage | {metrics_result.coverage:.2f} |",
            f"| Motif Count | {metrics_result.motif_count} |",
            f"| Code Usage | {metrics_result.code_usage_percent:.1f}% |",
            f"| Perplexity | {metrics_result.perplexity:.2f} |",
            f"",
        ]
        
        # Add QA summary
        lines.extend([
            f"## Quality Assurance",
            f"",
            f"**QA Status**: {'✅ PASS' if qa_report['summary']['qa_pass'] else '❌ FAIL'}",
            f"",
        ])
        
        if qa_report["issues"]:
            lines.extend([
                f"### Issues",
                f"",
            ])
            for issue in qa_report["issues"]:
                lines.append(f"- {issue}")
            lines.append("")
        
        if qa_report["warnings"]:
            lines.extend([
                f"### Warnings",
                f"",
            ])
            for warning in qa_report["warnings"]:
                lines.append(f"- {warning}")
            lines.append("")
        
        # Add plots section
        lines.extend([
            f"## Visualizations",
            f"",
            f"- [Confusion Matrix](plots/confusion_matrix.png)",
            f"- [Reliability Diagram](plots/reliability_diagram.png)",
            f"- [Confidence Distribution](plots/confidence_histogram.png)",
            f"- [Per-Class Metrics](plots/per_class_metrics.png)",
            f"- [Code Usage](plots/code_usage.png)",
            f"",
        ])
        
        # Add configuration if provided
        if config:
            lines.extend([
                f"## Configuration",
                f"",
                f"```json",
                json.dumps(config, indent=2),
                f"```",
                f"",
            ])
        
        # Write report
        with open(save_path, 'w') as f:
            f.write('\n'.join(lines))