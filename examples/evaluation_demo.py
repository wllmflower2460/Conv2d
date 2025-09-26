#!/usr/bin/env python3
"""Demonstration of evaluation and reporting system.

Shows how to:
1. Calculate standard metrics
2. Analyze calibration
3. Generate evaluation bundles
4. Track QA issues
5. Create reproducible reports
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from sklearn.datasets import make_classification

from conv2d.metrics import (
    BundleGenerator,
    CalibrationAnalyzer,
    MetricsCalculator,
    QATracker,
)


def example_basic_metrics():
    """Example 1: Basic metrics calculation."""
    print("=" * 60)
    print("Example 1: Basic Metrics")
    print("=" * 60)
    
    # Create synthetic predictions
    n_samples = 1000
    n_classes = 4
    
    # Generate data
    X, y_true = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_classes=n_classes,
        random_state=42,
    )
    
    # Simulate predictions (with some errors)
    y_prob = np.random.rand(n_samples, n_classes).astype(np.float32)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Add some correct predictions
    correct_mask = np.random.rand(n_samples) < 0.7
    y_pred[correct_mask] = y_true[correct_mask]
    
    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.compute_all(y_true, y_pred, y_prob)
    
    print(f"\nMetrics Summary:")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"  Macro-F1: {metrics.macro_f1:.3f}")
    print(f"  ECE: {metrics.ece:.3f}")
    print(f"  MCE: {metrics.mce:.3f}")
    print(f"  Coverage: {metrics.coverage:.2f}")
    print(f"  Motif count: {metrics.motif_count}")
    
    # Show per-class F1
    print(f"\nPer-class F1 scores:")
    for i, f1 in enumerate(metrics.per_class_f1):
        print(f"  Class {i}: {f1:.3f}")
    
    print()


def example_calibration_analysis():
    """Example 2: Calibration analysis with reliability diagrams."""
    print("=" * 60)
    print("Example 2: Calibration Analysis")
    print("=" * 60)
    
    # Create predictions with varying calibration
    n_samples = 1000
    
    # Well-calibrated model
    y_true = np.random.randint(0, 2, n_samples)
    confidence_well = np.random.uniform(0.5, 1.0, n_samples)
    y_prob_well = np.zeros((n_samples, 2), dtype=np.float32)
    
    for i in range(n_samples):
        if y_true[i] == 1:
            y_prob_well[i, 1] = confidence_well[i]
            y_prob_well[i, 0] = 1 - confidence_well[i]
        else:
            y_prob_well[i, 0] = confidence_well[i]
            y_prob_well[i, 1] = 1 - confidence_well[i]
    
    # Overconfident model
    y_prob_over = y_prob_well.copy()
    y_prob_over = np.power(y_prob_over, 0.5)  # Push toward extremes
    y_prob_over = y_prob_over / y_prob_over.sum(axis=1, keepdims=True)
    
    # Underconfident model
    y_prob_under = y_prob_well.copy()
    y_prob_under = np.power(y_prob_under, 2)  # Push toward center
    y_prob_under = y_prob_under / y_prob_under.sum(axis=1, keepdims=True)
    
    # Analyze calibration
    analyzer = CalibrationAnalyzer(n_bins=10)
    
    calib_well = analyzer.analyze(y_true, y_prob_well)
    calib_over = analyzer.analyze(y_true, y_prob_over)
    calib_under = analyzer.analyze(y_true, y_prob_under)
    
    print("\nCalibration Results:")
    print(f"  Well-calibrated:  ECE={calib_well.ece:.3f}, MCE={calib_well.mce:.3f}")
    print(f"  Overconfident:    ECE={calib_over.ece:.3f}, MCE={calib_over.mce:.3f}")
    print(f"  Underconfident:   ECE={calib_under.ece:.3f}, MCE={calib_under.mce:.3f}")
    
    # Save reliability diagram
    output_dir = Path("temp_plots")
    output_dir.mkdir(exist_ok=True)
    
    analyzer.plot_reliability_diagram(
        calib_well,
        save_path=output_dir / "reliability_well.png",
        title="Well-Calibrated Model"
    )
    
    print(f"\nReliability diagram saved to {output_dir}")
    print()


def example_qa_tracking():
    """Example 3: Quality assurance tracking."""
    print("=" * 60)
    print("Example 3: QA Tracking")
    print("=" * 60)
    
    # Create data with various issues
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Add NaN values
    nan_mask = np.random.rand(n_samples, n_features) < 0.01
    X[nan_mask] = np.nan
    
    # Add Inf values
    inf_mask = np.random.rand(n_samples, n_features) < 0.005
    X[inf_mask] = np.inf
    
    # Add outliers
    outlier_mask = np.random.rand(n_samples) < 0.05
    X[outlier_mask, :10] *= 100
    
    # Add zero-variance features
    X[:, 20:25] = 1.0
    
    # Track QA issues
    tracker = QATracker()
    qa_stats = tracker.check_data(X)
    
    print("\nQA Statistics:")
    print(f"  Total samples: {qa_stats['total_samples']}")
    print(f"  NaN events: {qa_stats['nan_events']}")
    print(f"  Inf events: {qa_stats['inf_events']}")
    print(f"  Zero variance features: {qa_stats['zero_variance_features']}")
    print(f"  Outlier samples: {qa_stats['outlier_samples']}")
    print(f"  NaN rate: {qa_stats['nan_rate']:.3%}")
    print(f"  Outlier rate: {qa_stats['outlier_rate']:.3%}")
    
    # Record fallbacks
    tracker.record_fallback("Missing sensor data")
    tracker.record_failed_window("Insufficient samples")
    
    final_stats = tracker.get_stats()
    print(f"\nAdditional Issues:")
    print(f"  Mean fallbacks: {final_stats['mean_fallbacks']}")
    print(f"  Failed windows: {final_stats['failed_windows']}")
    print()


def example_evaluation_bundle():
    """Example 4: Complete evaluation bundle generation."""
    print("=" * 60)
    print("Example 4: Evaluation Bundle")
    print("=" * 60)
    
    # Create realistic evaluation scenario
    n_samples = 500
    n_classes = 4
    
    # Generate data
    X, y_true = make_classification(
        n_samples=n_samples,
        n_features=100,
        n_informative=50,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=42,
    )
    X = X.astype(np.float32)
    y_true = y_true.astype(np.int32)
    
    # Simulate model predictions
    # Add noise to true labels for predictions
    y_pred = y_true.copy()
    error_mask = np.random.rand(n_samples) < 0.25
    y_pred[error_mask] = np.random.randint(0, n_classes, error_mask.sum())
    
    # Generate probabilities
    y_prob = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(n_samples):
        y_prob[i, y_pred[i]] = np.random.uniform(0.6, 0.95)
        remaining = 1 - y_prob[i, y_pred[i]]
        other_probs = np.random.dirichlet(np.ones(n_classes - 1)) * remaining
        j = 0
        for k in range(n_classes):
            if k != y_pred[i]:
                y_prob[i, k] = other_probs[j]
                j += 1
    
    # Simulate FSQ codes
    codes = np.random.randint(0, 240, (n_samples, 100), dtype=np.int32)
    # Make some codes more common
    common_codes = np.random.choice(20, size=(n_samples, 30))
    codes[:, :30] = common_codes
    
    # Configuration
    config = {
        "model": "conv2d_fsq",
        "dataset": "example",
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
        },
        "fsq": {
            "levels": [8, 6, 5],
            "embedding_dim": 64,
        },
    }
    
    # Generate bundle
    generator = BundleGenerator(
        output_base=Path("reports"),
        save_raw_predictions=True,
    )
    
    bundle = generator.generate(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        codes=codes,
        config=config,
        exp_name="example_experiment",
        X_data=X,
    )
    
    print(f"\nBundle Generated:")
    print(f"  Hash: {bundle.exp_hash}")
    print(f"  Output: {bundle.output_dir}")
    
    print(f"\nKey Metrics:")
    print(f"  Accuracy: {bundle.metrics['accuracy']:.3f}")
    print(f"  Macro-F1: {bundle.metrics['macro_f1']:.3f}")
    print(f"  ECE: {bundle.calibration.get('ece', 0):.3f}")
    
    print(f"\nGenerated Files:")
    for name, path in bundle.plots.items():
        print(f"  {name}: {path.name}")
    
    print(f"\nQA Summary:")
    print(f"  Issues: {bundle.qa_stats['summary']['total_issues']}")
    print(f"  Warnings: {bundle.qa_stats['summary']['total_warnings']}")
    print(f"  QA Pass: {bundle.qa_stats['summary']['qa_pass']}")
    print()


def example_behavioral_metrics():
    """Example 5: Behavioral-specific metrics."""
    print("=" * 60)
    print("Example 5: Behavioral Metrics")
    print("=" * 60)
    
    # Create motif sequence
    T = 1000
    motif_sequence = np.zeros(T, dtype=np.int32)
    
    # Add realistic behavioral patterns
    current_pos = 0
    motifs = [0, 1, 2, 3]
    
    while current_pos < T:
        # Choose motif
        motif = np.random.choice(motifs)
        # Choose duration (realistic dwell times)
        if motif == 0:  # Rest - longer
            duration = np.random.poisson(50)
        elif motif == 1:  # Walk - medium
            duration = np.random.poisson(30)
        else:  # Run/play - shorter
            duration = np.random.poisson(15)
        
        duration = min(duration, T - current_pos)
        motif_sequence[current_pos:current_pos + duration] = motif
        current_pos += duration
    
    # Calculate behavioral metrics
    calculator = MetricsCalculator()
    behavioral_metrics = calculator.compute_behavioral_metrics(motif_sequence)
    
    print("\nBehavioral Metrics:")
    print(f"  Transition rate: {behavioral_metrics['transition_rate']:.3f}")
    print(f"  Mean dwell: {behavioral_metrics['mean_dwell']:.1f} frames")
    print(f"  Median dwell: {behavioral_metrics['median_dwell']:.1f} frames")
    print(f"  Min dwell: {behavioral_metrics['min_dwell']:.0f} frames")
    print(f"  Max dwell: {behavioral_metrics['max_dwell']:.0f} frames")
    print(f"  Motif entropy: {behavioral_metrics['motif_entropy']:.2f}")
    
    # Show motif distribution
    unique, counts = np.unique(motif_sequence, return_counts=True)
    print("\nMotif Distribution:")
    for motif, count in zip(unique, counts):
        print(f"  Motif {motif}: {count/T:.1%}")
    print()


def cleanup_temp_files():
    """Clean up temporary files created by examples."""
    import shutil
    
    # Clean up temp directories
    for dir_name in ["temp_plots", "reports"]:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"Cleaned up {dir_name}/")


def main():
    """Run all evaluation examples."""
    examples = [
        example_basic_metrics,
        example_calibration_analysis,
        example_qa_tracking,
        example_evaluation_bundle,
        example_behavioral_metrics,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example {example_func.__name__} failed: {e}")
            print()
    
    # Clean up
    cleanup_temp_files()


if __name__ == "__main__":
    main()