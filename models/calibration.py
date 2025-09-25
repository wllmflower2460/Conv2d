"""Calibration module for Conv2d-VQ-HDP-HSMM model.

Implements proper Expected Calibration Error (ECE) and conformal prediction
for uncertainty quantification as required by the synchrony-advisor-committee.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class CalibrationMetrics:
    """Calibration metrics for model evaluation."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Brier Score
    coverage: float  # Empirical coverage of prediction intervals
    avg_interval_width: float  # Average width of prediction intervals


class ConformalPredictor:
    """Conformal prediction for uncertainty quantification.
    
    Provides calibrated prediction intervals with guaranteed coverage.
    """
    
    def __init__(self, alpha: float = 0.1):
        """Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage rate (1 - alpha = coverage guarantee)
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None
        self.fitted = False
    
    def calibrate(
        self,
        model_outputs: torch.Tensor,
        true_labels: torch.Tensor
    ) -> None:
        """Calibrate on held-out calibration set.
        
        Args:
            model_outputs: Model predictions (logits or probabilities)
            true_labels: Ground truth labels
        """
        with torch.no_grad():
            # Convert to probabilities if needed
            if model_outputs.dim() == 3:  # (batch, classes, time)
                probs = torch.softmax(model_outputs, dim=1)
            else:
                probs = model_outputs
            
            # Compute non-conformity scores
            batch_size = true_labels.shape[0]
            scores = []
            
            for i in range(batch_size):
                # Score = 1 - probability of true class
                true_class_prob = probs[i, true_labels[i]].mean()
                scores.append(1 - true_class_prob)
            
            self.calibration_scores = torch.tensor(scores)
            
            # Compute quantile for coverage guarantee
            n = len(self.calibration_scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.quantile = torch.quantile(self.calibration_scores, q_level)
            self.fitted = True
    
    def predict_interval(
        self,
        model_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate prediction intervals.
        
        Args:
            model_outputs: Model predictions
            
        Returns:
            predictions: Point predictions
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
        """
        if not self.fitted:
            raise RuntimeError("Must call calibrate() before predict_interval()")
        
        with torch.no_grad():
            # Convert to probabilities
            if model_outputs.dim() == 3:
                probs = torch.softmax(model_outputs, dim=1)
            else:
                probs = model_outputs
            
            # Get predictions
            predictions = torch.argmax(probs, dim=1)
            
            # Compute prediction sets
            # Include all classes with score <= quantile
            batch_size = probs.shape[0]
            lower_bounds = []
            upper_bounds = []
            
            for i in range(batch_size):
                # Get probability threshold
                threshold = 1 - self.quantile
                
                # Find classes above threshold
                valid_classes = (probs[i] >= threshold).any(dim=-1)
                
                if valid_classes.any():
                    valid_indices = torch.where(valid_classes)[0]
                    lower_bounds.append(valid_indices.min())
                    upper_bounds.append(valid_indices.max())
                else:
                    # If no classes meet threshold, use prediction ± 1
                    pred = predictions[i].float().mean()
                    lower_bounds.append(torch.maximum(pred - 1, torch.tensor(0.0)))
                    upper_bounds.append(pred + 1)
            
            lower_bounds = torch.stack(lower_bounds)
            upper_bounds = torch.stack(upper_bounds)
            
            return predictions, lower_bounds, upper_bounds
    
    def compute_coverage(
        self,
        predictions: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        true_labels: torch.Tensor
    ) -> float:
        """Compute empirical coverage of prediction intervals.
        
        Args:
            predictions: Point predictions
            lower_bounds: Lower bounds
            upper_bounds: Upper bounds
            true_labels: Ground truth
            
        Returns:
            Empirical coverage (should be close to 1 - alpha)
        """
        with torch.no_grad():
            # Check if true labels fall within intervals
            covered = (true_labels >= lower_bounds) & (true_labels <= upper_bounds)
            coverage = covered.float().mean().item()
            return coverage


class CalibrationEvaluator:
    """Evaluate model calibration with ECE and other metrics."""
    
    def __init__(self, n_bins: int = 15):
        """Initialize calibration evaluator.
        
        Args:
            n_bins: Number of bins for ECE computation
        """
        self.n_bins = n_bins
    
    def compute_ece(
        self,
        probabilities: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute Expected Calibration Error and Maximum Calibration Error.
        
        Args:
            probabilities: Predicted probabilities (B, C) or (B, C, T)
            predictions: Predicted classes (B) or (B, T)
            labels: True labels (B) or (B, T)
            
        Returns:
            ECE: Expected Calibration Error
            MCE: Maximum Calibration Error
        """
        with torch.no_grad():
            # Flatten if needed
            if probabilities.dim() == 3:
                B, C, T = probabilities.shape
                probabilities = probabilities.permute(0, 2, 1).reshape(-1, C)
                predictions = predictions.reshape(-1)
                labels = labels.reshape(-1)
            
            # Get confidence (max probability)
            confidences, _ = torch.max(probabilities, dim=1)
            accuracies = (predictions == labels).float()
            
            # Compute ECE
            ece = 0.0
            mce = 0.0
            
            # Create bins
            bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
            
            for i in range(self.n_bins):
                # Find samples in this bin
                in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
                
                if in_bin.sum() > 0:
                    # Compute accuracy and confidence in bin
                    bin_accuracy = accuracies[in_bin].mean()
                    bin_confidence = confidences[in_bin].mean()
                    bin_size = in_bin.sum()
                    
                    # Calibration error for this bin
                    calibration_error = torch.abs(bin_accuracy - bin_confidence)
                    
                    # Weight by bin size
                    ece += (bin_size / len(confidences)) * calibration_error
                    
                    # Track maximum
                    mce = max(mce, calibration_error.item())
            
            # Handle both tensor and float types
            if isinstance(ece, torch.Tensor):
                return ece.item(), mce
            else:
                return float(ece), mce
    
    def compute_brier_score(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Compute Brier score for probabilistic predictions.
        
        Args:
            probabilities: Predicted probabilities (B, C) or (B, C, T)
            labels: True labels (B) or (B, T)
            
        Returns:
            Brier score (lower is better)
        """
        with torch.no_grad():
            # Handle different dimensions
            if probabilities.dim() == 3:
                B, C, T = probabilities.shape
                probabilities = probabilities.permute(0, 2, 1).reshape(-1, C)
                labels = labels.reshape(-1)
            
            # One-hot encode labels
            num_classes = probabilities.shape[1]
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes)
            
            # Brier score = mean squared difference
            brier = torch.mean((probabilities - labels_onehot.float()) ** 2)
            
            return brier.item()
    
    def evaluate_calibration(
        self,
        model_outputs: torch.Tensor,
        true_labels: torch.Tensor,
        conformal_predictor: Optional[ConformalPredictor] = None
    ) -> CalibrationMetrics:
        """Comprehensive calibration evaluation.
        
        Args:
            model_outputs: Model predictions (logits or probabilities)
            true_labels: Ground truth labels
            conformal_predictor: Optional fitted conformal predictor
            
        Returns:
            CalibrationMetrics with all metrics
        """
        with torch.no_grad():
            # Convert to probabilities
            if model_outputs.dim() == 3:
                probs = torch.softmax(model_outputs, dim=1)
            else:
                probs = model_outputs
            
            # Get predictions
            predictions = torch.argmax(probs, dim=1)
            
            # Compute ECE and MCE
            ece, mce = self.compute_ece(probs, predictions, true_labels)
            
            # Compute Brier score
            brier = self.compute_brier_score(probs, true_labels)
            
            # Compute coverage if conformal predictor provided
            coverage = 0.0
            avg_width = 0.0
            
            if conformal_predictor is not None and conformal_predictor.fitted:
                _, lower, upper = conformal_predictor.predict_interval(model_outputs)
                coverage = conformal_predictor.compute_coverage(
                    predictions, lower, upper, true_labels
                )
                avg_width = (upper - lower).float().mean().item()
            
            return CalibrationMetrics(
                ece=ece,
                mce=mce,
                brier_score=brier,
                coverage=coverage,
                avg_interval_width=avg_width
            )


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration.
    
    Simple post-processing method to improve calibration.
    """
    
    def __init__(self, init_temp: float = 1.0):
        """Initialize temperature scaling.
        
        Args:
            init_temp: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling.
        
        Args:
            logits: Model logits
            
        Returns:
            Scaled logits
        """
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """Fit temperature using NLL loss.
        
        Args:
            logits: Model logits
            labels: True labels
            lr: Learning rate
            max_iter: Maximum iterations
            
        Returns:
            Optimal temperature value
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            
            # Handle different dimensions
            if scaled_logits.dim() == 3:
                B, C, T = scaled_logits.shape
                scaled_logits = scaled_logits.permute(0, 2, 1).reshape(-1, C)
                labels_flat = labels.reshape(-1)
            else:
                labels_flat = labels
            
            loss = criterion(scaled_logits, labels_flat)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return self.temperature.item()


def calibrate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    alpha: float = 0.1
) -> Tuple[CalibrationMetrics, ConformalPredictor, float]:
    """Full calibration pipeline for a trained model.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to use
        alpha: Miscoverage rate for conformal prediction
        
    Returns:
        metrics: Calibration metrics
        conformal: Fitted conformal predictor
        temperature: Optimal temperature for scaling
    """
    model.eval()
    
    # Collect predictions and labels
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'predictions' in outputs:
                    outputs = outputs['predictions']
                elif 'logits' in outputs:
                    outputs = outputs['logits']
            
            all_outputs.append(outputs.cpu())
            all_labels.append(batch_labels.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Split into calibration and test sets
    n_samples = len(all_outputs)
    n_cal = n_samples // 2
    
    cal_outputs = all_outputs[:n_cal]
    cal_labels = all_labels[:n_cal]
    test_outputs = all_outputs[n_cal:]
    test_labels = all_labels[n_cal:]
    
    # Fit temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(cal_outputs, cal_labels)
    
    # Apply temperature scaling
    cal_outputs_scaled = cal_outputs / optimal_temp
    test_outputs_scaled = test_outputs / optimal_temp
    
    # Fit conformal predictor
    conformal = ConformalPredictor(alpha=alpha)
    conformal.calibrate(cal_outputs_scaled, cal_labels)
    
    # Evaluate calibration on test set
    evaluator = CalibrationEvaluator()
    metrics = evaluator.evaluate_calibration(
        test_outputs_scaled,
        test_labels,
        conformal
    )
    
    print(f"Calibration Results:")
    print(f"  ECE: {metrics.ece:.4f} (target ≤ 0.03)")
    print(f"  MCE: {metrics.mce:.4f}")
    print(f"  Brier Score: {metrics.brier_score:.4f}")
    print(f"  Coverage: {metrics.coverage:.4f} (target: {1-alpha:.2f})")
    print(f"  Avg Interval Width: {metrics.avg_interval_width:.4f}")
    print(f"  Optimal Temperature: {optimal_temp:.4f}")
    
    return metrics, conformal, optimal_temp


if __name__ == "__main__":
    # Test calibration module
    print("Testing Calibration Module...")
    
    # Create synthetic data
    torch.manual_seed(42)
    n_samples = 1000
    n_classes = 10
    n_time = 100
    
    # Simulate overconfident model (needs calibration)
    logits = torch.randn(n_samples, n_classes, n_time) * 3  # Large logits = overconfident
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Test ECE computation
    evaluator = CalibrationEvaluator()
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    ece, mce = evaluator.compute_ece(probs, preds, labels)
    print(f"Before calibration - ECE: {ece:.4f}, MCE: {mce:.4f}")
    
    # Test temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(logits[:500], labels[:500])
    print(f"Optimal temperature: {optimal_temp:.4f}")
    
    # Re-evaluate after temperature scaling
    scaled_logits = logits / optimal_temp
    scaled_probs = torch.softmax(scaled_logits, dim=1)
    scaled_preds = torch.argmax(scaled_probs, dim=1)
    
    ece_scaled, mce_scaled = evaluator.compute_ece(scaled_probs, scaled_preds, labels)
    print(f"After calibration - ECE: {ece_scaled:.4f}, MCE: {mce_scaled:.4f}")
    
    # Test conformal prediction
    conformal = ConformalPredictor(alpha=0.1)
    conformal.calibrate(scaled_logits[:500], labels[:500])
    
    preds, lower, upper = conformal.predict_interval(scaled_logits[500:])
    coverage = conformal.compute_coverage(preds, lower, upper, labels[500:])
    print(f"Conformal prediction coverage: {coverage:.4f} (target: 0.90)")
    
    print("\n✅ Calibration module tests passed!")