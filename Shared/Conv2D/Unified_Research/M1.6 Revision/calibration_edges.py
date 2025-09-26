"""
Calibration Edge Cases Fixes
Addresses ECE binning, conformal quantiles, and dimension-safe operations
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union


class CalibrationMetrics:
    """
    Fixed calibration metrics addressing committee feedback.
    
    FIXES:
    1. ECE first bin now left-closed [0, b1] instead of (0, b1]
    2. Conformal quantile clamped to [0, 1]
    3. Dimension-agnostic interval selection
    """
    
    def __init__(self, n_bins=15):
        """
        Initialize calibration metrics.
        
        Args:
            n_bins: Number of bins for ECE calculation
        """
        self.n_bins = n_bins
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.confidences = []
        self.accuracies = []
        self.predictions = []
        self.labels = []
    
    def update(self, probs, labels):
        """
        Update with batch of predictions.
        
        Args:
            probs: Predicted probabilities (B, C) or (B, C, T)
            labels: True labels (B,) or (B, T)
        """
        # Get predicted class and confidence
        if probs.dim() == 3:  # (B, C, T)
            max_probs, preds = probs.max(dim=1)  # (B, T)
        else:  # (B, C)
            max_probs, preds = probs.max(dim=1)  # (B,)
        
        # Store for later computation
        self.confidences.append(max_probs.cpu())
        self.predictions.append(preds.cpu())
        self.labels.append(labels.cpu())
        
        # Compute accuracies
        correct = (preds == labels).float()
        self.accuracies.append(correct.cpu())
    
    def compute_ece(self):
        """
        Compute Expected Calibration Error with fixed binning.
        
        FIX: First bin is now left-closed [0, b1] instead of (0, b1]
        
        Returns:
            ece: Expected calibration error
            per_bin_stats: Statistics for each bin
        """
        if not self.confidences:
            return 0.0, []
        
        # Concatenate all batches
        confidences = torch.cat(self.confidences)
        accuracies = torch.cat(self.accuracies)
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        per_bin_stats = []
        
        for i in range(self.n_bins):
            # FIX: First bin is left-closed
            if i == 0:
                in_bin = (confidences >= bin_lowers[i]) & (confidences <= bin_uppers[i])
            else:
                in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
            
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].float().mean()
                
                # ECE contribution from this bin
                ece += prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                
                per_bin_stats.append({
                    'bin_id': i,
                    'range': f"[{bin_lowers[i]:.2f}, {bin_uppers[i]:.2f}]" if i == 0 
                            else f"({bin_lowers[i]:.2f}, {bin_uppers[i]:.2f}]",
                    'count': in_bin.sum().item(),
                    'avg_conf': avg_confidence_in_bin.item(),
                    'accuracy': accuracy_in_bin.item(),
                    'gap': (avg_confidence_in_bin - accuracy_in_bin).item()
                })
        
        return ece.item(), per_bin_stats
    
    def compute_brier_score(self):
        """
        Compute Brier score for probability calibration.
        
        Returns:
            brier_score: Mean squared difference between predicted probs and outcomes
        """
        if not self.predictions:
            return 0.0
        
        confidences = torch.cat(self.confidences)
        accuracies = torch.cat(self.accuracies)
        
        # Brier score: mean((p - y)^2) where y ∈ {0, 1}
        brier_score = ((confidences - accuracies) ** 2).mean()
        
        return brier_score.item()


class ConformalPredictor:
    """
    Conformal prediction with fixed quantile computation.
    
    FIX: Quantile level clamped to [0, 1] to prevent edge cases
    """
    
    def __init__(self, alpha=0.1):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.q_level = None
        self.threshold = None
    
    def calibrate(self, probs, labels):
        """
        Calibrate conformal predictor on validation set.
        
        Args:
            probs: Predicted probabilities (N, C) or (N, C, T)
            labels: True labels (N,) or (N, T)
        """
        n = len(labels)
        
        # Compute conformity scores (1 - prob of true class)
        if probs.dim() == 3:  # (N, C, T)
            # Flatten time dimension for calibration
            probs = probs.reshape(-1, probs.shape[1])
            labels = labels.reshape(-1)
        
        # Get probability of true class
        true_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()
        self.calibration_scores = 1 - true_class_probs
        
        # FIX: Clamp quantile level to [0, 1]
        q_level = float(np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_level = min(max(q_level, 0.0), 1.0)
        
        # Compute threshold
        self.threshold = torch.quantile(self.calibration_scores, self.q_level)
        
        return self.threshold
    
    def predict_set(self, probs):
        """
        Generate prediction sets with guaranteed coverage.
        
        FIX: Dimension-agnostic interval selection
        
        Args:
            probs: Predicted probabilities (B, C) or (B, C, T)
            
        Returns:
            prediction_sets: List of prediction sets
            set_sizes: Sizes of prediction sets
        """
        if self.threshold is None:
            raise ValueError("Must calibrate before prediction")
        
        batch_size = probs.shape[0]
        prediction_sets = []
        set_sizes = []
        
        for i in range(batch_size):
            # FIX: Handle both 2D and 3D cases
            if probs.dim() == 3:  # (B, C, T)
                # Check any time step exceeds threshold
                valid_classes = (probs[i] >= 1 - self.threshold).any(dim=-1)
            else:  # (B, C)
                # Direct threshold check
                valid_classes = (probs[i] >= 1 - self.threshold)
            
            # Get indices of valid classes
            pred_set = torch.where(valid_classes)[0].tolist()
            
            # Ensure at least one prediction
            if len(pred_set) == 0:
                # Take argmax as fallback
                if probs.dim() == 3:
                    pred_set = [probs[i].max(dim=-1).values.argmax().item()]
                else:
                    pred_set = [probs[i].argmax().item()]
            
            prediction_sets.append(pred_set)
            set_sizes.append(len(pred_set))
        
        return prediction_sets, set_sizes
    
    def evaluate_coverage(self, prediction_sets, labels):
        """
        Evaluate empirical coverage of prediction sets.
        
        Args:
            prediction_sets: List of prediction sets
            labels: True labels
            
        Returns:
            coverage: Empirical coverage rate
            avg_set_size: Average prediction set size
        """
        covered = 0
        total_size = 0
        
        for pred_set, label in zip(prediction_sets, labels):
            if label.item() in pred_set:
                covered += 1
            total_size += len(pred_set)
        
        coverage = covered / len(labels)
        avg_set_size = total_size / len(labels)
        
        return coverage, avg_set_size


class TemperatureScaling:
    """Temperature scaling for calibration."""
    
    def __init__(self):
        """Initialize temperature scaling."""
        self.temperature = 1.0
    
    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """
        Fit temperature using validation set.
        
        Args:
            logits: Raw logits (N, C)
            labels: True labels (N,)
            lr: Learning rate
            max_iter: Maximum iterations
            
        Returns:
            Final temperature value
        """
        temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        criterion = torch.nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        self.temperature = temperature.item()
        return self.temperature
    
    def calibrate(self, logits):
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw logits
            
        Returns:
            Calibrated probabilities
        """
        return torch.softmax(logits / self.temperature, dim=-1)


# Integration test
def test_calibration_suite():
    """Test all calibration components with edge cases."""
    
    print("="*60)
    print("CALIBRATION TEST SUITE")
    print("="*60)
    
    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 1000
    n_classes = 10
    
    # Test both 2D and 3D cases
    for dims in ['2D', '3D']:
        print(f"\n{dims} Test Case:")
        print("-"*40)
        
        if dims == '2D':
            logits = torch.randn(n_samples, n_classes)
            labels = torch.randint(0, n_classes, (n_samples,))
        else:
            n_time = 5
            logits = torch.randn(n_samples, n_classes, n_time)
            labels = torch.randint(0, n_classes, (n_samples, n_time))
        
        # Convert to probabilities
        if dims == '3D':
            probs = torch.softmax(logits, dim=1)
        else:
            probs = torch.softmax(logits, dim=-1)
        
        # Test calibration metrics
        metrics = CalibrationMetrics(n_bins=10)
        metrics.update(probs[:800], labels[:800])
        
        ece, bin_stats = metrics.compute_ece()
        brier = metrics.compute_brier_score()
        
        print(f"ECE: {ece:.4f}")
        print(f"Brier Score: {brier:.4f}")
        
        # Show first few bins
        print("\nFirst 3 ECE bins:")
        for stat in bin_stats[:3]:
            print(f"  {stat['range']}: n={stat['count']}, "
                  f"conf={stat['avg_conf']:.3f}, acc={stat['accuracy']:.3f}")
        
        # Test conformal prediction
        if dims == '2D':  # Simpler for 2D case
            conformal = ConformalPredictor(alpha=0.1)
            
            # Calibrate
            threshold = conformal.calibrate(probs[:500], labels[:500])
            print(f"\nConformal threshold: {threshold:.4f}")
            print(f"Quantile level (clamped): {conformal.q_level:.4f}")
            
            # Predict
            pred_sets, set_sizes = conformal.predict_set(probs[500:600])
            coverage, avg_size = conformal.evaluate_coverage(
                pred_sets, labels[500:600]
            )
            
            print(f"Coverage: {coverage:.2%} (target: 90%)")
            print(f"Avg set size: {avg_size:.2f}")
    
    # Test edge cases
    print("\n" + "="*60)
    print("EDGE CASE TESTS")
    print("="*60)
    
    # Edge case 1: All predictions at confidence 0
    print("\n1. Zero confidence edge case:")
    metrics = CalibrationMetrics(n_bins=5)
    zero_probs = torch.zeros(100, 10)
    zero_probs[:, 0] = 1e-10  # Tiny prob to avoid numerical issues
    zero_labels = torch.zeros(100, dtype=torch.long)
    metrics.update(zero_probs, zero_labels)
    ece, _ = metrics.compute_ece()
    print(f"   ECE with near-zero confidence: {ece:.4f}")
    
    # Edge case 2: Perfect calibration
    print("\n2. Perfect calibration:")
    perfect_probs = torch.eye(10).repeat(10, 1)  # Perfect predictions
    perfect_labels = torch.arange(10).repeat(10)
    metrics.reset()
    metrics.update(perfect_probs, perfect_labels)
    ece, _ = metrics.compute_ece()
    print(f"   ECE with perfect calibration: {ece:.4f}")
    
    # Edge case 3: Extreme quantile
    print("\n3. Extreme quantile (n=1):")
    conformal = ConformalPredictor(alpha=0.1)
    single_prob = torch.tensor([[0.9, 0.1]])
    single_label = torch.tensor([0])
    threshold = conformal.calibrate(single_prob, single_label)
    print(f"   Clamped quantile: {conformal.q_level:.4f}")


if __name__ == "__main__":
    test_calibration_suite()
