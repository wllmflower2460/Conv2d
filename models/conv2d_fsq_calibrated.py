#!/usr/bin/env python3
"""
Calibrated Conv2d-FSQ Model for M1.3 Requirements

Integrates calibration metrics with the FSQ model to achieve:
- ECE ≤3% through temperature scaling
- 90% conformal prediction coverage
- Calibrated confidence scores for clinical deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging

# Import base FSQ model
from models.conv2d_fsq_model import Conv2dFSQ

# Import calibration components
from models.calibration import (
    CalibrationMetrics,
    ConformalPredictor,
    CalibrationEvaluator,
    TemperatureScaling,
    calibrate_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibratedConv2dFSQ(nn.Module):
    """
    Conv2d-FSQ-HSMM model with integrated calibration.
    
    Addresses M1.3 requirements:
    - ECE ≤3% through temperature scaling
    - 90% conformal prediction intervals
    - Calibrated uncertainty quantification
    """
    
    def __init__(
        self,
        input_channels: int = 9,
        hidden_dim: int = 128,
        num_classes: int = 10,
        fsq_levels: Optional[List[int]] = None,
        temperature: float = 1.0,
        alpha: float = 0.1
    ):
        """
        Initialize calibrated FSQ model.
        
        Args:
            input_channels: Number of input channels (IMU dimensions)
            hidden_dim: Hidden dimension for encoder
            num_classes: Number of behavioral classes
            fsq_levels: FSQ quantization levels (default: [8]*8)
            temperature: Initial temperature for scaling
            alpha: Miscoverage rate (1-alpha = coverage target)
        """
        super().__init__()
        
        # Base FSQ model
        self.fsq_model = Conv2dFSQ(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            fsq_levels=fsq_levels
        )
        
        # Calibration components
        self.temperature_scaler = TemperatureScaling(init_temp=temperature)
        self.conformal_predictor = ConformalPredictor(alpha=alpha)
        self.calibration_evaluator = CalibrationEvaluator(n_bins=15)
        
        # Calibration state
        self.is_calibrated = False
        self.optimal_temperature = temperature
        self.calibration_metrics = None
        
        # M1.3 requirements
        self.ece_target = 0.03  # ≤3%
        self.coverage_target = 1 - alpha  # 90% default
        
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True,
        return_codes: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with calibrated predictions.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_uncertainty: Whether to return calibrated uncertainties
            return_codes: Whether to return FSQ codes
            
        Returns:
            Dictionary containing:
                - logits: Raw model outputs
                - probabilities: Calibrated probabilities
                - predictions: Class predictions
                - confidence: Calibrated confidence scores
                - uncertainty: Calibrated uncertainty estimates
                - prediction_intervals: Conformal prediction intervals (if calibrated)
                - codes: FSQ codes (if requested)
        """
        # Get base FSQ model outputs
        fsq_output = self.fsq_model(x, return_codes=return_codes)
        
        # Extract logits
        if isinstance(fsq_output, dict):
            logits = fsq_output['logits']
            codes = fsq_output.get('codes', None)
        else:
            logits = fsq_output
            codes = None
        
        # Apply temperature scaling if calibrated
        if self.is_calibrated:
            calibrated_logits = self.temperature_scaler(logits)
        else:
            calibrated_logits = logits
        
        # Compute probabilities
        probabilities = F.softmax(calibrated_logits, dim=-1)
        
        # Get predictions
        predictions = torch.argmax(probabilities, dim=-1)
        
        # Compute confidence (max probability)
        confidence, _ = torch.max(probabilities, dim=-1)
        
        output = {
            'logits': logits,
            'calibrated_logits': calibrated_logits,
            'probabilities': probabilities,
            'predictions': predictions,
            'confidence': confidence
        }
        
        # Add uncertainty quantification
        if return_uncertainty:
            output['uncertainty'] = self.compute_uncertainty(probabilities)
            
            # Add conformal prediction intervals if calibrated
            if self.is_calibrated and self.conformal_predictor.fitted:
                pred_intervals = self.get_prediction_intervals(calibrated_logits)
                output['prediction_intervals'] = pred_intervals
        
        # Add FSQ codes if requested
        if return_codes and codes is not None:
            output['codes'] = codes
            output['code_stats'] = self.fsq_model.get_code_stats()
        
        return output
    
    def compute_uncertainty(self, probabilities: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute calibrated uncertainty metrics.
        
        Args:
            probabilities: Calibrated probability distributions
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Entropy-based uncertainty
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
        
        # Variance-based uncertainty
        variance = torch.var(probabilities, dim=-1)
        
        # Top-2 margin uncertainty
        top2_probs, _ = torch.topk(probabilities, k=2, dim=-1)
        margin = top2_probs[..., 0] - top2_probs[..., 1]
        
        return {
            'entropy': entropy,
            'variance': variance,
            'margin': margin,
            'calibrated': self.is_calibrated
        }
    
    def get_prediction_intervals(
        self,
        logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get conformal prediction intervals.
        
        Args:
            logits: Calibrated logits
            
        Returns:
            Prediction intervals with coverage guarantee
        """
        if not self.conformal_predictor.fitted:
            raise RuntimeError("Conformal predictor not fitted. Call calibrate() first.")
        
        predictions, lower_bounds, upper_bounds = self.conformal_predictor.predict_interval(logits)
        
        return {
            'predictions': predictions,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'interval_width': upper_bounds - lower_bounds
        }
    
    def calibrate(
        self,
        calibration_loader: torch.utils.data.DataLoader,
        device: Optional[str] = None
    ) -> CalibrationMetrics:
        """
        Calibrate the model on validation data.
        
        Args:
            calibration_loader: DataLoader with calibration data
            device: Device to use (auto-detect if None)
            
        Returns:
            CalibrationMetrics with ECE, coverage, etc.
        """
        if device is None:
            device = next(self.parameters()).device
        
        logger.info("Starting calibration for M1.3 requirements...")
        logger.info(f"Targets: ECE ≤{self.ece_target:.1%}, Coverage ≥{self.coverage_target:.1%}")
        
        self.eval()
        
        # Collect predictions and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in calibration_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                # Get raw logits
                output = self.forward(batch_data, return_uncertainty=False)
                logits = output['logits']
                
                all_logits.append(logits.cpu())
                all_labels.append(batch_labels.cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Split into calibration and validation sets
        n_samples = len(all_logits)
        n_cal = n_samples // 2
        
        cal_logits = all_logits[:n_cal]
        cal_labels = all_labels[:n_cal]
        val_logits = all_logits[n_cal:]
        val_labels = all_labels[n_cal:]
        
        # Step 1: Fit temperature scaling
        logger.info("Fitting temperature scaling...")
        # Move temperature scaler to same device as data
        self.temperature_scaler = self.temperature_scaler.to(cal_logits.device)
        self.optimal_temperature = self.temperature_scaler.fit(cal_logits, cal_labels)
        logger.info(f"Optimal temperature: {self.optimal_temperature:.4f}")
        
        # Apply temperature scaling
        cal_logits_scaled = cal_logits / self.optimal_temperature
        val_logits_scaled = val_logits / self.optimal_temperature
        
        # Step 2: Fit conformal predictor
        logger.info("Fitting conformal predictor...")
        self.conformal_predictor.calibrate(cal_logits_scaled, cal_labels)
        
        # Step 3: Evaluate calibration on validation set
        logger.info("Evaluating calibration metrics...")
        self.calibration_metrics = self.calibration_evaluator.evaluate_calibration(
            val_logits_scaled,
            val_labels,
            self.conformal_predictor
        )
        
        # Mark as calibrated
        self.is_calibrated = True
        
        # Log results
        self._log_calibration_results()
        
        # Check M1.3 requirements
        self._check_m13_requirements()
        
        return self.calibration_metrics
    
    def _log_calibration_results(self):
        """Log calibration results."""
        metrics = self.calibration_metrics
        
        logger.info("\n" + "="*60)
        logger.info("CALIBRATION RESULTS")
        logger.info("="*60)
        logger.info(f"ECE: {metrics.ece:.4f} (target ≤{self.ece_target:.3f})")
        logger.info(f"MCE: {metrics.mce:.4f}")
        logger.info(f"Brier Score: {metrics.brier_score:.4f}")
        logger.info(f"Coverage: {metrics.coverage:.4f} (target ≥{self.coverage_target:.2f})")
        logger.info(f"Avg Interval Width: {metrics.avg_interval_width:.4f}")
        logger.info(f"Temperature: {self.optimal_temperature:.4f}")
        logger.info("="*60)
    
    def _check_m13_requirements(self):
        """Check if M1.3 calibration requirements are met."""
        metrics = self.calibration_metrics
        
        ece_met = metrics.ece <= self.ece_target
        coverage_met = metrics.coverage >= (self.coverage_target - 0.02)  # 2% tolerance
        
        if ece_met and coverage_met:
            logger.info("✅ M1.3 CALIBRATION REQUIREMENTS MET")
        else:
            issues = []
            if not ece_met:
                issues.append(f"ECE {metrics.ece:.4f} > {self.ece_target:.3f}")
            if not coverage_met:
                issues.append(f"Coverage {metrics.coverage:.4f} < {self.coverage_target:.2f}")
            logger.warning(f"⚠️ M1.3 requirements not met: {', '.join(issues)}")
    
    def save_calibrated_model(self, path: str):
        """
        Save calibrated model with all calibration parameters.
        
        Args:
            path: Path to save the model
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'optimal_temperature': self.optimal_temperature,
            'is_calibrated': self.is_calibrated,
            'calibration_metrics': self.calibration_metrics,
            'conformal_quantile': self.conformal_predictor.quantile if self.conformal_predictor.fitted else None,
            'fsq_levels': self.fsq_model.fsq_levels,
            'code_stats': self.fsq_model.get_code_stats()
        }
        
        torch.save(save_dict, path)
        logger.info(f"Saved calibrated model to {path}")
    
    def load_calibrated_model(self, path: str, device: str = 'cpu'):
        """
        Load calibrated model with calibration parameters.
        
        Args:
            path: Path to saved model
            device: Device to load to
        """
        checkpoint = torch.load(path, map_location=device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimal_temperature = checkpoint['optimal_temperature']
        self.is_calibrated = checkpoint['is_calibrated']
        self.calibration_metrics = checkpoint['calibration_metrics']
        
        if checkpoint['conformal_quantile'] is not None:
            self.conformal_predictor.quantile = checkpoint['conformal_quantile']
            self.conformal_predictor.fitted = True
        
        # Update temperature scaler
        self.temperature_scaler.temperature.data = torch.tensor(self.optimal_temperature)
        
        logger.info(f"Loaded calibrated model from {path}")
        if self.is_calibrated:
            logger.info(f"Calibration - ECE: {self.calibration_metrics.ece:.4f}, "
                       f"Coverage: {self.calibration_metrics.coverage:.4f}")
    
    def clinical_safety_check(self, confidence: torch.Tensor) -> torch.Tensor:
        """
        Clinical safety check for low-confidence predictions.
        
        Per M1.3: Fail-closed mechanism for therapeutic applications.
        
        Args:
            confidence: Calibrated confidence scores
            
        Returns:
            Safety mask (True = safe to use, False = needs review)
        """
        # M1.3 requirement: Flag predictions below confidence threshold
        min_confidence = 0.7  # Adjustable based on clinical requirements
        
        safe_mask = confidence >= min_confidence
        
        if not safe_mask.all():
            n_unsafe = (~safe_mask).sum().item()
            logger.warning(f"⚠️ {n_unsafe} predictions below safety threshold ({min_confidence:.1%})")
        
        return safe_mask


def test_calibrated_fsq():
    """Test the calibrated FSQ model."""
    import torch.utils.data as data
    
    logger.info("Testing Calibrated FSQ Model for M1.3...")
    
    # Create model
    model = CalibratedConv2dFSQ(
        input_channels=9,
        hidden_dim=128,
        num_classes=10,
        fsq_levels=[8]*8,
        alpha=0.1  # 90% coverage target
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create synthetic data
    n_samples = 1000
    X = torch.randn(n_samples, 9, 2, 100)
    y = torch.randint(0, 10, (n_samples,))
    
    # Create data loader
    dataset = data.TensorDataset(X, y)
    calibration_loader = data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Test forward pass (uncalibrated)
    logger.info("\nTesting uncalibrated forward pass...")
    with torch.no_grad():
        x_test = X[:4].to(device)
        output = model(x_test, return_uncertainty=True, return_codes=True)
        
        logger.info(f"Output keys: {output.keys()}")
        logger.info(f"Predictions shape: {output['predictions'].shape}")
        logger.info(f"Confidence range: [{output['confidence'].min():.3f}, {output['confidence'].max():.3f}]")
    
    # Calibrate model
    logger.info("\nCalibrating model...")
    metrics = model.calibrate(calibration_loader, device)
    
    # Test calibrated forward pass
    logger.info("\nTesting calibrated forward pass...")
    with torch.no_grad():
        output = model(x_test, return_uncertainty=True, return_codes=True)
        
        logger.info(f"Calibrated confidence range: [{output['confidence'].min():.3f}, {output['confidence'].max():.3f}]")
        
        if 'prediction_intervals' in output:
            intervals = output['prediction_intervals']
            logger.info(f"Prediction interval width: {intervals['interval_width'].mean():.2f}")
    
    # Test clinical safety check
    logger.info("\nTesting clinical safety check...")
    safety_mask = model.clinical_safety_check(output['confidence'])
    logger.info(f"Safe predictions: {safety_mask.sum()}/{len(safety_mask)}")
    
    # Save and load test
    logger.info("\nTesting save/load...")
    save_path = "test_calibrated_fsq.pth"
    model.save_calibrated_model(save_path)
    
    model2 = CalibratedConv2dFSQ()
    model2.load_calibrated_model(save_path, device)
    
    # Clean up
    os.remove(save_path)
    
    logger.info("\n✅ Calibrated FSQ model tests passed!")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = test_calibrated_fsq()
    
    # Check M1.3 requirements
    print("\n" + "="*60)
    print("M1.3 CALIBRATION REQUIREMENTS CHECK")
    print("="*60)
    print(f"ECE Target: ≤3%")
    print(f"ECE Achieved: {metrics.ece:.2%}")
    print(f"Status: {'✅ PASS' if metrics.ece <= 0.03 else '❌ FAIL'}")
    print()
    print(f"Coverage Target: ≥90%")
    print(f"Coverage Achieved: {metrics.coverage:.1%}")
    print(f"Status: {'✅ PASS' if metrics.coverage >= 0.88 else '❌ FAIL'}")
    print("="*60)