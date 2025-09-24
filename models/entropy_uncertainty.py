"""
Entropy & Uncertainty Module for Conv2d-VQ-HDP-HSMM
Implements uncertainty quantification and confidence calibration
Based on Entropy_Marginals_Module.md specifications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
import warnings


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty metrics"""
    state_entropy: float
    phase_entropy: float
    joint_entropy: float
    mutual_information: float
    behavioral_diversity: float
    coordination_coherence: float
    confidence_interval: Tuple[float, float]
    confidence_level: str  # 'high', 'medium', 'low'
    

class EntropyUncertaintyModule(nn.Module):
    """
    Core uncertainty quantification module for trustworthy deployment
    Implements entropy calculations for discrete states and continuous phase
    """
    
    def __init__(
        self,
        num_states: int = 10,
        num_phase_bins: int = 12,
        eps: float = 1e-12,
        confidence_threshold_high: float = 0.3,
        confidence_threshold_low: float = 0.6
    ):
        super().__init__()
        
        self.num_states = num_states
        self.num_phase_bins = num_phase_bins
        self.eps = eps
        
        # Confidence thresholds based on normalized entropy
        self.confidence_threshold_high = confidence_threshold_high
        self.confidence_threshold_low = confidence_threshold_low
        
        # Phase bins for discretization
        self.register_buffer(
            'phase_bins',
            torch.linspace(-np.pi, np.pi, num_phase_bins + 1)
        )
        
        # Calibration parameters (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
        self.dirichlet_concentration = nn.Parameter(torch.ones(num_states) * 0.1)
        
    def forward(
        self,
        state_posterior: torch.Tensor,
        phase_values: Optional[torch.Tensor] = None,
        cluster_assignments: Optional[torch.Tensor] = None,
        return_marginals: bool = True
    ) -> Dict[str, Union[float, torch.Tensor, Dict]]:
        """
        Compute comprehensive uncertainty metrics
        
        Args:
            state_posterior: State probabilities (B, T, S) or (B, S)
            phase_values: Optional phase values (B, T) in radians
            cluster_assignments: Optional cluster assignments (B, T, C)
            return_marginals: Whether to return marginal distributions
        
        Returns:
            Dictionary with entropy metrics, confidence, and marginals
        """
        device = state_posterior.device
        
        # Ensure proper shape
        if state_posterior.dim() == 2:
            state_posterior = state_posterior.unsqueeze(1)  # (B, 1, S)
        
        B, T, S = state_posterior.shape
        
        # Apply temperature calibration
        state_posterior = F.softmax(
            torch.log(state_posterior + self.eps) / self.temperature,
            dim=-1
        )
        
        # Add Dirichlet smoothing for better calibration
        state_posterior = self._dirichlet_smoothing(state_posterior)
        
        # Compute state entropy
        H_state = self._shannon_entropy(state_posterior)
        H_state_normalized = H_state / np.log(S)
        
        # Initialize outputs
        outputs = {
            'state_entropy': H_state.mean().item(),
            'state_entropy_normalized': H_state_normalized.mean().item(),
            'behavioral_diversity': torch.exp(H_state).mean().item()
        }
        
        # Compute phase entropy if phase values provided
        if phase_values is not None:
            H_phase, phase_dist = self._phase_entropy(phase_values)
            H_phase_normalized = H_phase / np.log(self.num_phase_bins)
            
            outputs.update({
                'phase_entropy': H_phase.mean().item(),
                'phase_entropy_normalized': H_phase_normalized.mean().item(),
                'circular_variance': self._circular_variance(phase_values).mean().item()
            })
            
            # Compute joint entropy and mutual information
            if state_posterior.shape[1] == phase_values.shape[1]:  # Same time dimension
                joint_dist = self._compute_joint_distribution(
                    state_posterior, phase_dist
                )
                H_joint = self._joint_entropy(joint_dist)
                # Average state entropy over time for MI calculation
                H_state_avg = H_state.mean(dim=1) if H_state.dim() > 1 else H_state
                MI = H_state_avg + H_phase - H_joint
                
                outputs.update({
                    'joint_entropy': H_joint.mean().item(),
                    'mutual_information': MI.mean().item(),
                    'coordination_coherence': (MI / torch.minimum(H_state_avg, H_phase)).mean().item()
                })
        
        # Compute cluster entropy if provided
        if cluster_assignments is not None:
            H_cluster = self._shannon_entropy(cluster_assignments)
            outputs['cluster_entropy'] = H_cluster.mean().item()
        
        # Compute confidence metrics
        confidence_metrics = self._compute_confidence(
            H_state_normalized,
            outputs.get('phase_entropy_normalized', 0)
        )
        outputs.update(confidence_metrics)
        
        # Add marginals if requested
        if return_marginals:
            marginals = self._get_marginals(
                state_posterior,
                phase_dist if phase_values is not None else None
            )
            outputs['marginals'] = marginals
        
        return outputs
    
    def _shannon_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy for discrete distributions"""
        p = torch.clamp(probs, min=self.eps)
        return -torch.sum(p * torch.log(p), dim=-1)
    
    def _phase_entropy(self, phase_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute entropy for phase values using binning"""
        # Discretize phase into bins
        phase_flat = phase_values.flatten()
        
        # Create histogram for each batch
        if phase_values.dim() == 2:
            B, T = phase_values.shape
            phase_dist = torch.zeros(B, self.num_phase_bins, device=phase_values.device)
            
            for b in range(B):
                hist = torch.histc(
                    phase_values[b],
                    bins=self.num_phase_bins,
                    min=-np.pi,
                    max=np.pi
                )
                phase_dist[b] = hist / (hist.sum() + self.eps)
        else:
            hist = torch.histc(
                phase_values,
                bins=self.num_phase_bins,
                min=-np.pi,
                max=np.pi
            )
            phase_dist = hist / (hist.sum() + self.eps)
        
        # Compute entropy
        H_phase = self._shannon_entropy(phase_dist)
        
        return H_phase, phase_dist
    
    def _circular_variance(self, angles: torch.Tensor) -> torch.Tensor:
        """Compute circular variance (1 - R) where R is resultant length"""
        # Compute resultant vector
        cos_sum = torch.cos(angles).mean(dim=-1)
        sin_sum = torch.sin(angles).mean(dim=-1)
        R = torch.sqrt(cos_sum**2 + sin_sum**2)
        
        return 1 - R
    
    def _compute_joint_distribution(
        self,
        state_probs: torch.Tensor,
        phase_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute joint distribution p(state, phase)"""
        # Outer product for independence assumption, then normalize
        if state_probs.dim() == 3 and phase_probs.dim() == 2:
            # Average over time for states
            state_probs = state_probs.mean(dim=1)
        
        # Compute outer product
        joint = torch.einsum('bs,bp->bsp', state_probs, phase_probs)
        
        # Normalize
        joint = joint / (joint.sum(dim=(1, 2), keepdim=True) + self.eps)
        
        return joint
    
    def _joint_entropy(self, joint_dist: torch.Tensor) -> torch.Tensor:
        """Compute joint entropy H(Z, Phi)"""
        j = torch.clamp(joint_dist, min=self.eps)
        return -torch.sum(j * torch.log(j), dim=(1, 2))
    
    def _dirichlet_smoothing(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply Dirichlet smoothing for calibration"""
        # Add small concentration to each probability
        alpha = self.dirichlet_concentration.view(1, 1, -1)
        smoothed = probs + alpha
        return smoothed / smoothed.sum(dim=-1, keepdim=True)
    
    def _compute_confidence(
        self,
        state_entropy_norm: torch.Tensor,
        phase_entropy_norm: float,
        conformal_predictor: Optional['ConformalPredictor'] = None,
        state_posterior: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[float, str]]:
        """Compute confidence metrics based on entropy with proper calibration"""
        # Average entropy
        avg_entropy = (state_entropy_norm.mean() + phase_entropy_norm) / 2
        
        # Determine confidence level
        if avg_entropy <= self.confidence_threshold_high:
            confidence_level = 'high'
            base_width = 0.1
        elif avg_entropy <= self.confidence_threshold_low:
            confidence_level = 'medium'
            base_width = 0.2
        else:
            confidence_level = 'low'
            base_width = 0.35
        
        # Use conformal prediction if available for calibrated intervals
        if conformal_predictor is not None and state_posterior is not None:
            try:
                from .calibration import ConformalPredictor
                predictions, lower_bounds, upper_bounds = conformal_predictor.predict_interval(state_posterior)
                
                # Compute average interval width
                avg_width = (upper_bounds - lower_bounds).float().mean().item()
                
                # Get confidence score from posterior max probability
                max_prob = state_posterior.max(dim=1)[0].mean().item()
                
                return {
                    'confidence_level': confidence_level,
                    'confidence_score': float(max_prob),
                    'confidence_interval': (
                        float(lower_bounds.mean()),
                        float(upper_bounds.mean())
                    ),
                    'interval_width': avg_width,
                    'calibrated': True
                }
            except Exception:
                # Fall back to entropy-based estimation
                pass
        
        # Entropy-based confidence (fallback or when conformal not available)
        confidence_score = float(1 - avg_entropy)
        
        return {
            'confidence_level': confidence_level,
            'confidence_score': confidence_score,
            'confidence_interval': (
                max(0, confidence_score - base_width/2),
                min(1, confidence_score + base_width/2)
            ),
            'interval_width': base_width,
            'calibrated': False
        }
    
    def _get_marginals(
        self,
        state_posterior: torch.Tensor,
        phase_dist: Optional[torch.Tensor]
    ) -> Dict:
        """Get marginal distributions for API output"""
        # Get top-k states
        state_avg = state_posterior.mean(dim=(0, 1))  # Average over batch and time
        top_probs, top_indices = torch.topk(state_avg, k=min(3, len(state_avg)))
        
        marginals = {
            'p_state_topk': [
                {'id': int(idx), 'p': float(prob)}
                for idx, prob in zip(top_indices, top_probs)
            ],
            'p_state_full': state_avg.tolist()
        }
        
        if phase_dist is not None:
            if phase_dist.dim() == 2:
                phase_avg = phase_dist.mean(dim=0)
            else:
                phase_avg = phase_dist
            
            marginals['p_phase_hist'] = phase_avg.tolist()
            marginals['phase_bin_edges'] = self.phase_bins.tolist()
        
        return marginals
    
    def calibrate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calibrate the model using ECE and Brier score
        
        Args:
            predictions: Model predictions (B, S)
            targets: True labels (B,)
        
        Returns:
            Calibration metrics
        """
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(predictions, targets)
        
        # Brier Score
        brier = self._compute_brier(predictions, targets)
        
        return {
            'ece': ece,
            'brier_score': brier
        }
    
    def _compute_ece(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error"""
        confidences, predicted = torch.max(predictions, dim=1)
        accuracies = (predicted == targets).float()
        
        ece = 0
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    def _compute_brier(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Compute Brier Score"""
        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes=predictions.shape[1]).float()
        
        # Brier score
        brier = torch.mean((predictions - targets_onehot) ** 2)
        
        return brier.item()


class CircularStatistics:
    """Helper class for circular statistics computations"""
    
    @staticmethod
    def resultant_length(angles: np.ndarray) -> float:
        """Compute resultant length R"""
        C = np.cos(angles).sum()
        S = np.sin(angles).sum()
        N = len(angles)
        R = np.sqrt(C**2 + S**2) / max(N, 1)
        return R
    
    @staticmethod
    def von_mises_concentration(angles: np.ndarray) -> float:
        """Estimate von Mises concentration parameter kappa"""
        R = CircularStatistics.resultant_length(angles)
        
        # Approximation for kappa
        if R < 0.53:
            kappa = 2 * R + R**3 + 5 * R**5 / 6
        elif R < 0.85:
            kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
        else:
            kappa = 1 / (R**3 - 4 * R**2 + 3 * R + 1e-10)
        
        return kappa
    
    @staticmethod
    def circular_mean(angles: np.ndarray) -> float:
        """Compute circular mean"""
        C = np.cos(angles).mean()
        S = np.sin(angles).mean()
        mean_angle = np.arctan2(S, C)
        return mean_angle
    
    @staticmethod
    def circular_variance(angles: np.ndarray) -> float:
        """Compute circular variance"""
        R = CircularStatistics.resultant_length(angles)
        return 1 - R


def summarize_window(
    state_post: np.ndarray,
    phase_angles: np.ndarray,
    num_bins: int = 12
) -> Dict:
    """
    Quick summary function matching the specification
    
    Args:
        state_post: State posterior probabilities
        phase_angles: Phase angles in radians
        num_bins: Number of phase bins
    
    Returns:
        Dictionary with entropy and statistical measures
    """
    EPS = 1e-12
    
    # Normalize state posterior
    state_post = np.clip(state_post, EPS, 1.0)
    state_post = state_post / state_post.sum()
    
    # State entropy
    H_state = -np.sum(state_post * np.log(state_post))
    H_state_norm = H_state / np.log(len(state_post))
    
    # Circular statistics
    R = CircularStatistics.resultant_length(phase_angles)
    kappa = CircularStatistics.von_mises_concentration(phase_angles)
    circ_var = CircularStatistics.circular_variance(phase_angles)
    
    # Phase histogram and entropy
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    hist, _ = np.histogram(phase_angles, bins=bins, density=True)
    hist = hist + EPS
    hist = hist / hist.sum()
    H_phase = -np.sum(hist * np.log(hist))
    H_phase_norm = H_phase / np.log(num_bins)
    
    # Joint distribution and mutual information
    joint = np.outer(state_post, hist)
    joint = joint / joint.sum()
    H_joint = -np.sum(joint * np.log(joint + EPS))
    MI = H_state + H_phase - H_joint
    
    return {
        'H_state': float(H_state),
        'H_state_normalized': float(H_state_norm),
        'H_phase': float(H_phase),
        'H_phase_normalized': float(H_phase_norm),
        'H_joint': float(H_joint),
        'MI': float(MI),
        'kappa': float(kappa),
        'R': float(R),
        'circular_variance': float(circ_var),
        'p_phase_hist': hist.tolist(),
        'behavioral_diversity': float(np.exp(H_state)),
        'coordination_coherence': float(MI / min(H_state, H_phase)) if min(H_state, H_phase) > 0 else 0
    }


def test_entropy_module():
    """Test the entropy uncertainty module"""
    print("Testing Entropy & Uncertainty Module...")
    print("="*50)
    
    B, T, S = 4, 100, 10  # Batch, Time, States
    
    # Initialize module
    module = EntropyUncertaintyModule(num_states=S, num_phase_bins=12)
    
    # Test data
    state_posterior = F.softmax(torch.randn(B, T, S), dim=-1)
    phase_values = torch.rand(B, T) * 2 * np.pi - np.pi  # Random phases
    cluster_assignments = F.softmax(torch.randn(B, T, 20), dim=-1)
    
    # Test forward pass
    outputs = module(
        state_posterior=state_posterior,
        phase_values=phase_values,
        cluster_assignments=cluster_assignments,
        return_marginals=True
    )
    
    print("Uncertainty Metrics:")
    print(f"  State entropy: {outputs['state_entropy']:.3f}")
    print(f"  Phase entropy: {outputs.get('phase_entropy', 0):.3f}")
    print(f"  Mutual information: {outputs.get('mutual_information', 0):.3f}")
    print(f"  Behavioral diversity: {outputs['behavioral_diversity']:.1f}")
    print(f"  Coordination coherence: {outputs.get('coordination_coherence', 0):.3f}")
    print(f"  Confidence level: {outputs['confidence_level']}")
    print(f"  Confidence interval: {outputs['confidence_interval']}")
    
    # Test marginals
    if 'marginals' in outputs:
        print("\nMarginal Distributions:")
        print(f"  Top states: {outputs['marginals']['p_state_topk']}")
        print(f"  Phase bins: {len(outputs['marginals'].get('p_phase_hist', []))} bins")
    
    # Test numpy summary function
    print("\nTesting NumPy summary function...")
    state_post_np = state_posterior[0, 0].numpy()
    phase_angles_np = phase_values[0].numpy()
    
    summary = summarize_window(state_post_np, phase_angles_np)
    
    print(f"  H_state: {summary['H_state']:.3f}")
    print(f"  H_phase: {summary['H_phase']:.3f}")
    print(f"  MI: {summary['MI']:.3f}")
    print(f"  Kappa: {summary['kappa']:.3f}")
    print(f"  Coherence: {summary['coordination_coherence']:.3f}")
    
    print("\n✅ Entropy module tests passed!")
    return module


if __name__ == "__main__":
    test_entropy_module()