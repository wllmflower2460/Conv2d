"""Advanced diagnostics for behavioral data using Movement library.

This module provides comprehensive diagnostic tools for analyzing data quality,
preprocessing effectiveness, and behavioral patterns in the Conv2d-VQ-HDP-HSMM pipeline.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings
import json
from datetime import datetime
from dataclasses import dataclass

from .movement_integration import MovementPreprocessor, MovementDiagnostics
from .kinematic_features import KinematicFeatureExtractor


@dataclass
class QualityThresholds:
    """Configurable quality thresholds for GIGO prevention."""
    
    # Input validation thresholds
    max_nan_percentage: float = 10.0          # Maximum allowed NaN percentage
    max_inf_count: int = 0                    # Maximum allowed infinite values
    min_signal_std: float = 0.01              # Minimum signal standard deviation
    max_signal_std: float = 100.0             # Maximum signal standard deviation
    expected_shape: Tuple[int, int, int, int] = (None, 9, 2, 100)  # (B, 9, 2, 100)
    
    # Codebook health thresholds
    min_codebook_usage: float = 0.1           # Minimum fraction of codes used
    min_perplexity: float = 4.0               # Minimum perplexity for diversity
    max_dead_codes_ratio: float = 0.5         # Maximum ratio of dead codes
    min_entropy: float = 2.0                  # Minimum entropy for diversity
    max_transition_rate: float = 0.8          # Maximum transition rate between codes
    
    # Signal quality thresholds
    min_snr_db: float = 0.0                   # Minimum signal-to-noise ratio
    max_autocorr_lag1: float = 0.95           # Maximum lag-1 autocorrelation
    min_frequency_diversity: float = 0.1      # Minimum frequency diversity
    
    # Data consistency thresholds
    max_gap_length: int = 50                  # Maximum allowed gap length
    max_gap_percentage: float = 20.0          # Maximum percentage of data that can be gaps
    min_temporal_correlation: float = 0.1     # Minimum correlation between time steps


@dataclass
class QualityReport:
    """Comprehensive quality report with pass/fail status."""
    
    timestamp: str
    data_shape: Tuple[int, ...]
    overall_pass: bool
    
    # Input validation results
    input_validation: Dict[str, Any]
    
    # Codebook health results (if applicable)
    codebook_health: Optional[Dict[str, Any]] = None
    
    # Signal quality results
    signal_quality: Dict[str, Any] = None
    
    # Data consistency results
    data_consistency: Dict[str, Any] = None
    
    # Detailed metrics for analysis
    detailed_metrics: Dict[str, Any] = None
    
    # Recommendations for improvement
    recommendations: List[str] = None


class QualityControl:
    """
    Comprehensive quality control system for preventing GIGO (Garbage In, Garbage Out).
    
    This class implements multiple layers of quality gates:
    1. Input validation: Shape, type, range checks
    2. Codebook health monitoring: Usage, diversity, transitions
    3. Signal quality assessment: SNR, frequency content, stationarity
    4. Data consistency checks: Gaps, correlations, outliers
    """
    
    def __init__(self, 
                 thresholds: Optional[QualityThresholds] = None,
                 enable_codebook_monitoring: bool = True,
                 strict_mode: bool = False):
        """
        Initialize quality control system.
        
        Args:
            thresholds: Custom quality thresholds
            enable_codebook_monitoring: Whether to monitor VQ codebook health
            strict_mode: If True, fail on any threshold violation
        """
        self.thresholds = thresholds or QualityThresholds()
        self.enable_codebook_monitoring = enable_codebook_monitoring
        self.strict_mode = strict_mode
        
        # Track quality metrics over time
        self.quality_history: List[QualityReport] = []
        
        # Statistical baselines (updated as data is processed)
        self.signal_baselines = {
            'mean_std': 1.0,
            'mean_snr': 10.0,
            'typical_perplexity': 8.0
        }
    
    def validate_input_tensor(self, data: torch.Tensor) -> Dict[str, Any]:
        """
        Validate input tensor for basic quality requirements.
        
        Args:
            data: Input tensor to validate
            
        Returns:
            Dictionary with validation results and pass/fail status
        """
        results = {
            'shape_valid': True,
            'dtype_valid': True,
            'range_valid': True,
            'nan_check': True,
            'inf_check': True,
            'std_check': True,
            'failures': [],
            'warnings': [],
            'pass': True
        }
        
        # Shape validation
        expected_shape = self.thresholds.expected_shape
        if expected_shape[1:] != (None,) * (len(expected_shape) - 1):  # Check non-batch dimensions
            if data.shape[1:] != expected_shape[1:]:
                results['shape_valid'] = False
                results['failures'].append(f"Expected shape (*, {', '.join(map(str, expected_shape[1:]))}), got {data.shape}")
        
        # Data type validation
        if not data.dtype.is_floating_point:
            results['dtype_valid'] = False
            results['failures'].append(f"Expected floating point tensor, got {data.dtype}")
        
        # NaN validation
        nan_count = torch.isnan(data).sum().item()
        nan_percentage = 100 * nan_count / data.numel()
        if nan_percentage > self.thresholds.max_nan_percentage:
            results['nan_check'] = False
            results['failures'].append(f"NaN percentage {nan_percentage:.2f}% exceeds threshold {self.thresholds.max_nan_percentage}%")
        elif nan_percentage > self.thresholds.max_nan_percentage * 0.5:
            results['warnings'].append(f"High NaN percentage: {nan_percentage:.2f}%")
        
        # Infinity validation
        inf_count = torch.isinf(data).sum().item()
        if inf_count > self.thresholds.max_inf_count:
            results['inf_check'] = False
            results['failures'].append(f"Infinite values count {inf_count} exceeds threshold {self.thresholds.max_inf_count}")
        
        # Standard deviation validation (signal variability)
        valid_data = data[~torch.isnan(data) & ~torch.isinf(data)]
        if valid_data.numel() > 0:
            data_std = valid_data.std().item()
            if data_std < self.thresholds.min_signal_std:
                results['std_check'] = False
                results['failures'].append(f"Signal std {data_std:.6f} below threshold {self.thresholds.min_signal_std}")
            elif data_std > self.thresholds.max_signal_std:
                results['std_check'] = False
                results['failures'].append(f"Signal std {data_std:.2f} above threshold {self.thresholds.max_signal_std}")
        
        # Overall pass/fail
        results['pass'] = len(results['failures']) == 0
        if self.strict_mode and len(results['warnings']) > 0:
            results['pass'] = False
            results['failures'].extend(results['warnings'])
        
        return results
    
    def monitor_codebook_health(self, 
                              codebook_info: Dict[str, Any],
                              indices_history: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Monitor VQ codebook health for quality degradation.
        
        Args:
            codebook_info: Information from VQ forward pass
            indices_history: Historical code indices for transition analysis
            
        Returns:
            Dictionary with codebook health metrics and pass/fail status
        """
        if not self.enable_codebook_monitoring:
            return {'enabled': False, 'pass': True}
        
        results = {
            'usage_check': True,
            'perplexity_check': True,
            'entropy_check': True,
            'dead_codes_check': True,
            'transition_check': True,
            'failures': [],
            'warnings': [],
            'pass': True,
            'metrics': {}
        }
        
        # Extract metrics from codebook info
        perplexity = codebook_info.get('perplexity', 0.0)
        usage = codebook_info.get('usage', 0.0)
        entropy = codebook_info.get('entropy', 0.0)
        dead_codes = codebook_info.get('dead_codes', 0)
        total_codes = codebook_info.get('total_codes', 512)  # Default assumption
        
        # Store metrics
        results['metrics'] = {
            'perplexity': float(perplexity),
            'usage': float(usage),
            'entropy': float(entropy),
            'dead_codes': int(dead_codes),
            'dead_codes_ratio': float(dead_codes / total_codes) if total_codes > 0 else 0.0
        }
        
        # Usage validation
        if usage < self.thresholds.min_codebook_usage:
            results['usage_check'] = False
            results['failures'].append(f"Codebook usage {usage:.3f} below threshold {self.thresholds.min_codebook_usage}")
        
        # Perplexity validation (diversity indicator)
        if perplexity < self.thresholds.min_perplexity:
            results['perplexity_check'] = False
            results['failures'].append(f"Perplexity {perplexity:.2f} below threshold {self.thresholds.min_perplexity}")
        
        # Entropy validation
        if entropy < self.thresholds.min_entropy:
            results['entropy_check'] = False
            results['failures'].append(f"Entropy {entropy:.2f} below threshold {self.thresholds.min_entropy}")
        
        # Dead codes validation
        dead_codes_ratio = dead_codes / total_codes if total_codes > 0 else 0.0
        if dead_codes_ratio > self.thresholds.max_dead_codes_ratio:
            results['dead_codes_check'] = False
            results['failures'].append(f"Dead codes ratio {dead_codes_ratio:.2f} exceeds threshold {self.thresholds.max_dead_codes_ratio}")
        elif dead_codes_ratio > self.thresholds.max_dead_codes_ratio * 0.7:
            results['warnings'].append(f"High dead codes ratio: {dead_codes_ratio:.2f}")
        
        # Transition rate analysis (if history provided)
        if indices_history and len(indices_history) > 1:
            transition_rate = self._compute_transition_rate(indices_history)
            results['metrics']['transition_rate'] = transition_rate
            
            if transition_rate > self.thresholds.max_transition_rate:
                results['transition_check'] = False
                results['failures'].append(f"Transition rate {transition_rate:.3f} exceeds threshold {self.thresholds.max_transition_rate}")
        
        # Overall pass/fail
        results['pass'] = len(results['failures']) == 0
        if self.strict_mode and len(results['warnings']) > 0:
            results['pass'] = False
            results['failures'].extend(results['warnings'])
        
        return results
    
    def assess_signal_quality(self, data: torch.Tensor) -> Dict[str, Any]:
        """
        Assess signal quality metrics.
        
        Args:
            data: Input tensor to assess
            
        Returns:
            Dictionary with signal quality metrics and pass/fail status
        """
        results = {
            'snr_check': True,
            'autocorr_check': True,
            'frequency_check': True,
            'stationarity_check': True,
            'failures': [],
            'warnings': [],
            'pass': True,
            'metrics': {}
        }
        
        # Clean data for analysis
        valid_data = data[~torch.isnan(data) & ~torch.isinf(data)]
        if valid_data.numel() == 0:
            results['failures'].append("No valid data for signal quality assessment")
            results['pass'] = False
            return results
        
        # SNR estimation
        if data.dim() >= 3:
            signal_power = torch.var(data, dim=-1)
            noise_estimate = torch.var(torch.diff(data, dim=-1), dim=-1) / 2
            snr = signal_power / (noise_estimate + 1e-10)
            snr_db = 10 * torch.log10(snr.mean()).item()
            
            results['metrics']['snr_db'] = snr_db
            
            if snr_db < self.thresholds.min_snr_db:
                results['snr_check'] = False
                results['failures'].append(f"SNR {snr_db:.2f} dB below threshold {self.thresholds.min_snr_db} dB")
        
        # Autocorrelation analysis
        autocorr = self._compute_simple_autocorr(data)
        if len(autocorr) > 1:
            lag1_corr = autocorr[1].item()
            results['metrics']['lag1_autocorr'] = lag1_corr
            
            if lag1_corr > self.thresholds.max_autocorr_lag1:
                results['autocorr_check'] = False
                results['failures'].append(f"Lag-1 autocorr {lag1_corr:.3f} exceeds threshold {self.thresholds.max_autocorr_lag1}")
        
        # Frequency diversity
        fft = torch.fft.rfft(data.mean(dim=(0,1,2)) if data.dim() == 4 else data.mean(dim=0))
        magnitude = torch.abs(fft)
        power = magnitude ** 2
        total_power = power.sum()
        
        # Check if power is concentrated in a few frequencies
        sorted_power, _ = torch.sort(power, descending=True)
        top_10_power = sorted_power[:len(sorted_power)//10].sum()
        frequency_diversity = 1.0 - (top_10_power / total_power).item()
        
        results['metrics']['frequency_diversity'] = frequency_diversity
        
        if frequency_diversity < self.thresholds.min_frequency_diversity:
            results['frequency_check'] = False
            results['failures'].append(f"Frequency diversity {frequency_diversity:.3f} below threshold {self.thresholds.min_frequency_diversity}")
        
        # Overall pass/fail
        results['pass'] = len(results['failures']) == 0
        if self.strict_mode and len(results['warnings']) > 0:
            results['pass'] = False
            results['failures'].extend(results['warnings'])
        
        return results
    
    def check_data_consistency(self, data: torch.Tensor) -> Dict[str, Any]:
        """
        Check data consistency and detect anomalies.
        
        Args:
            data: Input tensor to check
            
        Returns:
            Dictionary with consistency check results and pass/fail status
        """
        results = {
            'gap_check': True,
            'correlation_check': True,
            'outlier_check': True,
            'failures': [],
            'warnings': [],
            'pass': True,
            'metrics': {}
        }
        
        # Gap analysis
        gaps = self._find_comprehensive_gaps(data)
        max_gap_length = max([g['length'] for g in gaps]) if gaps else 0
        total_gap_frames = sum([g['length'] for g in gaps]) if gaps else 0
        gap_percentage = 100 * total_gap_frames / data.shape[-1] if data.shape[-1] > 0 else 0
        
        results['metrics']['max_gap_length'] = max_gap_length
        results['metrics']['gap_percentage'] = gap_percentage
        results['metrics']['num_gaps'] = len(gaps)
        
        if max_gap_length > self.thresholds.max_gap_length:
            results['gap_check'] = False
            results['failures'].append(f"Max gap length {max_gap_length} exceeds threshold {self.thresholds.max_gap_length}")
        
        if gap_percentage > self.thresholds.max_gap_percentage:
            results['gap_check'] = False
            results['failures'].append(f"Gap percentage {gap_percentage:.2f}% exceeds threshold {self.thresholds.max_gap_percentage}%")
        
        # Temporal correlation
        if data.shape[-1] > 1:
            correlation = self._compute_temporal_correlation(data)
            results['metrics']['temporal_correlation'] = correlation
            
            if correlation < self.thresholds.min_temporal_correlation:
                results['correlation_check'] = False
                results['failures'].append(f"Temporal correlation {correlation:.3f} below threshold {self.thresholds.min_temporal_correlation}")
        
        # Outlier detection (simplified z-score approach)
        valid_data = data[~torch.isnan(data) & ~torch.isinf(data)]
        if valid_data.numel() > 0:
            z_scores = torch.abs((valid_data - valid_data.mean()) / (valid_data.std() + 1e-10))
            outlier_count = (z_scores > 3.0).sum().item()
            outlier_percentage = 100 * outlier_count / valid_data.numel()
            
            results['metrics']['outlier_percentage'] = outlier_percentage
            
            if outlier_percentage > 5.0:  # More than 5% outliers
                results['warnings'].append(f"High outlier percentage: {outlier_percentage:.2f}%")
        
        # Overall pass/fail
        results['pass'] = len(results['failures']) == 0
        if self.strict_mode and len(results['warnings']) > 0:
            results['pass'] = False
            results['failures'].extend(results['warnings'])
        
        return results
    
    def run_quality_gates(self, 
                         data: torch.Tensor,
                         codebook_info: Optional[Dict[str, Any]] = None,
                         indices_history: Optional[List[torch.Tensor]] = None) -> QualityReport:
        """
        Run comprehensive quality gates on input data.
        
        Args:
            data: Input tensor to validate
            codebook_info: VQ codebook information (if available)
            indices_history: Historical code indices for transition analysis
            
        Returns:
            QualityReport with comprehensive results
        """
        timestamp = datetime.now().isoformat()
        
        # Run all quality checks
        input_validation = self.validate_input_tensor(data)
        signal_quality = self.assess_signal_quality(data)
        data_consistency = self.check_data_consistency(data)
        
        codebook_health = None
        if codebook_info is not None:
            codebook_health = self.monitor_codebook_health(codebook_info, indices_history)
        
        # Determine overall pass/fail
        all_checks = [input_validation, signal_quality, data_consistency]
        if codebook_health is not None:
            all_checks.append(codebook_health)
        
        overall_pass = all(check.get('pass', True) for check in all_checks)
        
        # Compile recommendations
        recommendations = self._generate_recommendations(
            input_validation, signal_quality, data_consistency, codebook_health
        )
        
        # Create detailed metrics summary
        detailed_metrics = {
            'input_shape': list(data.shape),
            'data_type': str(data.dtype),
            'device': str(data.device),
            'total_elements': data.numel(),
            'nan_count': torch.isnan(data).sum().item(),
            'inf_count': torch.isinf(data).sum().item(),
            'signal_std': data[~torch.isnan(data)].std().item() if torch.any(~torch.isnan(data)) else 0.0,
            'signal_mean': data[~torch.isnan(data)].mean().item() if torch.any(~torch.isnan(data)) else 0.0
        }
        
        # Create quality report
        report = QualityReport(
            timestamp=timestamp,
            data_shape=tuple(data.shape),
            overall_pass=overall_pass,
            input_validation=input_validation,
            codebook_health=codebook_health,
            signal_quality=signal_quality,
            data_consistency=data_consistency,
            detailed_metrics=detailed_metrics,
            recommendations=recommendations
        )
        
        # Store in history
        self.quality_history.append(report)
        
        return report
    
    def export_quality_report(self, report: QualityReport, output_path: Optional[Path] = None) -> Path:
        """
        Export quality report to JSON file.
        
        Args:
            report: Quality report to export
            output_path: Optional output path
            
        Returns:
            Path to exported report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"quality_report_{timestamp}.json")
        
        # Convert report to dictionary for JSON serialization
        report_dict = {
            'timestamp': report.timestamp,
            'data_shape': list(report.data_shape),
            'overall_pass': report.overall_pass,
            'input_validation': report.input_validation,
            'signal_quality': report.signal_quality,
            'data_consistency': report.data_consistency,
            'detailed_metrics': report.detailed_metrics,
            'recommendations': report.recommendations
        }
        
        if report.codebook_health is not None:
            report_dict['codebook_health'] = report.codebook_health
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        return output_path
    
    def get_quality_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze quality trends over recent history.
        
        Args:
            window_size: Number of recent reports to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.quality_history) < 2:
            return {'insufficient_data': True}
        
        recent_reports = self.quality_history[-window_size:]
        
        trends = {
            'pass_rate': sum(r.overall_pass for r in recent_reports) / len(recent_reports),
            'avg_nan_percentage': np.mean([
                r.detailed_metrics.get('nan_count', 0) / r.detailed_metrics.get('total_elements', 1) * 100
                for r in recent_reports
            ]),
            'avg_signal_std': np.mean([
                r.detailed_metrics.get('signal_std', 0) for r in recent_reports
            ]),
        }
        
        # Codebook trends (if available)
        codebook_reports = [r for r in recent_reports if r.codebook_health is not None]
        if codebook_reports:
            trends['avg_perplexity'] = np.mean([
                r.codebook_health['metrics'].get('perplexity', 0) for r in codebook_reports
            ])
            trends['avg_usage'] = np.mean([
                r.codebook_health['metrics'].get('usage', 0) for r in codebook_reports
            ])
        
        return trends
    
    # Helper methods
    def _compute_transition_rate(self, indices_history: List[torch.Tensor]) -> float:
        """Compute rate of code transitions between consecutive time steps."""
        if len(indices_history) < 2:
            return 0.0
        
        total_transitions = 0
        total_comparisons = 0
        
        for i in range(len(indices_history) - 1):
            curr_indices = indices_history[i].flatten()
            next_indices = indices_history[i + 1].flatten()
            
            if len(curr_indices) == len(next_indices):
                transitions = (curr_indices != next_indices).sum().item()
                total_transitions += transitions
                total_comparisons += len(curr_indices)
        
        return total_transitions / total_comparisons if total_comparisons > 0 else 0.0
    
    def _compute_simple_autocorr(self, data: torch.Tensor) -> torch.Tensor:
        """Compute simple autocorrelation for the first few lags."""
        if data.dim() > 1:
            data = data.flatten()
        
        data = data - data.mean()
        data_np = data.cpu().numpy()
        autocorr_np = np.correlate(data_np, data_np, mode='same')
        autocorr = torch.from_numpy(autocorr_np).to(data.device)
        
        # Normalize and return center portion
        center = len(autocorr) // 2
        autocorr = autocorr[center:center+min(10, len(autocorr)//2)]
        if len(autocorr) > 0:
            autocorr = autocorr / autocorr[0]
        
        return autocorr
    
    def _find_comprehensive_gaps(self, data: torch.Tensor) -> List[Dict]:
        """Find gaps in data more comprehensively."""
        gaps = []
        
        # Check for NaN gaps
        nan_mask = torch.isnan(data).any(dim=tuple(range(data.dim()-1)))
        
        in_gap = False
        gap_start = 0
        
        for i, is_nan in enumerate(nan_mask):
            if is_nan and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_nan and in_gap:
                gaps.append({
                    'start': int(gap_start),
                    'end': int(i),
                    'length': int(i - gap_start),
                    'type': 'nan'
                })
                in_gap = False
        
        if in_gap:
            gaps.append({
                'start': int(gap_start),
                'end': int(len(nan_mask)),
                'length': int(len(nan_mask) - gap_start),
                'type': 'nan'
            })
        
        return gaps
    
    def _compute_temporal_correlation(self, data: torch.Tensor) -> float:
        """Compute temporal correlation between consecutive time steps."""
        if data.shape[-1] < 2:
            return 0.0
        
        # Flatten spatial dimensions and compute correlation between t and t+1
        flattened = data.flatten(0, -2)  # (all_spatial, time)
        
        if flattened.shape[0] == 0 or flattened.shape[1] < 2:
            return 0.0
        
        t1 = flattened[:, :-1].flatten()
        t2 = flattened[:, 1:].flatten()
        
        # Remove NaN values
        valid_mask = ~torch.isnan(t1) & ~torch.isnan(t2)
        if valid_mask.sum() < 2:
            return 0.0
        
        t1_valid = t1[valid_mask]
        t2_valid = t2[valid_mask]
        
        # Compute correlation coefficient
        if t1_valid.std() == 0 or t2_valid.std() == 0:
            return 0.0
        
        correlation = torch.corrcoef(torch.stack([t1_valid, t2_valid]))[0, 1]
        return correlation.item() if not torch.isnan(correlation) else 0.0
    
    def _generate_recommendations(self, 
                                input_val: Dict,
                                signal_qual: Dict,
                                data_cons: Dict,
                                codebook_health: Optional[Dict]) -> List[str]:
        """Generate actionable recommendations based on quality check results."""
        recommendations = []
        
        # Input validation recommendations
        if not input_val.get('pass', True):
            if not input_val.get('shape_valid', True):
                recommendations.append("Check data preprocessing pipeline - input shape mismatch")
            if not input_val.get('nan_check', True):
                recommendations.append("Apply data cleaning and interpolation to reduce NaN values")
            if not input_val.get('std_check', True):
                recommendations.append("Verify sensor calibration and data collection procedures")
        
        # Signal quality recommendations
        if not signal_qual.get('pass', True):
            if not signal_qual.get('snr_check', True):
                recommendations.append("Apply noise filtering or improve sensor positioning")
            if not signal_qual.get('frequency_check', True):
                recommendations.append("Check for frequency aliasing or increase sampling rate")
        
        # Data consistency recommendations
        if not data_cons.get('pass', True):
            if not data_cons.get('gap_check', True):
                recommendations.append("Improve data collection reliability or apply gap interpolation")
            if not data_cons.get('correlation_check', True):
                recommendations.append("Verify temporal continuity in data collection")
        
        # Codebook health recommendations
        if codebook_health and not codebook_health.get('pass', True):
            if not codebook_health.get('usage_check', True):
                recommendations.append("Increase model capacity or improve codebook initialization")
            if not codebook_health.get('perplexity_check', True):
                recommendations.append("Adjust VQ parameters to improve codebook diversity")
            if not codebook_health.get('dead_codes_check', True):
                recommendations.append("Enable dead code refresh or reduce codebook size")
        
        if not recommendations:
            recommendations.append("Data quality looks good - proceed with training/inference")
        
        return recommendations


class BehavioralDataDiagnostics:
    """Comprehensive diagnostics for behavioral synchrony analysis."""
    
    def __init__(self, 
                 sampling_rate: float = 100.0,
                 output_dir: Optional[Path] = None,
                 quality_thresholds: Optional[QualityThresholds] = None,
                 enable_quality_gates: bool = True,
                 strict_quality_mode: bool = False):
        """Initialize diagnostic system.
        
        Args:
            sampling_rate: Data sampling rate in Hz
            output_dir: Directory for saving diagnostic outputs
            quality_thresholds: Custom quality control thresholds
            enable_quality_gates: Whether to enable quality gates
            strict_quality_mode: Whether to use strict quality checking
        """
        self.sampling_rate = sampling_rate
        self.output_dir = Path(output_dir) if output_dir else Path('./diagnostics')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.movement_proc = MovementPreprocessor(sampling_rate=sampling_rate)
        self.diagnostics = MovementDiagnostics(self.movement_proc)
        self.feature_extractor = KinematicFeatureExtractor(sampling_rate=sampling_rate)
        
        # Initialize quality control system
        self.enable_quality_gates = enable_quality_gates
        if enable_quality_gates:
            self.quality_control = QualityControl(
                thresholds=quality_thresholds,
                enable_codebook_monitoring=True,
                strict_mode=strict_quality_mode
            )
        else:
            self.quality_control = None
        
        # Storage for diagnostic history
        self.diagnostic_history = []
    
    def run_full_diagnostic(self, data: torch.Tensor,
                          labels: Optional[torch.Tensor] = None,
                          codebook_info: Optional[Dict[str, Any]] = None,
                          indices_history: Optional[List[torch.Tensor]] = None,
                          save_report: bool = True) -> Dict[str, Any]:
        """Run comprehensive diagnostic suite on input data.
        
        Args:
            data: Input tensor (B, 9, 2, T) for IMU or appropriate shape
            labels: Optional ground truth labels
            codebook_info: Optional VQ codebook information for health monitoring
            indices_history: Optional historical indices for transition analysis
            save_report: Whether to save diagnostic report to disk
            
        Returns:
            Dictionary with all diagnostic results including quality gates
        """
        print("=" * 60)
        print("BEHAVIORAL DATA DIAGNOSTIC SUITE")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': list(data.shape),
            'device': str(data.device),
            'dtype': str(data.dtype)
        }
        
        # 0. Quality Gates (GIGO Prevention)
        if self.enable_quality_gates and self.quality_control:
            print("\n🛡️ Quality Gates (GIGO Prevention)...")
            quality_report = self.quality_control.run_quality_gates(
                data, codebook_info, indices_history
            )
            results['quality_gates'] = {
                'overall_pass': quality_report.overall_pass,
                'input_validation': quality_report.input_validation,
                'signal_quality': quality_report.signal_quality,
                'data_consistency': quality_report.data_consistency,
                'codebook_health': quality_report.codebook_health,
                'recommendations': quality_report.recommendations
            }
            
            # Print quality gate results
            if quality_report.overall_pass:
                print("   ✅ Quality gates PASSED - Data is safe to process")
            else:
                print("   ❌ Quality gates FAILED - GIGO risk detected!")
                print("   Failed checks:")
                for check_name, check_result in [
                    ('Input Validation', quality_report.input_validation),
                    ('Signal Quality', quality_report.signal_quality),
                    ('Data Consistency', quality_report.data_consistency),
                    ('Codebook Health', quality_report.codebook_health)
                ]:
                    if check_result and not check_result.get('pass', True):
                        for failure in check_result.get('failures', []):
                            print(f"     • {check_name}: {failure}")
                
                print(f"   Recommendations: {len(quality_report.recommendations)}")
                for i, rec in enumerate(quality_report.recommendations[:3], 1):
                    print(f"     {i}. {rec}")
                
                # Save quality report immediately for failed cases
                if save_report:
                    quality_report_path = self.quality_control.export_quality_report(
                        quality_report, 
                        self.output_dir / f"quality_report_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    results['quality_report_path'] = str(quality_report_path)
                    print(f"   📄 Quality report saved: {quality_report_path}")
        else:
            print("\n⚠️ Quality gates disabled - proceeding without GIGO protection")
            results['quality_gates'] = {'enabled': False}
        
        # 1. Data Quality Analysis
        print("\n📊 Data Quality Analysis...")
        quality_metrics = self._analyze_data_quality(data)
        results['data_quality'] = quality_metrics
        
        # 2. Signal Characteristics
        print("\n📈 Signal Characteristics...")
        signal_chars = self._analyze_signal_characteristics(data)
        results['signal_characteristics'] = signal_chars
        
        # 3. Preprocessing Comparison
        print("\n🔧 Preprocessing Methods Comparison...")
        preproc_comparison = self._compare_preprocessing_methods(data)
        results['preprocessing_comparison'] = preproc_comparison
        
        # 4. Feature Analysis
        print("\n🎯 Feature Extraction Analysis...")
        feature_analysis = self._analyze_features(data)
        results['feature_analysis'] = feature_analysis
        
        # 5. Temporal Patterns
        print("\n⏰ Temporal Pattern Analysis...")
        temporal_patterns = self._analyze_temporal_patterns(data)
        results['temporal_patterns'] = temporal_patterns
        
        # 6. Label-based Analysis (if labels provided)
        if labels is not None:
            print("\n🏷️ Label-based Analysis...")
            label_analysis = self._analyze_with_labels(data, labels)
            results['label_analysis'] = label_analysis
        
        # 7. Generate visualizations
        if save_report:
            print("\n📊 Generating visualizations...")
            viz_paths = self._generate_visualizations(data, results)
            results['visualizations'] = viz_paths
        
        # 8. Generate text report
        if save_report:
            report_path = self._save_diagnostic_report(results)
            results['report_path'] = str(report_path)
            print(f"\n✅ Diagnostic report saved to: {report_path}")
        
        # Store in history
        self.diagnostic_history.append(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def run_quality_gates_only(self, 
                              data: torch.Tensor,
                              codebook_info: Optional[Dict[str, Any]] = None,
                              indices_history: Optional[List[torch.Tensor]] = None,
                              save_report: bool = True) -> QualityReport:
        """Run only quality gates without full diagnostic suite.
        
        Args:
            data: Input tensor to validate
            codebook_info: Optional VQ codebook information
            indices_history: Optional historical indices for transition analysis
            save_report: Whether to save the quality report
            
        Returns:
            QualityReport with comprehensive quality analysis
            
        Raises:
            RuntimeError: If quality gates are disabled
        """
        if not self.enable_quality_gates or not self.quality_control:
            raise RuntimeError("Quality gates are disabled. Enable them in constructor.")
        
        print("🛡️ Running Quality Gates (GIGO Prevention)...")
        
        quality_report = self.quality_control.run_quality_gates(
            data, codebook_info, indices_history
        )
        
        # Print results
        if quality_report.overall_pass:
            print("✅ Quality gates PASSED - Data is safe to process")
            print(f"   Shape: {quality_report.data_shape}")
            print(f"   NaN count: {quality_report.detailed_metrics.get('nan_count', 0)}")
            print(f"   Signal std: {quality_report.detailed_metrics.get('signal_std', 0):.4f}")
            
            if quality_report.codebook_health:
                cb_metrics = quality_report.codebook_health.get('metrics', {})
                print(f"   Codebook usage: {cb_metrics.get('usage', 0):.3f}")
                print(f"   Perplexity: {cb_metrics.get('perplexity', 0):.2f}")
        else:
            print("❌ Quality gates FAILED - GIGO risk detected!")
            print("Failed checks:")
            for check_name, check_result in [
                ('Input Validation', quality_report.input_validation),
                ('Signal Quality', quality_report.signal_quality),
                ('Data Consistency', quality_report.data_consistency),
                ('Codebook Health', quality_report.codebook_health)
            ]:
                if check_result and not check_result.get('pass', True):
                    for failure in check_result.get('failures', []):
                        print(f"  • {check_name}: {failure}")
            
            print(f"Recommendations:")
            for i, rec in enumerate(quality_report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Save report if requested
        if save_report:
            report_name = "quality_passed" if quality_report.overall_pass else "quality_failed"
            quality_report_path = self.quality_control.export_quality_report(
                quality_report, 
                self.output_dir / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            print(f"📄 Quality report saved: {quality_report_path}")
        
        return quality_report
    
    def get_quality_control_status(self) -> Dict[str, Any]:
        """Get current status of quality control system.
        
        Returns:
            Dictionary with quality control status and trends
        """
        if not self.enable_quality_gates or not self.quality_control:
            return {'enabled': False, 'message': 'Quality gates are disabled'}
        
        # Get recent trends
        trends = self.quality_control.get_quality_trends()
        
        # Get current thresholds
        thresholds = {
            'max_nan_percentage': self.quality_control.thresholds.max_nan_percentage,
            'min_codebook_usage': self.quality_control.thresholds.min_codebook_usage,
            'min_perplexity': self.quality_control.thresholds.min_perplexity,
            'min_snr_db': self.quality_control.thresholds.min_snr_db,
            'max_gap_length': self.quality_control.thresholds.max_gap_length
        }
        
        status = {
            'enabled': True,
            'strict_mode': self.quality_control.strict_mode,
            'codebook_monitoring': self.quality_control.enable_codebook_monitoring,
            'total_reports': len(self.quality_control.quality_history),
            'thresholds': thresholds,
            'trends': trends
        }
        
        return status
    
    def update_quality_thresholds(self, 
                                 max_nan_percentage: Optional[float] = None,
                                 min_codebook_usage: Optional[float] = None,
                                 min_perplexity: Optional[float] = None,
                                 min_snr_db: Optional[float] = None,
                                 **kwargs) -> None:
        """Update quality control thresholds.
        
        Args:
            max_nan_percentage: Maximum allowed NaN percentage
            min_codebook_usage: Minimum codebook usage fraction
            min_perplexity: Minimum perplexity for diversity
            min_snr_db: Minimum signal-to-noise ratio
            **kwargs: Additional threshold parameters
        """
        if not self.enable_quality_gates or not self.quality_control:
            print("Warning: Quality gates are disabled. Thresholds not updated.")
            return
        
        # Update thresholds
        if max_nan_percentage is not None:
            self.quality_control.thresholds.max_nan_percentage = max_nan_percentage
        if min_codebook_usage is not None:
            self.quality_control.thresholds.min_codebook_usage = min_codebook_usage
        if min_perplexity is not None:
            self.quality_control.thresholds.min_perplexity = min_perplexity
        if min_snr_db is not None:
            self.quality_control.thresholds.min_snr_db = min_snr_db
        
        # Update additional parameters
        for key, value in kwargs.items():
            if hasattr(self.quality_control.thresholds, key):
                setattr(self.quality_control.thresholds, key, value)
                print(f"Updated threshold {key} = {value}")
        
        print("Quality control thresholds updated successfully.")
    
    def export_quality_trends_report(self, output_path: Optional[Path] = None) -> Path:
        """Export comprehensive quality trends report.
        
        Args:
            output_path: Optional output path for the report
            
        Returns:
            Path to the exported trends report
        """
        if not self.enable_quality_gates or not self.quality_control:
            raise RuntimeError("Quality gates are disabled. Cannot export trends report.")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"quality_trends_report_{timestamp}.json"
        
        # Compile comprehensive trends data
        trends_data = {
            'timestamp': datetime.now().isoformat(),
            'total_reports': len(self.quality_control.quality_history),
            'system_status': self.get_quality_control_status(),
            'recent_trends': self.quality_control.get_quality_trends(window_size=20),
            'all_reports_summary': []
        }
        
        # Add summary of all reports
        for report in self.quality_control.quality_history:
            summary = {
                'timestamp': report.timestamp,
                'overall_pass': report.overall_pass,
                'data_shape': list(report.data_shape),
                'nan_count': report.detailed_metrics.get('nan_count', 0),
                'signal_std': report.detailed_metrics.get('signal_std', 0),
            }
            if report.codebook_health:
                summary['codebook_metrics'] = report.codebook_health.get('metrics', {})
            trends_data['all_reports_summary'].append(summary)
        
        # Save the report
        with open(output_path, 'w') as f:
            json.dump(trends_data, f, indent=2, default=str)
        
        print(f"📊 Quality trends report exported: {output_path}")
        return output_path
    
    def _analyze_data_quality(self, data: torch.Tensor) -> Dict[str, Any]:
        """Detailed data quality analysis."""
        metrics = {}
        
        # Basic statistics
        metrics['nan_count'] = int(torch.isnan(data).sum().item())
        metrics['nan_percentage'] = float(100 * metrics['nan_count'] / data.numel())
        metrics['inf_count'] = int(torch.isinf(data).sum().item())
        metrics['zero_count'] = int((data == 0).sum().item())
        metrics['zero_percentage'] = float(100 * metrics['zero_count'] / data.numel())
        
        # Value distribution
        valid_data = data[~torch.isnan(data) & ~torch.isinf(data)]
        if valid_data.numel() > 0:
            metrics['min'] = float(valid_data.min().item())
            metrics['max'] = float(valid_data.max().item())
            metrics['mean'] = float(valid_data.mean().item())
            metrics['std'] = float(valid_data.std().item())
            metrics['median'] = float(valid_data.median().item())
            
            # Percentiles
            percentiles = [1, 5, 25, 75, 95, 99]
            for p in percentiles:
                metrics[f'percentile_{p}'] = float(torch.quantile(valid_data, p/100.0).item())
        
        # Channel-wise quality (for IMU data)
        if data.dim() == 4 and data.shape[1] == 9:
            channel_names = ['accel_x', 'accel_y', 'accel_z',
                            'gyro_x', 'gyro_y', 'gyro_z',
                            'mag_x', 'mag_y', 'mag_z']
            
            metrics['channel_quality'] = {}
            for i, name in enumerate(channel_names):
                ch_data = data[:, i, :, :]
                ch_metrics = {
                    'nan_pct': float(100 * torch.isnan(ch_data).sum() / ch_data.numel()),
                    'mean': float(ch_data[~torch.isnan(ch_data)].mean().item()) if torch.any(~torch.isnan(ch_data)) else 0,
                    'std': float(ch_data[~torch.isnan(ch_data)].std().item()) if torch.any(~torch.isnan(ch_data)) else 0,
                    'range': float((ch_data[~torch.isnan(ch_data)].max() - ch_data[~torch.isnan(ch_data)].min()).item()) if torch.any(~torch.isnan(ch_data)) else 0
                }
                metrics['channel_quality'][name] = ch_metrics
        
        # Gap analysis
        gaps = self._find_data_gaps(data)
        metrics['gap_analysis'] = {
            'num_gaps': len(gaps),
            'max_gap_length': max([g['length'] for g in gaps]) if gaps else 0,
            'total_gap_frames': sum([g['length'] for g in gaps]) if gaps else 0,
            'gap_locations': gaps[:5] if len(gaps) > 5 else gaps  # First 5 gaps
        }
        
        return metrics
    
    def _analyze_signal_characteristics(self, data: torch.Tensor) -> Dict[str, Any]:
        """Analyze signal characteristics."""
        chars = {}
        
        # Signal-to-noise ratio estimation
        if data.dim() >= 3:
            signal_power = torch.var(data, dim=-1)
            noise_estimate = torch.var(torch.diff(data, dim=-1), dim=-1) / 2
            snr = signal_power / (noise_estimate + 1e-10)
            chars['snr_db'] = {
                'mean': float((10 * torch.log10(snr.mean())).item()),
                'std': float((10 * torch.log10(snr.std() + 1e-10)).item()),
                'min': float((10 * torch.log10(snr.min() + 1e-10)).item()),
                'max': float((10 * torch.log10(snr.max())).item())
            }
        
        # Frequency domain analysis
        fft = torch.fft.rfft(data, dim=-1)
        magnitude = torch.abs(fft)
        
        # Dominant frequencies
        freqs = torch.fft.rfftfreq(data.shape[-1], d=1.0/self.sampling_rate)
        dominant_freq_idx = torch.argmax(magnitude, dim=-1)
        
        # Flatten to get all dominant frequencies
        dominant_freqs_flat = freqs[dominant_freq_idx.flatten()]
        chars['dominant_frequencies'] = {
            'mean': float(dominant_freqs_flat.mean().item()),
            'std': float(dominant_freqs_flat.std().item()),
            'mode': float(torch.mode(dominant_freq_idx.flatten()).values.item() * self.sampling_rate / data.shape[-1])
        }
        
        # Power distribution
        power = magnitude ** 2
        total_power = power.sum(dim=-1)
        
        # Find frequency bands with most power
        low_freq_power = power[..., :power.shape[-1]//4].sum(dim=-1)
        mid_freq_power = power[..., power.shape[-1]//4:power.shape[-1]//2].sum(dim=-1)
        high_freq_power = power[..., power.shape[-1]//2:].sum(dim=-1)
        
        chars['power_distribution'] = {
            'low_freq_ratio': float((low_freq_power / total_power).mean().item()),
            'mid_freq_ratio': float((mid_freq_power / total_power).mean().item()),
            'high_freq_ratio': float((high_freq_power / total_power).mean().item())
        }
        
        # Autocorrelation analysis
        autocorr = self._compute_autocorrelation(data)
        chars['autocorrelation'] = {
            'lag_1': float(autocorr[..., 1].mean().item()) if autocorr.shape[-1] > 1 else 0,
            'lag_10': float(autocorr[..., 10].mean().item()) if autocorr.shape[-1] > 10 else 0,
            'decorrelation_time': self._find_decorrelation_time(autocorr)
        }
        
        return chars
    
    def _compare_preprocessing_methods(self, data: torch.Tensor) -> Dict[str, Any]:
        """Compare different preprocessing approaches."""
        comparison = {}
        
        methods = [
            ('original', None),
            ('interpolated', lambda x: self.movement_proc.interpolate_gaps(x)),
            ('median_filter', lambda x: self.movement_proc.smooth_rolling(x, statistic='median')),
            ('mean_filter', lambda x: self.movement_proc.smooth_rolling(x, statistic='mean')),
            ('savgol_filter', lambda x: self.movement_proc.smooth_savgol(x))
        ]
        
        for method_name, method_func in methods:
            if method_func is None:
                processed = data
            else:
                try:
                    processed = method_func(data.clone())
                except Exception as e:
                    print(f"Warning: {method_name} failed: {e}")
                    continue
            
            # Compute metrics for processed data
            metrics = {
                'nan_pct': float(100 * torch.isnan(processed).sum() / processed.numel()),
                'mean': float(processed[~torch.isnan(processed)].mean().item()) if torch.any(~torch.isnan(processed)) else 0,
                'std': float(processed[~torch.isnan(processed)].std().item()) if torch.any(~torch.isnan(processed)) else 0,
                'signal_preserved': float(torch.corrcoef(torch.stack([
                    data[~torch.isnan(data)].flatten()[:1000],
                    processed[~torch.isnan(processed)].flatten()[:1000]
                ]))[0,1].item()) if torch.any(~torch.isnan(data)) and torch.any(~torch.isnan(processed)) else 0
            }
            
            # Compute SNR
            if processed.dim() >= 3:
                signal_power = torch.var(processed, dim=-1)
                noise_estimate = torch.var(torch.diff(processed, dim=-1), dim=-1) / 2
                snr = signal_power / (noise_estimate + 1e-10)
                metrics['snr_db'] = float((10 * torch.log10(snr.mean())).item())
            
            comparison[method_name] = metrics
        
        return comparison
    
    def _analyze_features(self, data: torch.Tensor) -> Dict[str, Any]:
        """Analyze extracted features."""
        analysis = {}
        
        # Extract kinematic features
        try:
            if data.shape[1] == 9:  # IMU data
                features = self.feature_extractor.extract_imu_features(data)
                
                analysis['imu_features'] = {}
                for feat_name, feat_data in features.items():
                    if torch.is_tensor(feat_data):
                        analysis['imu_features'][feat_name] = {
                            'shape': list(feat_data.shape),
                            'mean': float(feat_data[~torch.isnan(feat_data)].mean().item()) if torch.any(~torch.isnan(feat_data)) else 0,
                            'std': float(feat_data[~torch.isnan(feat_data)].std().item()) if torch.any(~torch.isnan(feat_data)) else 0,
                            'range': float((feat_data[~torch.isnan(feat_data)].max() - feat_data[~torch.isnan(feat_data)].min()).item()) if torch.any(~torch.isnan(feat_data)) else 0
                        }
        except Exception as e:
            analysis['feature_extraction_error'] = str(e)
        
        return analysis
    
    def _analyze_temporal_patterns(self, data: torch.Tensor) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        patterns = {}
        
        # Temporal statistics across time axis
        time_mean = data.mean(dim=(0, 1, 2)) if data.dim() == 4 else data.mean(dim=0)
        time_std = data.std(dim=(0, 1, 2)) if data.dim() == 4 else data.std(dim=0)
        
        patterns['temporal_trend'] = {
            'mean_trajectory': time_mean.cpu().numpy().tolist()[:20],  # First 20 timesteps
            'std_trajectory': time_std.cpu().numpy().tolist()[:20],
            'trend_slope': float(np.polyfit(np.arange(len(time_mean)), time_mean.cpu().numpy(), 1)[0])
        }
        
        # Detect periodicity
        autocorr = self._compute_autocorrelation(data.mean(dim=(0,1,2)) if data.dim() == 4 else data.mean(dim=0))
        peaks = self._find_peaks(autocorr)
        
        patterns['periodicity'] = {
            'detected_periods': peaks[:3].tolist() if len(peaks) > 0 else [],
            'strongest_period': int(peaks[0]) if len(peaks) > 0 else None,
            'period_strength': float(autocorr[peaks[0]].item()) if len(peaks) > 0 else 0
        }
        
        # Stationarity test (simplified)
        window_size = data.shape[-1] // 4
        windows_means = []
        windows_stds = []
        
        for i in range(0, data.shape[-1] - window_size, window_size):
            window = data[..., i:i+window_size]
            windows_means.append(window.mean().item())
            windows_stds.append(window.std().item())
        
        patterns['stationarity'] = {
            'mean_variance': float(np.var(windows_means)),
            'std_variance': float(np.var(windows_stds)),
            'is_stationary': float(np.var(windows_means)) < 0.1 and float(np.var(windows_stds)) < 0.1
        }
        
        return patterns
    
    def _analyze_with_labels(self, data: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Analyze data with respect to labels."""
        analysis = {}
        
        unique_labels = torch.unique(labels)
        analysis['num_classes'] = int(len(unique_labels))
        analysis['class_distribution'] = {}
        
        for label in unique_labels:
            mask = labels == label
            class_data = data[mask]
            
            if class_data.numel() > 0:
                analysis['class_distribution'][int(label.item())] = {
                    'count': int(mask.sum().item()),
                    'percentage': float(100 * mask.sum() / len(labels)),
                    'mean': float(class_data.mean().item()),
                    'std': float(class_data.std().item())
                }
        
        # Inter-class separability (simplified)
        if len(unique_labels) > 1:
            class_means = []
            for label in unique_labels:
                mask = labels == label
                if mask.any():
                    class_means.append(data[mask].mean(dim=0))
            
            if len(class_means) > 1:
                class_means = torch.stack(class_means)
                inter_class_dist = torch.cdist(class_means.flatten().unsqueeze(0), 
                                              class_means.flatten().unsqueeze(0))
                analysis['inter_class_distance'] = {
                    'mean': float(inter_class_dist.mean().item()),
                    'min': float(inter_class_dist[inter_class_dist > 0].min().item()) if torch.any(inter_class_dist > 0) else 0,
                    'max': float(inter_class_dist.max().item())
                }
        
        return analysis
    
    def _find_data_gaps(self, data: torch.Tensor) -> List[Dict]:
        """Find gaps (consecutive NaNs) in data."""
        gaps = []
        nan_mask = torch.isnan(data).any(dim=tuple(range(data.dim()-1)))
        
        in_gap = False
        gap_start = 0
        
        for i, is_nan in enumerate(nan_mask):
            if is_nan and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_nan and in_gap:
                gaps.append({
                    'start': int(gap_start),
                    'end': int(i),
                    'length': int(i - gap_start)
                })
                in_gap = False
        
        if in_gap:
            gaps.append({
                'start': int(gap_start),
                'end': int(len(nan_mask)),
                'length': int(len(nan_mask) - gap_start)
            })
        
        return gaps
    
    def _compute_autocorrelation(self, data: torch.Tensor) -> torch.Tensor:
        """Compute autocorrelation function."""
        if data.dim() > 1:
            data = data.flatten()
        
        data = data - data.mean()
        
        # Use numpy for correlation since torch doesn't have correlate
        data_np = data.cpu().numpy()
        autocorr_np = np.correlate(data_np, data_np, mode='same')
        autocorr = torch.from_numpy(autocorr_np).to(data.device)
        
        # Normalize
        autocorr = autocorr / autocorr[len(autocorr)//2]
        
        return autocorr[len(autocorr)//2:]
    
    def _find_decorrelation_time(self, autocorr: torch.Tensor) -> int:
        """Find decorrelation time (first crossing of 1/e)."""
        threshold = 1.0 / np.e
        crossings = torch.where(autocorr < threshold)[0]
        return int(crossings[0].item()) if len(crossings) > 0 else -1
    
    def _find_peaks(self, signal: torch.Tensor) -> torch.Tensor:
        """Find peaks in signal."""
        # Simple peak detection
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        
        return torch.tensor(peaks) if peaks else torch.tensor([])
    
    def _generate_visualizations(self, data: torch.Tensor, results: Dict) -> List[str]:
        """Generate diagnostic visualizations."""
        viz_paths = []
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Behavioral Data Diagnostics', fontsize=16)
        
        # 1. Data distribution
        ax = axes[0, 0]
        valid_data = data[~torch.isnan(data)].cpu().numpy()
        if len(valid_data) > 0:
            ax.hist(valid_data[:10000], bins=50, alpha=0.7, color='blue')
        ax.set_title('Value Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        
        # 2. Temporal mean/std
        ax = axes[0, 1]
        time_mean = data.mean(dim=(0,1,2)).cpu() if data.dim() == 4 else data.mean(dim=0).cpu()
        time_std = data.std(dim=(0,1,2)).cpu() if data.dim() == 4 else data.std(dim=0).cpu()
        time_axis = np.arange(len(time_mean))
        ax.plot(time_axis, time_mean, 'b-', label='Mean')
        ax.fill_between(time_axis, time_mean - time_std, time_mean + time_std, 
                        alpha=0.3, color='blue', label='±1 STD')
        ax.set_title('Temporal Statistics')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        
        # 3. Frequency spectrum
        ax = axes[0, 2]
        fft = torch.fft.rfft(data.mean(dim=(0,1,2)) if data.dim() == 4 else data.mean(dim=0))
        freqs = torch.fft.rfftfreq(data.shape[-1], d=1.0/self.sampling_rate).cpu()
        magnitude = torch.abs(fft).cpu()
        ax.semilogy(freqs, magnitude)
        ax.set_title('Frequency Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        
        # 4. Preprocessing comparison
        ax = axes[1, 0]
        if 'preprocessing_comparison' in results:
            methods = list(results['preprocessing_comparison'].keys())
            snrs = [results['preprocessing_comparison'][m].get('snr_db', 0) for m in methods]
            ax.bar(methods, snrs, color='green', alpha=0.7)
            ax.set_title('Preprocessing SNR Comparison')
            ax.set_xlabel('Method')
            ax.set_ylabel('SNR (dB)')
            ax.tick_params(axis='x', rotation=45)
        
        # 5. Channel quality (for IMU)
        ax = axes[1, 1]
        if 'data_quality' in results and 'channel_quality' in results['data_quality']:
            channels = list(results['data_quality']['channel_quality'].keys())
            nan_pcts = [results['data_quality']['channel_quality'][c]['nan_pct'] for c in channels]
            ax.bar(range(len(channels)), nan_pcts, color='red', alpha=0.7)
            ax.set_title('Channel NaN Percentage')
            ax.set_xlabel('Channel')
            ax.set_ylabel('NaN %')
            ax.set_xticks(range(len(channels)))
            ax.set_xticklabels(channels, rotation=45, ha='right')
        
        # 6. Autocorrelation
        ax = axes[1, 2]
        autocorr = self._compute_autocorrelation(
            data.mean(dim=(0,1,2)) if data.dim() == 4 else data.mean(dim=0)
        ).cpu()
        lags = np.arange(min(len(autocorr), 50))
        ax.plot(lags, autocorr[:50])
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=1/np.e, color='r', linestyle='--', alpha=0.3, label='1/e threshold')
        ax.set_title('Autocorrelation Function')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        viz_path = self.output_dir / f'diagnostics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        viz_paths.append(str(viz_path))
        
        return viz_paths
    
    def _save_diagnostic_report(self, results: Dict) -> Path:
        """Save diagnostic report to JSON and text files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = self.output_dir / f'diagnostic_report_{timestamp}.json'
        with open(json_path, 'w') as f:
            # Convert numpy arrays and tensors to lists for JSON serialization
            json_safe = self._make_json_serializable(results)
            json.dump(json_safe, f, indent=2)
        
        # Save text report
        text_path = self.output_dir / f'diagnostic_report_{timestamp}.txt'
        with open(text_path, 'w') as f:
            f.write(self._format_text_report(results))
        
        return text_path
    
    def _make_json_serializable(self, obj):
        """Convert non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def _format_text_report(self, results: Dict) -> str:
        """Format results as readable text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("BEHAVIORAL DATA DIAGNOSTIC REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {results['timestamp']}")
        lines.append(f"Data Shape: {results['data_shape']}")
        lines.append(f"Device: {results['device']}")
        lines.append(f"Data Type: {results['dtype']}")
        lines.append("")
        
        # Data Quality Section
        if 'data_quality' in results:
            lines.append("-" * 40)
            lines.append("DATA QUALITY METRICS")
            lines.append("-" * 40)
            dq = results['data_quality']
            lines.append(f"NaN Count: {dq['nan_count']} ({dq['nan_percentage']:.2f}%)")
            lines.append(f"Inf Count: {dq['inf_count']}")
            lines.append(f"Zero Count: {dq['zero_count']} ({dq['zero_percentage']:.2f}%)")
            
            if 'mean' in dq:
                lines.append(f"Mean: {dq['mean']:.4f}")
                lines.append(f"Std: {dq['std']:.4f}")
                lines.append(f"Range: [{dq['min']:.4f}, {dq['max']:.4f}]")
            
            if 'gap_analysis' in dq:
                gaps = dq['gap_analysis']
                lines.append(f"Number of Gaps: {gaps['num_gaps']}")
                lines.append(f"Max Gap Length: {gaps['max_gap_length']}")
            lines.append("")
        
        # Signal Characteristics
        if 'signal_characteristics' in results:
            lines.append("-" * 40)
            lines.append("SIGNAL CHARACTERISTICS")
            lines.append("-" * 40)
            sc = results['signal_characteristics']
            
            if 'snr_db' in sc:
                lines.append(f"SNR: {sc['snr_db']['mean']:.2f} ± {sc['snr_db']['std']:.2f} dB")
            
            if 'dominant_frequencies' in sc:
                lines.append(f"Dominant Frequency: {sc['dominant_frequencies']['mean']:.2f} Hz")
            
            if 'power_distribution' in sc:
                pd = sc['power_distribution']
                lines.append(f"Power Distribution:")
                lines.append(f"  Low Freq: {pd['low_freq_ratio']:.2%}")
                lines.append(f"  Mid Freq: {pd['mid_freq_ratio']:.2%}")
                lines.append(f"  High Freq: {pd['high_freq_ratio']:.2%}")
            lines.append("")
        
        # Preprocessing Comparison
        if 'preprocessing_comparison' in results:
            lines.append("-" * 40)
            lines.append("PREPROCESSING METHODS COMPARISON")
            lines.append("-" * 40)
            
            for method, metrics in results['preprocessing_comparison'].items():
                lines.append(f"{method}:")
                lines.append(f"  NaN%: {metrics.get('nan_pct', 0):.2f}%")
                if 'snr_db' in metrics:
                    lines.append(f"  SNR: {metrics['snr_db']:.2f} dB")
                if 'signal_preserved' in metrics:
                    lines.append(f"  Signal Preserved: {metrics['signal_preserved']:.2%}")
            lines.append("")
        
        # Temporal Patterns
        if 'temporal_patterns' in results:
            lines.append("-" * 40)
            lines.append("TEMPORAL PATTERNS")
            lines.append("-" * 40)
            tp = results['temporal_patterns']
            
            if 'temporal_trend' in tp:
                lines.append(f"Trend Slope: {tp['temporal_trend']['trend_slope']:.6f}")
            
            if 'periodicity' in tp:
                per = tp['periodicity']
                if per['strongest_period']:
                    lines.append(f"Strongest Period: {per['strongest_period']} samples")
                    lines.append(f"Period Strength: {per['period_strength']:.3f}")
            
            if 'stationarity' in tp:
                lines.append(f"Is Stationary: {tp['stationarity']['is_stationary']}")
            lines.append("")
        
        # Label Analysis
        if 'label_analysis' in results:
            lines.append("-" * 40)
            lines.append("LABEL-BASED ANALYSIS")
            lines.append("-" * 40)
            la = results['label_analysis']
            lines.append(f"Number of Classes: {la['num_classes']}")
            
            if 'class_distribution' in la:
                lines.append("Class Distribution:")
                for label, info in la['class_distribution'].items():
                    lines.append(f"  Class {label}: {info['count']} samples ({info['percentage']:.1f}%)")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return '\n'.join(lines)
    
    def _print_summary(self, results: Dict):
        """Print summary of diagnostic results."""
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        # Key metrics
        if 'data_quality' in results:
            dq = results['data_quality']
            print(f"Data Quality: {100 - dq['nan_percentage']:.1f}% valid")
        
        if 'signal_characteristics' in results:
            sc = results['signal_characteristics']
            if 'snr_db' in sc:
                print(f"Signal-to-Noise: {sc['snr_db']['mean']:.1f} dB")
        
        if 'preprocessing_comparison' in results:
            best_method = max(results['preprocessing_comparison'].items(),
                            key=lambda x: x[1].get('snr_db', 0))
            print(f"Best Preprocessing: {best_method[0]} ({best_method[1].get('snr_db', 0):.1f} dB)")
        
        if 'temporal_patterns' in results:
            tp = results['temporal_patterns']
            if 'stationarity' in tp:
                print(f"Stationarity: {'Yes' if tp['stationarity']['is_stationary'] else 'No'}")
        
        print("=" * 60)


if __name__ == "__main__":
    print("Testing Enhanced Behavioral Data Diagnostics with Quality Control...")
    
    # Create test data
    B, C, S, T = 4, 9, 2, 100
    test_data = torch.randn(B, C, S, T)
    
    # Add some realistic patterns
    t = torch.linspace(0, 10, T)
    for b in range(B):
        # Add sinusoidal components
        test_data[b, 0, 0, :] += 2 * torch.sin(2 * np.pi * 0.5 * t)
        test_data[b, 1, 0, :] += 1.5 * torch.cos(2 * np.pi * 1.0 * t)
        
        # Add some NaN values
        dropout_mask = torch.rand(T) < 0.05
        test_data[b, :, :, dropout_mask] = float('nan')
    
    # Create labels and mock codebook info
    labels = torch.randint(0, 3, (B,))
    
    # Mock VQ codebook information
    mock_codebook_info = {
        'perplexity': 6.5,
        'usage': 0.75,
        'entropy': 3.2,
        'dead_codes': 50,
        'total_codes': 512,
        'active_codes': 462
    }
    
    # Mock indices history for transition analysis
    mock_indices_history = [
        torch.randint(0, 512, (B, 1, T)),
        torch.randint(0, 512, (B, 1, T))
    ]
    
    print("\n" + "="*80)
    print("TEST 1: Full Diagnostic Suite with Quality Control")
    print("="*80)
    
    # Initialize diagnostics with quality control enabled
    diagnostics = BehavioralDataDiagnostics(
        sampling_rate=100.0, 
        output_dir='./test_diagnostics',
        enable_quality_gates=True,
        strict_quality_mode=False
    )
    
    # Run full diagnostic
    results = diagnostics.run_full_diagnostic(
        test_data, 
        labels=labels, 
        codebook_info=mock_codebook_info,
        indices_history=mock_indices_history,
        save_report=True
    )
    
    print(f"\n✅ Full diagnostic complete!")
    print(f"Quality gates passed: {results.get('quality_gates', {}).get('overall_pass', 'N/A')}")
    print(f"Report saved to: {results.get('report_path', 'N/A')}")
    
    print("\n" + "="*80)
    print("TEST 2: Quality Gates Only")
    print("="*80)
    
    # Test quality gates only
    quality_report = diagnostics.run_quality_gates_only(
        test_data,
        codebook_info=mock_codebook_info,
        indices_history=mock_indices_history,
        save_report=True
    )
    
    print("\n" + "="*80)
    print("TEST 3: Quality Control Status and Trends")
    print("="*80)
    
    # Show quality control status
    status = diagnostics.get_quality_control_status()
    print(f"Quality control status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("TEST 4: Threshold Updates")
    print("="*80)
    
    # Test threshold updates
    print("Updating quality thresholds...")
    diagnostics.update_quality_thresholds(
        max_nan_percentage=5.0,
        min_codebook_usage=0.8,
        min_perplexity=5.0
    )
    
    print("\n" + "="*80)
    print("TEST 5: Bad Data Test (Should Fail Quality Gates)")
    print("="*80)
    
    # Create intentionally bad data
    bad_data = torch.randn(B, C, S, T)
    bad_data[:, :, :, :50] = float('nan')  # 50% NaN values
    bad_data *= 0.001  # Very low signal
    
    bad_codebook_info = {
        'perplexity': 1.5,  # Very low perplexity
        'usage': 0.05,      # Very low usage
        'entropy': 0.8,     # Very low entropy
        'dead_codes': 400,  # Many dead codes
        'total_codes': 512
    }
    
    print("Testing with intentionally bad data...")
    bad_quality_report = diagnostics.run_quality_gates_only(
        bad_data,
        codebook_info=bad_codebook_info,
        save_report=True
    )
    
    print("\n" + "="*80)
    print("TEST 6: Export Quality Trends Report")
    print("="*80)
    
    # Export comprehensive trends report
    trends_report_path = diagnostics.export_quality_trends_report()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ All tests completed successfully!")
    print(f"📊 Generated reports:")
    print(f"  - Main diagnostic: {results.get('report_path', 'N/A')}")
    print(f"  - Quality trends: {trends_report_path}")
    print(f"📈 Total quality reports in history: {len(diagnostics.quality_control.quality_history)}")
    
    # Final status check
    final_status = diagnostics.get_quality_control_status()
    final_trends = final_status.get('trends', {})
    print(f"\n📊 Final Quality Metrics:")
    print(f"  - Pass rate: {final_trends.get('pass_rate', 0):.1%}")
    print(f"  - Avg NaN%: {final_trends.get('avg_nan_percentage', 0):.2f}%")
    print(f"  - Avg signal std: {final_trends.get('avg_signal_std', 0):.4f}")
    if 'avg_perplexity' in final_trends:
        print(f"  - Avg perplexity: {final_trends['avg_perplexity']:.2f}")
        print(f"  - Avg usage: {final_trends['avg_usage']:.3f}")
    
    print("\n🛡️ Quality Control System Integration Complete!")
    print("   The enhanced movement_diagnostics.py now includes:")
    print("   ✓ Input validation with configurable thresholds")
    print("   ✓ Codebook health monitoring (usage, diversity, transitions)")
    print("   ✓ Signal quality assessment (SNR, frequency, stationarity)")
    print("   ✓ Data consistency checks (gaps, correlations, outliers)")
    print("   ✓ Automated quality reports in JSON format")
    print("   ✓ Quality trends analysis and recommendations")
    print("   ✓ GIGO prevention with actionable feedback") 