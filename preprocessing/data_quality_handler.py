#!/usr/bin/env python3
"""
Improved data quality handling with configurable NaN strategies and comprehensive logging.

This module addresses the issue of silently replacing NaN values with zeros by providing:
1. Multiple replacement strategies (zero, mean, median, interpolation, drop)
2. Detailed logging of data quality issues
3. Configurable thresholds for automatic strategy selection
4. Quality scoring and reporting
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional, Literal
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NaNStrategy = Literal['zero', 'mean', 'median', 'interpolate', 'drop', 'raise']

class DataQualityHandler:
    """Comprehensive data quality monitoring and correction handler."""
    
    def __init__(
        self,
        nan_threshold_warn: float = 5.0,
        nan_threshold_error: float = 20.0,
        inf_threshold_warn: float = 1.0,
        default_nan_strategy: NaNStrategy = 'mean',
        auto_select_strategy: bool = True
    ):
        """
        Initialize data quality handler.
        
        Args:
            nan_threshold_warn: Percentage of NaN to trigger warning
            nan_threshold_error: Percentage of NaN to trigger error/drop
            inf_threshold_warn: Percentage of Inf to trigger warning
            default_nan_strategy: Default strategy for NaN handling
            auto_select_strategy: Automatically select best strategy based on data
        """
        self.nan_threshold_warn = nan_threshold_warn
        self.nan_threshold_error = nan_threshold_error
        self.inf_threshold_warn = inf_threshold_warn
        self.default_nan_strategy = default_nan_strategy
        self.auto_select_strategy = auto_select_strategy
        
        # Track correction history
        self.correction_history = []
        
    def analyze_quality(self, data: np.ndarray, name: str = "data") -> Dict:
        """
        Analyze data quality and return detailed metrics.
        
        Args:
            data: Input data array (B, C, T) or similar
            name: Name of the dataset for logging
            
        Returns:
            Dictionary containing quality metrics
        """
        metrics = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'shape': data.shape,
            'dtype': str(data.dtype),
            'total_elements': data.size,
        }
        
        # NaN analysis
        nan_mask = np.isnan(data)
        nan_count = nan_mask.sum()
        nan_percentage = (nan_count / data.size) * 100
        
        metrics['nan'] = {
            'count': int(nan_count),
            'percentage': float(nan_percentage),
            'locations': np.where(nan_mask) if nan_count < 100 else 'too_many',
            'affected_samples': int(np.any(nan_mask, axis=tuple(range(1, len(data.shape)))).sum()),
        }
        
        # Per-channel NaN analysis for multidimensional data
        if len(data.shape) >= 2:
            metrics['nan']['by_channel'] = [
                float(np.isnan(data[:, i]).sum() / data[:, i].size * 100)
                for i in range(data.shape[1])
            ]
        
        # Inf analysis
        inf_mask = np.isinf(data)
        inf_count = inf_mask.sum()
        inf_percentage = (inf_count / data.size) * 100
        
        metrics['inf'] = {
            'count': int(inf_count),
            'percentage': float(inf_percentage),
            'pos_inf': int(np.isposinf(data).sum()),
            'neg_inf': int(np.isneginf(data).sum()),
        }
        
        # Statistical properties (excluding NaN/Inf)
        finite_mask = np.isfinite(data)
        if np.any(finite_mask):
            finite_data = data[finite_mask]
            metrics['stats'] = {
                'mean': float(np.mean(finite_data)),
                'std': float(np.std(finite_data)),
                'min': float(np.min(finite_data)),
                'max': float(np.max(finite_data)),
                'median': float(np.median(finite_data)),
            }
        else:
            metrics['stats'] = None
            
        # Quality score (0-100)
        metrics['quality_score'] = self._calculate_quality_score(metrics)
        
        # Recommended action
        metrics['recommended_action'] = self._recommend_action(metrics)
        
        return metrics
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score from 0-100."""
        score = 100.0
        
        # Penalize NaN values
        nan_penalty = min(metrics['nan']['percentage'] * 2, 50)
        score -= nan_penalty
        
        # Penalize Inf values
        inf_penalty = min(metrics['inf']['percentage'] * 5, 30)
        score -= inf_penalty
        
        # Penalize low variance (might indicate constant/dead sensors)
        if metrics['stats'] and metrics['stats']['std'] < 1e-6:
            score -= 10
            
        return max(0, score)
    
    def _recommend_action(self, metrics: Dict) -> Dict:
        """Recommend correction action based on metrics."""
        nan_pct = metrics['nan']['percentage']
        inf_pct = metrics['inf']['percentage']
        
        action = {
            'needed': False,
            'nan_strategy': 'none',
            'severity': 'ok'
        }
        
        if nan_pct > 0 or inf_pct > 0:
            action['needed'] = True
            
            # Determine NaN strategy
            if nan_pct > self.nan_threshold_error:
                action['nan_strategy'] = 'drop'
                action['severity'] = 'critical'
            elif nan_pct > self.nan_threshold_warn:
                action['nan_strategy'] = 'interpolate'
                action['severity'] = 'warning'
            elif nan_pct > 0:
                action['nan_strategy'] = 'mean'
                action['severity'] = 'info'
                
        return action
    
    def correct_data(
        self,
        data: np.ndarray,
        nan_strategy: Optional[NaNStrategy] = None,
        log_details: bool = True,
        name: str = "data"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply corrections to data based on quality issues.
        
        Args:
            data: Input data to correct
            nan_strategy: Override strategy for NaN handling
            log_details: Whether to log detailed correction info
            name: Name for logging
            
        Returns:
            Tuple of (corrected_data, correction_report)
        """
        # Analyze quality first
        metrics = self.analyze_quality(data, name)
        
        # Initialize report
        report = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'original_shape': data.shape,
            'quality_before': metrics['quality_score'],
            'corrections_applied': []
        }
        
        # Determine NaN strategy
        if nan_strategy is None:
            if self.auto_select_strategy:
                nan_strategy = metrics['recommended_action']['nan_strategy']
                if nan_strategy == 'none':
                    nan_strategy = self.default_nan_strategy
            else:
                nan_strategy = self.default_nan_strategy
                
        # Apply NaN corrections
        if metrics['nan']['count'] > 0:
            data, nan_report = self._correct_nan(
                data, nan_strategy, metrics['nan'], log_details
            )
            report['corrections_applied'].append(nan_report)
            
        # Apply Inf corrections
        if metrics['inf']['count'] > 0:
            data, inf_report = self._correct_inf(
                data, metrics['inf'], log_details
            )
            report['corrections_applied'].append(inf_report)
            
        # Re-analyze quality after corrections
        metrics_after = self.analyze_quality(data, f"{name}_corrected")
        report['quality_after'] = metrics_after['quality_score']
        report['final_shape'] = data.shape
        
        # Log summary
        if log_details:
            self._log_correction_summary(report)
            
        # Store in history
        self.correction_history.append(report)
        
        return data, report
    
    def _correct_nan(
        self,
        data: np.ndarray,
        strategy: str,
        nan_metrics: Dict,
        log_details: bool
    ) -> Tuple[np.ndarray, Dict]:
        """Apply NaN correction strategy."""
        nan_report = {
            'type': 'nan_correction',
            'strategy': strategy,
            'count_before': nan_metrics['count'],
            'percentage_before': nan_metrics['percentage']
        }
        
        if log_details:
            logger.info(f"Applying NaN strategy: {strategy}")
            logger.info(f"NaN count: {nan_metrics['count']} ({nan_metrics['percentage']:.2f}%)")
        
        if strategy == 'zero':
            data = np.nan_to_num(data, nan=0.0)
            nan_report['replacement_value'] = 0.0
            
        elif strategy == 'mean':
            # Replace with per-channel mean
            if len(data.shape) >= 2:
                for i in range(data.shape[1]):
                    channel = data[:, i]
                    if np.any(np.isnan(channel)):
                        mean_val = np.nanmean(channel)
                        data[:, i] = np.where(np.isnan(channel), mean_val, channel)
                        if log_details:
                            logger.debug(f"Channel {i}: replaced NaN with mean={mean_val:.4f}")
            else:
                mean_val = np.nanmean(data)
                data = np.where(np.isnan(data), mean_val, data)
            nan_report['replacement_value'] = 'channel_means'
            
        elif strategy == 'median':
            # Replace with per-channel median
            if len(data.shape) >= 2:
                for i in range(data.shape[1]):
                    channel = data[:, i]
                    if np.any(np.isnan(channel)):
                        median_val = np.nanmedian(channel)
                        data[:, i] = np.where(np.isnan(channel), median_val, channel)
            else:
                median_val = np.nanmedian(data)
                data = np.where(np.isnan(data), median_val, data)
            nan_report['replacement_value'] = 'channel_medians'
            
        elif strategy == 'interpolate':
            # Linear interpolation (best for time series)
            data = self._interpolate_nan(data)
            nan_report['replacement_value'] = 'interpolated'
            
        elif strategy == 'drop':
            # Remove samples containing NaN
            if len(data.shape) >= 2:
                valid_mask = ~np.any(np.isnan(data), axis=tuple(range(1, len(data.shape))))
                dropped = (~valid_mask).sum()
                data = data[valid_mask]
                nan_report['samples_dropped'] = int(dropped)
                if log_details:
                    logger.warning(f"Dropped {dropped} samples containing NaN")
            
        elif strategy == 'raise':
            raise ValueError(f"Data contains {nan_metrics['count']} NaN values")
            
        else:
            warnings.warn(f"Unknown NaN strategy: {strategy}, using 'mean'")
            return self._correct_nan(data, 'mean', nan_metrics, log_details)
            
        nan_report['count_after'] = np.isnan(data).sum()
        return data, nan_report
    
    def _correct_inf(
        self,
        data: np.ndarray,
        inf_metrics: Dict,
        log_details: bool
    ) -> Tuple[np.ndarray, Dict]:
        """Correct infinite values by clipping."""
        inf_report = {
            'type': 'inf_correction',
            'count_before': inf_metrics['count'],
            'percentage_before': inf_metrics['percentage']
        }
        
        # Determine clipping range from finite values
        finite_mask = np.isfinite(data)
        if np.any(finite_mask):
            finite_data = data[finite_mask]
            clip_min = np.percentile(finite_data, 0.1)
            clip_max = np.percentile(finite_data, 99.9)
        else:
            clip_min, clip_max = -1e6, 1e6
            
        # Apply clipping
        data = np.clip(data, clip_min, clip_max)
        
        inf_report['clip_range'] = [float(clip_min), float(clip_max)]
        inf_report['count_after'] = np.isinf(data).sum()
        
        if log_details:
            logger.info(f"Clipped {inf_metrics['count']} Inf values to [{clip_min:.2f}, {clip_max:.2f}]")
            
        return data, inf_report
    
    def _interpolate_nan(self, data: np.ndarray) -> np.ndarray:
        """Interpolate NaN values in time series data."""
        if len(data.shape) == 3:  # (B, C, T)
            for b in range(data.shape[0]):
                for c in range(data.shape[1]):
                    signal = data[b, c, :]
                    if np.any(np.isnan(signal)):
                        # Linear interpolation
                        x = np.arange(len(signal))
                        valid = ~np.isnan(signal)
                        if np.sum(valid) >= 2:
                            data[b, c, :] = np.interp(x, x[valid], signal[valid])
                        else:
                            # Fall back to mean if not enough points
                            data[b, c, :] = np.nan_to_num(signal, nan=np.nanmean(signal))
        return data
    
    def _log_correction_summary(self, report: Dict):
        """Log a summary of corrections applied."""
        logger.info(f"Data quality corrections for '{report['name']}':")
        logger.info(f"  Quality: {report['quality_before']:.1f} → {report['quality_after']:.1f}")
        
        for correction in report['corrections_applied']:
            if correction['type'] == 'nan_correction':
                logger.info(f"  NaN: {correction['count_before']} → {correction['count_after']} "
                           f"(strategy: {correction['strategy']})")
            elif correction['type'] == 'inf_correction':
                logger.info(f"  Inf: {correction['count_before']} → {correction['count_after']} "
                           f"(clipped to {correction['clip_range']})")
    
    def generate_report(self) -> str:
        """Generate comprehensive quality report from history."""
        if not self.correction_history:
            return "No corrections applied yet."
            
        report_lines = [
            "DATA QUALITY CORRECTION REPORT",
            "=" * 60,
            f"Total datasets processed: {len(self.correction_history)}",
            ""
        ]
        
        for i, correction in enumerate(self.correction_history, 1):
            report_lines.extend([
                f"\n{i}. {correction['name']}",
                f"   Time: {correction['timestamp']}",
                f"   Shape: {correction['original_shape']} → {correction['final_shape']}",
                f"   Quality: {correction['quality_before']:.1f} → {correction['quality_after']:.1f}",
            ])
            
            for applied in correction['corrections_applied']:
                if applied['type'] == 'nan_correction':
                    report_lines.append(
                        f"   - NaN: {applied['percentage_before']:.2f}% corrected with '{applied['strategy']}'"
                    )
                elif applied['type'] == 'inf_correction':
                    report_lines.append(
                        f"   - Inf: {applied['count_before']} values clipped"
                    )
                    
        return "\n".join(report_lines)

# Example usage
if __name__ == "__main__":
    # Create sample data with quality issues
    np.random.seed(42)
    data = np.random.randn(100, 9, 100)
    
    # Inject some NaN values (3%)
    nan_mask = np.random.random(data.shape) < 0.03
    data[nan_mask] = np.nan
    
    # Inject some Inf values (0.5%)
    inf_mask = np.random.random(data.shape) < 0.005
    data[inf_mask] = np.inf
    
    # Create handler
    handler = DataQualityHandler()
    
    # Analyze quality
    print("BEFORE CORRECTION:")
    metrics = handler.analyze_quality(data, "sample_data")
    print(f"Quality Score: {metrics['quality_score']:.1f}")
    print(f"NaN: {metrics['nan']['percentage']:.2f}%")
    print(f"Inf: {metrics['inf']['percentage']:.2f}%")
    print(f"Recommended: {metrics['recommended_action']}")
    
    # Apply corrections
    print("\nAPPLYING CORRECTIONS...")
    corrected_data, report = handler.correct_data(data, name="sample_data")
    
    print("\nAFTER CORRECTION:")
    print(f"Shape: {corrected_data.shape}")
    print(f"NaN remaining: {np.isnan(corrected_data).sum()}")
    print(f"Inf remaining: {np.isinf(corrected_data).sum()}")
    
    # Generate report
    print("\n" + handler.generate_report())