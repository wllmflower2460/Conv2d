"""Quality Assurance gates for Conv2d pipeline."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import IntEnum


class ExitCode(IntEnum):
    """Standard exit codes for QA failures."""
    SUCCESS = 0
    GENERAL_ERROR = 1
    DATA_QUALITY_FAILURE = 2
    MODEL_CONVERGENCE_FAILURE = 3
    CONFIGURATION_ERROR = 4
    DEPLOYMENT_CHECK_FAILURE = 5


@dataclass
class QAResult:
    """Result of a quality assurance check."""
    passed: bool
    issues: List[str]
    metrics: Dict[str, Any]
    exit_code: ExitCode = ExitCode.SUCCESS
    
    def __post_init__(self):
        """Set appropriate exit code based on pass/fail."""
        if not self.passed and self.exit_code == ExitCode.SUCCESS:
            self.exit_code = ExitCode.GENERAL_ERROR


class QAGate:
    """Quality assurance gate for pipeline stages."""
    
    def __init__(self, name: str):
        self.name = name
        self.thresholds = self._get_default_thresholds()
    
    def _get_default_thresholds(self) -> Dict[str, Any]:
        """Get default QA thresholds by gate type."""
        thresholds = {
            "data_quality": {
                "max_nan_percent": 5.0,
                "max_outlier_percent": 10.0,
                "min_samples": 1000,
                "max_discontinuities": 100
            },
            "model_convergence": {
                "min_accuracy": 0.60,
                "min_f1": 0.55,
                "max_loss": 2.0,
                "min_improvement": 0.001
            },
            "calibration": {
                "max_ece": 0.10,
                "max_brier": 0.25,
                "min_coverage": 0.95
            },
            "deployment": {
                "max_model_size_mb": 10,
                "max_inference_ms": 100,
                "min_onnx_opset": 11,
                "required_formats": ["onnx", "coreml"]
            },
            "behavioral": {
                "min_motif_coverage": 0.90,
                "max_transition_rate": 0.5,
                "min_dwell_frames": 10,
                "max_perplexity": 10.0
            }
        }
        return thresholds.get(self.name, {})
    
    def check_data_quality(
        self, 
        nan_percent: float,
        outlier_percent: float,
        n_samples: int,
        discontinuities: int
    ) -> QAResult:
        """Check data quality metrics."""
        issues = []
        metrics = {
            "nan_percent": nan_percent,
            "outlier_percent": outlier_percent, 
            "n_samples": n_samples,
            "discontinuities": discontinuities
        }
        
        t = self.thresholds
        
        if nan_percent > t["max_nan_percent"]:
            issues.append(
                f"NaN rate {nan_percent:.1f}% exceeds threshold {t['max_nan_percent']}%"
            )
        
        if outlier_percent > t["max_outlier_percent"]:
            issues.append(
                f"Outlier rate {outlier_percent:.1f}% exceeds threshold {t['max_outlier_percent']}%"
            )
        
        if n_samples < t["min_samples"]:
            issues.append(
                f"Sample count {n_samples} below minimum {t['min_samples']}"
            )
        
        if discontinuities > t["max_discontinuities"]:
            issues.append(
                f"Discontinuities {discontinuities} exceeds threshold {t['max_discontinuities']}"
            )
        
        return QAResult(
            passed=len(issues) == 0,
            issues=issues,
            metrics=metrics,
            exit_code=ExitCode.DATA_QUALITY_FAILURE if issues else ExitCode.SUCCESS
        )
    
    def check_model_convergence(
        self,
        accuracy: float,
        f1_score: float,
        final_loss: float,
        improvement: float
    ) -> QAResult:
        """Check model training convergence."""
        issues = []
        metrics = {
            "accuracy": accuracy,
            "f1_score": f1_score,
            "final_loss": final_loss,
            "improvement": improvement
        }
        
        t = self.thresholds
        
        if accuracy < t["min_accuracy"]:
            issues.append(
                f"Accuracy {accuracy:.2%} below minimum {t['min_accuracy']:.2%}"
            )
        
        if f1_score < t["min_f1"]:
            issues.append(
                f"F1 score {f1_score:.3f} below minimum {t['min_f1']:.3f}"
            )
        
        if final_loss > t["max_loss"]:
            issues.append(
                f"Final loss {final_loss:.4f} exceeds threshold {t['max_loss']:.4f}"
            )
        
        if improvement < t["min_improvement"]:
            issues.append(
                f"Improvement {improvement:.4f} below minimum {t['min_improvement']:.4f}"
            )
        
        return QAResult(
            passed=len(issues) == 0,
            issues=issues,
            metrics=metrics,
            exit_code=ExitCode.MODEL_CONVERGENCE_FAILURE if issues else ExitCode.SUCCESS
        )
    
    def check_calibration(
        self,
        ece: float,
        brier_score: float,
        coverage: float
    ) -> QAResult:
        """Check model calibration metrics."""
        issues = []
        metrics = {
            "ece": ece,
            "brier_score": brier_score,
            "coverage": coverage
        }
        
        t = self.thresholds
        
        if ece > t["max_ece"]:
            issues.append(
                f"ECE {ece:.3f} exceeds threshold {t['max_ece']:.3f}"
            )
        
        if brier_score > t["max_brier"]:
            issues.append(
                f"Brier score {brier_score:.3f} exceeds threshold {t['max_brier']:.3f}"
            )
        
        if coverage < t["min_coverage"]:
            issues.append(
                f"Coverage {coverage:.2%} below minimum {t['min_coverage']:.2%}"
            )
        
        return QAResult(
            passed=len(issues) == 0,
            issues=issues,
            metrics=metrics
        )
    
    def check_deployment_readiness(
        self,
        model_size_mb: float,
        inference_ms: float,
        onnx_opset: int,
        available_formats: List[str]
    ) -> QAResult:
        """Check deployment readiness criteria."""
        issues = []
        metrics = {
            "model_size_mb": model_size_mb,
            "inference_ms": inference_ms,
            "onnx_opset": onnx_opset,
            "formats": available_formats
        }
        
        t = self.thresholds
        
        if model_size_mb > t["max_model_size_mb"]:
            issues.append(
                f"Model size {model_size_mb:.1f}MB exceeds limit {t['max_model_size_mb']}MB"
            )
        
        if inference_ms > t["max_inference_ms"]:
            issues.append(
                f"Inference time {inference_ms:.1f}ms exceeds limit {t['max_inference_ms']}ms"
            )
        
        if onnx_opset < t["min_onnx_opset"]:
            issues.append(
                f"ONNX opset {onnx_opset} below minimum {t['min_onnx_opset']}"
            )
        
        missing_formats = set(t["required_formats"]) - set(available_formats)
        if missing_formats:
            issues.append(
                f"Missing required formats: {', '.join(missing_formats)}"
            )
        
        return QAResult(
            passed=len(issues) == 0,
            issues=issues,
            metrics=metrics,
            exit_code=ExitCode.DEPLOYMENT_CHECK_FAILURE if issues else ExitCode.SUCCESS
        )
    
    def check_behavioral_metrics(
        self,
        motif_coverage: float,
        transition_rate: float,
        mean_dwell: float,
        perplexity: float
    ) -> QAResult:
        """Check behavioral analysis metrics."""
        issues = []
        metrics = {
            "motif_coverage": motif_coverage,
            "transition_rate": transition_rate,
            "mean_dwell": mean_dwell,
            "perplexity": perplexity
        }
        
        t = self.thresholds
        
        if motif_coverage < t["min_motif_coverage"]:
            issues.append(
                f"Motif coverage {motif_coverage:.2%} below minimum {t['min_motif_coverage']:.2%}"
            )
        
        if transition_rate > t["max_transition_rate"]:
            issues.append(
                f"Transition rate {transition_rate:.3f} exceeds threshold {t['max_transition_rate']:.3f}"
            )
        
        if mean_dwell < t["min_dwell_frames"]:
            issues.append(
                f"Mean dwell {mean_dwell:.1f} frames below minimum {t['min_dwell_frames']} frames"
            )
        
        if perplexity > t["max_perplexity"]:
            issues.append(
                f"Perplexity {perplexity:.2f} exceeds threshold {t['max_perplexity']:.2f}"
            )
        
        return QAResult(
            passed=len(issues) == 0,
            issues=issues,
            metrics=metrics
        )