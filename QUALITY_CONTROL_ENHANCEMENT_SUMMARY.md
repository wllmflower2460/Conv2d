# Quality Control Enhancement Summary

## Overview

The `movement_diagnostics.py` file has been significantly enhanced with comprehensive quality control measures to prevent GIGO (Garbage In, Garbage Out) issues. The new `QualityControl` class provides multiple layers of validation and monitoring for the Conv2d-VQ-HDP-HSMM pipeline.

## Key Enhancements

### 1. QualityControl Class
A comprehensive quality control system that implements four layers of validation:

- **Input Validation**: Shape, type, range, and NaN/Inf checks
- **Codebook Health Monitoring**: Usage, diversity, transitions, and dead code detection
- **Signal Quality Assessment**: SNR, frequency content, and stationarity analysis  
- **Data Consistency Checks**: Gap analysis, temporal correlations, and outlier detection

### 2. QualityThresholds Dataclass
Configurable thresholds for all quality metrics:

```python
@dataclass
class QualityThresholds:
    # Input validation thresholds
    max_nan_percentage: float = 10.0
    max_inf_count: int = 0
    min_signal_std: float = 0.01
    max_signal_std: float = 100.0
    expected_shape: Tuple[int, int, int, int] = (None, 9, 2, 100)
    
    # Codebook health thresholds  
    min_codebook_usage: float = 0.1
    min_perplexity: float = 4.0
    max_dead_codes_ratio: float = 0.5
    min_entropy: float = 2.0
    max_transition_rate: float = 0.8
    
    # Signal quality thresholds
    min_snr_db: float = 0.0
    max_autocorr_lag1: float = 0.95
    min_frequency_diversity: float = 0.1
    
    # Data consistency thresholds
    max_gap_length: int = 50
    max_gap_percentage: float = 20.0
    min_temporal_correlation: float = 0.1
```

### 3. QualityReport Dataclass
Structured quality reports with pass/fail status and actionable recommendations:

```python
@dataclass
class QualityReport:
    timestamp: str
    data_shape: Tuple[int, ...]
    overall_pass: bool
    input_validation: Dict[str, Any]
    codebook_health: Optional[Dict[str, Any]] = None
    signal_quality: Dict[str, Any] = None
    data_consistency: Dict[str, Any] = None
    detailed_metrics: Dict[str, Any] = None
    recommendations: List[str] = None
```

## Enhanced BehavioralDataDiagnostics

### Updated Constructor
```python
def __init__(self, 
             sampling_rate: float = 100.0,
             output_dir: Optional[Path] = None,
             quality_thresholds: Optional[QualityThresholds] = None,
             enable_quality_gates: bool = True,
             strict_quality_mode: bool = False):
```

### New Methods

#### `run_quality_gates_only()`
Runs only quality gates without the full diagnostic suite for fast validation.

#### `get_quality_control_status()` 
Returns current quality control system status and trends.

#### `update_quality_thresholds()`
Dynamically updates quality thresholds during runtime.

#### `export_quality_trends_report()`
Exports comprehensive quality trends analysis to JSON.

### Enhanced `run_full_diagnostic()`
Now includes quality gates as the first step:

```python
def run_full_diagnostic(self, data: torch.Tensor,
                      labels: Optional[torch.Tensor] = None,
                      codebook_info: Optional[Dict[str, Any]] = None,
                      indices_history: Optional[List[torch.Tensor]] = None,
                      save_report: bool = True) -> Dict[str, Any]:
```

## Quality Validation Layers

### 1. Input Validation
- **Shape Validation**: Ensures expected tensor dimensions (B, 9, 2, 100)
- **Data Type Validation**: Checks for floating-point tensors
- **NaN/Inf Detection**: Validates data completeness and numerical stability
- **Signal Variance**: Ensures reasonable signal amplitude ranges

### 2. Codebook Health Monitoring
- **Usage Monitoring**: Tracks fraction of codebook entries being used
- **Perplexity Analysis**: Measures codebook diversity and utilization
- **Dead Code Detection**: Identifies and reports unused codebook entries
- **Transition Rate Analysis**: Monitors temporal code switching patterns
- **Entropy Measurement**: Quantifies information content distribution

### 3. Signal Quality Assessment
- **SNR Estimation**: Signal-to-noise ratio calculation
- **Autocorrelation Analysis**: Temporal dependency assessment
- **Frequency Diversity**: Spectral content distribution analysis
- **Stationarity Checks**: Signal consistency over time

### 4. Data Consistency Checks
- **Gap Analysis**: Detection and measurement of data discontinuities
- **Temporal Correlation**: Cross-time consistency validation
- **Outlier Detection**: Statistical anomaly identification
- **Range Validation**: Value distribution assessment

## Integration with VQ Models

The quality control system integrates seamlessly with VQ models by accepting codebook information:

```python
# Example VQ integration
codebook_info = {
    'perplexity': vq_output['perplexity'],
    'usage': vq_output['usage'], 
    'entropy': vq_output['entropy'],
    'dead_codes': vq_output['dead_codes'],
    'total_codes': 512,
    'indices': vq_output['indices']
}

quality_report = diagnostics.run_quality_gates_only(
    data, 
    codebook_info=codebook_info,
    indices_history=previous_indices
)
```

## GIGO Prevention Workflow

1. **Pre-processing Validation**: Check input data quality before any processing
2. **Real-time Monitoring**: Continuous quality assessment during training/inference
3. **Threshold Enforcement**: Automatic rejection of data below quality standards
4. **Actionable Feedback**: Specific recommendations for quality improvement
5. **Trend Analysis**: Long-term quality monitoring and degradation detection

## Usage Examples

### Basic Quality Control
```python
from preprocessing.movement_diagnostics import BehavioralDataDiagnostics

# Initialize with quality control
diagnostics = BehavioralDataDiagnostics(
    enable_quality_gates=True,
    strict_quality_mode=False
)

# Run quality gates
quality_report = diagnostics.run_quality_gates_only(data, codebook_info)

if not quality_report.overall_pass:
    print("Quality gates failed!")
    for rec in quality_report.recommendations:
        print(f"- {rec}")
```

### Custom Thresholds
```python
from preprocessing.movement_diagnostics import QualityThresholds

# Define stricter thresholds for production
production_thresholds = QualityThresholds(
    max_nan_percentage=2.0,
    min_codebook_usage=0.8,
    min_perplexity=8.0,
    min_snr_db=10.0
)

diagnostics = BehavioralDataDiagnostics(
    quality_thresholds=production_thresholds,
    strict_quality_mode=True
)
```

### Training Integration
```python
def training_step(data, model):
    # Quality gate before processing
    quality_report = diagnostics.run_quality_gates_only(data)
    
    if not quality_report.overall_pass:
        # Skip bad batches or apply preprocessing
        return None
        
    # Proceed with training
    return model(data)
```

## Report Outputs

### JSON Quality Reports
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "overall_pass": false,
  "input_validation": {
    "pass": false,
    "failures": ["NaN percentage 15.5% exceeds threshold 10.0%"]
  },
  "codebook_health": {
    "pass": false,
    "metrics": {
      "perplexity": 3.2,
      "usage": 0.4,
      "dead_codes_ratio": 0.6
    }
  },
  "recommendations": [
    "Apply data cleaning and interpolation to reduce NaN values",
    "Adjust VQ parameters to improve codebook diversity"
  ]
}
```

### Quality Trends Reports
- Historical pass/fail rates
- Average quality metrics over time
- Degradation pattern detection
- Threshold effectiveness analysis

## Benefits

1. **GIGO Prevention**: Automated detection and rejection of poor-quality data
2. **Model Health Monitoring**: Continuous assessment of VQ codebook health
3. **Actionable Insights**: Specific recommendations for quality improvement
4. **Integration Ready**: Seamless integration with existing diagnostic workflows
5. **Configurable Standards**: Adjustable thresholds for different use cases
6. **Comprehensive Reporting**: Detailed JSON reports for analysis and debugging

## Files Modified/Created

- **Enhanced**: `/preprocessing/movement_diagnostics.py` - Main enhancement with new classes
- **Created**: `/example_quality_control.py` - Demonstration script
- **Created**: `/QUALITY_CONTROL_ENHANCEMENT_SUMMARY.md` - This documentation

The enhanced quality control system provides robust GIGO prevention while maintaining full backward compatibility with existing diagnostic workflows.