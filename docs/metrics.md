# Metrics and Calibration Documentation

The metrics system provides comprehensive evaluation with calibration analysis, reliability diagrams, and behavioral quality assessment for production-ready behavioral analysis models.

## Overview

Key features:
- **Standard metrics**: Accuracy, Macro-F1, ECE, MCE, Coverage
- **Calibration analysis**: Reliability diagrams, confidence histograms  
- **Behavioral metrics**: Transition rates, dwell times, entropy
- **Quality assurance**: NaN/Inf detection, outlier identification
- **Bundle generation**: Complete evaluation packages in `reports/EXP_HASH/`

## Architecture

```
Predictions + Ground Truth → Metrics Calculator → Comprehensive Results
                                    ↓                      ↓
                           Calibration Analyzer → Reliability Diagrams
                                    ↓                      ↓
                            Bundle Generator → reports/EXP_HASH/
```

## Core Components

### Metrics Calculator

```python
from conv2d.metrics.core import MetricsCalculator, MetricsResult

calculator = MetricsCalculator()
metrics = calculator.compute_all(y_true, y_pred, y_prob)

print(f"Accuracy: {metrics.accuracy:.1%}")
print(f"Macro F1: {metrics.macro_f1:.3f}")
print(f"ECE: {metrics.ece:.3f}")
```

### Calibration Analyzer

```python
from conv2d.metrics.calibration import CalibrationAnalyzer

analyzer = CalibrationAnalyzer(n_bins=10)
calibration = analyzer.analyze(y_true, y_pred, y_prob)

print(f"ECE: {calibration.ece:.3f}")
print(f"MCE: {calibration.mce:.3f}")
print(f"Brier Score: {calibration.brier:.3f}")
```

### Bundle Generator

```python
from conv2d.metrics.bundle import BundleGenerator

generator = BundleGenerator(output_base="reports")
bundle_path = generator.create_bundle(
    y_true=y_true,
    y_pred=y_pred, 
    y_prob=y_prob,
    config=config,
    exp_hash="a1b2c3d4",
)
```

## Standard Metrics

### Classification Metrics

```python
@dataclass
class MetricsResult:
    """Comprehensive metrics result container."""
    
    # Core classification metrics
    accuracy: float           # Overall accuracy
    macro_f1: float          # Macro-averaged F1 score
    per_class_f1: List[float] # F1 per class
    
    # Calibration metrics
    ece: float               # Expected Calibration Error
    mce: float               # Maximum Calibration Error
    brier: float             # Brier score
    
    # Coverage and reliability
    coverage: float          # Prediction coverage at threshold
    confidence_mean: float   # Mean confidence score
    confidence_std: float    # Confidence standard deviation
    
    # Behavioral metrics
    motif_count: int         # Number of unique behavioral motifs
    transition_rate: float   # Transitions per timestep
    mean_dwell_time: float   # Average state duration
    
    # Quality metrics
    code_usage_percent: float # Codebook utilization rate
    perplexity: float        # Effective codebook size
    entropy: float           # Behavioral entropy
```

### Metrics Calculation

```python
def compute_all(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    **kwargs
) -> MetricsResult:
    """Compute comprehensive metrics."""
    
    # Core classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()
    
    # Calibration metrics (if probabilities provided)
    if y_prob is not None:
        calibration = self.calibration_analyzer.analyze(y_true, y_pred, y_prob)
        ece = calibration.ece
        mce = calibration.mce
        brier = calibration.brier
        coverage = self._compute_coverage(y_true, y_pred, y_prob)
        confidence_mean = y_prob.mean()
        confidence_std = y_prob.std()
    else:
        ece = mce = brier = coverage = confidence_mean = confidence_std = 0.0
    
    # Behavioral metrics
    motif_count = len(np.unique(y_pred))
    transition_rate = self._compute_transition_rate(y_pred)
    mean_dwell_time = self._compute_mean_dwell_time(y_pred)
    
    # Quality metrics
    code_usage_percent = motif_count / kwargs.get('total_codes', motif_count) * 100
    perplexity = kwargs.get('perplexity', 0.0)
    entropy = self._compute_entropy(y_pred)
    
    return MetricsResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        ece=ece,
        mce=mce,
        brier=brier,
        coverage=coverage,
        confidence_mean=confidence_mean,
        confidence_std=confidence_std,
        motif_count=motif_count,
        transition_rate=transition_rate,
        mean_dwell_time=mean_dwell_time,
        code_usage_percent=code_usage_percent,
        perplexity=perplexity,
        entropy=entropy,
    )
```

## Calibration Analysis

### Expected Calibration Error (ECE)

```python
def compute_ece(
    self,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error."""
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    total_samples = len(y_prob)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this confidence bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Compute accuracy in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            
            # Average confidence in this bin
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece
```

### Maximum Calibration Error (MCE)

```python
def compute_mce(
    self,
    y_true: np.ndarray, 
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Maximum Calibration Error."""
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1] 
    bin_uppers = bin_boundaries[1:]
    
    max_calibration_error = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_calibration_error = max(max_calibration_error, calibration_error)
    
    return max_calibration_error
```

### Reliability Diagrams

```python
def plot_reliability_diagram(
    self,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate reliability diagram."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute bin statistics
    bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(self.n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            bin_accuracies.append(y_true[in_bin].mean())
            bin_confidences.append(y_prob[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, 
           edgecolor='black', label='Accuracy')
    ax1.plot(bin_confidences, bin_accuracies, 'ro-', label='Model')
    
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Reliability Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Confidence histogram
    ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, 
           edgecolor='black', color='skyblue')
    ax2.set_xlabel('Mean Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Histogram')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
```

## Behavioral Metrics

### Transition Analysis

```python
def _compute_transition_rate(self, y_pred: np.ndarray) -> float:
    """Compute transition rate across sequences."""
    
    if y_pred.ndim == 1:
        # Single sequence
        transitions = np.sum(np.diff(y_pred) != 0)
        return transitions / (len(y_pred) - 1)
    
    elif y_pred.ndim == 2:
        # Multiple sequences (B, T)
        total_transitions = 0
        total_timesteps = 0
        
        for b in range(y_pred.shape[0]):
            transitions = np.sum(np.diff(y_pred[b]) != 0)
            total_transitions += transitions
            total_timesteps += y_pred.shape[1] - 1
        
        return total_transitions / total_timesteps if total_timesteps > 0 else 0.0
    
    else:
        raise ValueError("y_pred must be 1D or 2D")

def _compute_mean_dwell_time(self, y_pred: np.ndarray) -> float:
    """Compute mean dwell time for behavioral states."""
    
    if y_pred.ndim == 1:
        segments = self._find_segments(y_pred)
        dwell_times = [end - start for start, end, _ in segments]
        return np.mean(dwell_times) if dwell_times else 0.0
    
    elif y_pred.ndim == 2:
        all_dwell_times = []
        
        for b in range(y_pred.shape[0]):
            segments = self._find_segments(y_pred[b])
            dwell_times = [end - start for start, end, _ in segments]
            all_dwell_times.extend(dwell_times)
        
        return np.mean(all_dwell_times) if all_dwell_times else 0.0
    
    else:
        raise ValueError("y_pred must be 1D or 2D")

def _find_segments(self, sequence: np.ndarray) -> List[Tuple[int, int, int]]:
    """Find contiguous segments in sequence."""
    
    segments = []
    if len(sequence) == 0:
        return segments
    
    start = 0
    current_state = sequence[0]
    
    for i in range(1, len(sequence)):
        if sequence[i] != current_state:
            # End of current segment
            segments.append((start, i, current_state))
            start = i
            current_state = sequence[i]
    
    # Final segment
    segments.append((start, len(sequence), current_state))
    
    return segments
```

### Entropy and Complexity

```python
def _compute_entropy(self, y_pred: np.ndarray) -> float:
    """Compute Shannon entropy of behavioral distribution."""
    
    unique_states, counts = np.unique(y_pred, return_counts=True)
    probabilities = counts / counts.sum()
    
    # Shannon entropy: -sum(p * log(p))
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
    
    return entropy

def compute_behavioral_complexity(
    self,
    y_pred: np.ndarray,
    window_size: int = 50,
) -> Dict[str, float]:
    """Compute behavioral complexity metrics."""
    
    metrics = {}
    
    # Overall entropy
    metrics['entropy'] = self._compute_entropy(y_pred)
    
    # Local entropy (windowed)
    if y_pred.ndim == 2:  # (B, T)
        local_entropies = []
        B, T = y_pred.shape
        
        for b in range(B):
            for t in range(0, T - window_size + 1, window_size // 2):
                window = y_pred[b, t:t + window_size]
                local_entropy = self._compute_entropy(window)
                local_entropies.append(local_entropy)
        
        metrics['local_entropy_mean'] = np.mean(local_entropies)
        metrics['local_entropy_std'] = np.std(local_entropies)
    
    # Transition entropy
    if y_pred.ndim == 2:
        transition_pairs = []
        for b in range(y_pred.shape[0]):
            for t in range(y_pred.shape[1] - 1):
                transition_pairs.append((y_pred[b, t], y_pred[b, t + 1]))
        
        if transition_pairs:
            unique_transitions, counts = np.unique(
                transition_pairs, axis=0, return_counts=True
            )
            transition_probs = counts / counts.sum()
            transition_entropy = -np.sum(
                transition_probs * np.log2(transition_probs + 1e-8)
            )
            metrics['transition_entropy'] = transition_entropy
    
    return metrics
```

## Quality Assurance

### Data Quality Checks

```python
class QualityAssurance:
    """Quality assurance for predictions and metrics."""
    
    def __init__(self, outlier_threshold: float = 5.0):
        self.outlier_threshold = outlier_threshold
    
    def check_data_quality(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Comprehensive data quality check."""
        
        issues = {
            'nan_in_true': np.isnan(y_true).any(),
            'nan_in_pred': np.isnan(y_pred).any(),
            'inf_in_true': np.isinf(y_true).any(),
            'inf_in_pred': np.isinf(y_pred).any(),
            'shape_mismatch': y_true.shape != y_pred.shape,
            'negative_labels': (y_true < 0).any() or (y_pred < 0).any(),
        }
        
        if y_prob is not None:
            issues.update({
                'nan_in_prob': np.isnan(y_prob).any(),
                'inf_in_prob': np.isinf(y_prob).any(),
                'prob_out_of_range': (y_prob < 0).any() or (y_prob > 1).any(),
                'prob_shape_mismatch': y_prob.shape[0] != y_true.shape[0],
            })
        
        # Check for outliers using IQR method
        outliers = self._detect_outliers(y_pred)
        issues['outliers_detected'] = len(outliers) > 0
        issues['outlier_indices'] = outliers.tolist() if len(outliers) > 0 else []
        
        # Summary
        issues['has_critical_issues'] = any([
            issues['nan_in_true'], issues['nan_in_pred'],
            issues['inf_in_true'], issues['inf_in_pred'],
            issues['shape_mismatch'],
        ])
        
        return issues
    
    def _detect_outliers(self, data: np.ndarray) -> np.ndarray:
        """Detect outliers using IQR method."""
        
        if data.dtype.kind in ['i', 'u']:  # Integer types (class labels)
            # For class labels, outliers are values way outside expected range
            unique_labels = np.unique(data)
            if len(unique_labels) > 1:
                label_range = unique_labels.max() - unique_labels.min()
                threshold = unique_labels.max() + label_range * 2
                outliers = np.where(data > threshold)[0]
            else:
                outliers = np.array([])
        else:
            # For continuous values, use IQR
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        
        return outliers
```

### Metric Validation

```python
def validate_metrics(self, metrics: MetricsResult) -> Dict[str, bool]:
    """Validate computed metrics for sanity."""
    
    validations = {
        'accuracy_in_range': 0.0 <= metrics.accuracy <= 1.0,
        'f1_in_range': 0.0 <= metrics.macro_f1 <= 1.0,
        'ece_in_range': 0.0 <= metrics.ece <= 1.0,
        'mce_in_range': 0.0 <= metrics.mce <= 1.0,
        'mce_geq_ece': metrics.mce >= metrics.ece,  # MCE should be ≥ ECE
        'brier_in_range': 0.0 <= metrics.brier <= 1.0,
        'confidence_in_range': 0.0 <= metrics.confidence_mean <= 1.0,
        'transition_rate_reasonable': 0.0 <= metrics.transition_rate <= 1.0,
        'dwell_time_positive': metrics.mean_dwell_time > 0,
        'motif_count_positive': metrics.motif_count > 0,
        'perplexity_positive': metrics.perplexity >= 0,
        'entropy_non_negative': metrics.entropy >= 0,
    }
    
    # Overall validation
    validations['all_valid'] = all(validations.values())
    
    return validations
```

## Bundle Generation

### Evaluation Bundle Structure

```python
def create_bundle(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
    exp_hash: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """Create comprehensive evaluation bundle."""
    
    # Generate experiment hash if not provided
    if exp_hash is None:
        exp_hash = self._generate_exp_hash(y_true, y_pred, config)
    
    # Create bundle directory
    bundle_dir = self.output_base / exp_hash
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute all metrics
    calculator = MetricsCalculator()
    metrics = calculator.compute_all(y_true, y_pred, y_prob)
    
    # Quality assurance
    qa = QualityAssurance()
    quality_issues = qa.check_data_quality(y_true, y_pred, y_prob)
    metric_validation = qa.validate_metrics(metrics)
    
    # Save metrics
    self._save_metrics(bundle_dir / "metrics.json", metrics)
    self._save_quality_report(bundle_dir / "quality.json", quality_issues)
    self._save_validation_report(bundle_dir / "validation.json", metric_validation)
    
    # Generate visualizations
    if y_prob is not None:
        self._generate_calibration_plots(
            bundle_dir, y_true, y_pred, y_prob
        )
    
    self._generate_confusion_matrix(bundle_dir, y_true, y_pred)
    self._generate_behavioral_analysis(bundle_dir, y_pred)
    
    # Save configuration and metadata
    if config is not None:
        self._save_config(bundle_dir / "config.yaml", config)
    
    if metadata is not None:
        self._save_metadata(bundle_dir / "metadata.json", metadata)
    
    # Generate bundle manifest
    manifest = self._create_manifest(bundle_dir)
    self._save_manifest(bundle_dir / "manifest.json", manifest)
    
    return str(bundle_dir)
```

### Bundle Contents

```
reports/EXP_HASH/
├── metrics.json              # All computed metrics
├── quality.json              # Data quality report
├── validation.json           # Metric validation results
├── config.yaml               # Experiment configuration
├── metadata.json             # Additional metadata
├── manifest.json             # Bundle contents manifest
├── plots/
│   ├── reliability_diagram.png
│   ├── confidence_histogram.png
│   ├── confusion_matrix.png
│   ├── behavioral_timeline.png
│   └── entropy_analysis.png
└── raw_data/
    ├── y_true.npy            # Ground truth labels
    ├── y_pred.npy            # Predictions
    └── y_prob.npy            # Probabilities (if available)
```

## Usage Examples

### Basic Evaluation

```python
import numpy as np
from conv2d.metrics.core import MetricsCalculator
from conv2d.metrics.bundle import BundleGenerator

# Sample data
y_true = np.random.randint(0, 4, 1000)
y_pred = np.random.randint(0, 4, 1000)
y_prob = np.random.rand(1000)

# Compute metrics
calculator = MetricsCalculator()
metrics = calculator.compute_all(y_true, y_pred, y_prob)

print(f"Accuracy: {metrics.accuracy:.1%}")
print(f"Macro F1: {metrics.macro_f1:.3f}")
print(f"ECE: {metrics.ece:.3f}")
print(f"Behavioral entropy: {metrics.entropy:.2f}")

# Generate evaluation bundle
generator = BundleGenerator(output_base="reports")
bundle_path = generator.create_bundle(y_true, y_pred, y_prob)
print(f"Evaluation bundle: {bundle_path}")
```

### Calibration Analysis

```python
from conv2d.metrics.calibration import CalibrationAnalyzer

# Analyze calibration
analyzer = CalibrationAnalyzer(n_bins=10)
calibration = analyzer.analyze(y_true, y_pred, y_prob)

print(f"Expected Calibration Error: {calibration.ece:.4f}")
print(f"Maximum Calibration Error: {calibration.mce:.4f}")
print(f"Brier Score: {calibration.brier:.4f}")

# Generate reliability diagram
fig = analyzer.plot_reliability_diagram(
    y_true, y_prob, save_path="reliability.png"
)
```

### Behavioral Analysis

```python
from conv2d.metrics.core import MetricsCalculator

# Sample temporal data (batch, time)
y_pred_temporal = np.random.randint(0, 4, (32, 100))

calculator = MetricsCalculator()

# Compute behavioral metrics
transition_rate = calculator._compute_transition_rate(y_pred_temporal)
mean_dwell = calculator._compute_mean_dwell_time(y_pred_temporal)
complexity = calculator.compute_behavioral_complexity(y_pred_temporal)

print(f"Transition rate: {transition_rate:.3f}")
print(f"Mean dwell time: {mean_dwell:.1f} frames")
print(f"Behavioral entropy: {complexity['entropy']:.2f}")
print(f"Local entropy (mean): {complexity['local_entropy_mean']:.2f}")
```

### Pipeline Integration

```python
from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.gmm import GMMClusterer
from conv2d.temporal.median import MedianHysteresisPolicy
from conv2d.metrics.bundle import BundleGenerator

# Full pipeline evaluation
x = torch.randn(100, 9, 2, 100, dtype=torch.float32)
y_true = np.random.randint(0, 4, 100)

# Encode features
result = encode_fsq(x)

# Cluster behaviors
clusterer = GMMClusterer(random_state=42)
y_pred_raw = clusterer.fit_predict(result.features.numpy(), k=4)

# Apply temporal smoothing
policy = MedianHysteresisPolicy(min_dwell=3)
y_pred_smooth = policy.smooth(y_pred_raw.reshape(1, -1))[0]

# Generate comprehensive evaluation
generator = BundleGenerator()
bundle_path = generator.create_bundle(
    y_true=y_true,
    y_pred=y_pred_smooth,
    config={
        "fsq": {"levels": [8, 6, 5], "embedding_dim": 64},
        "clustering": {"algorithm": "gmm", "k": 4},
        "temporal": {"policy": "median_hysteresis", "min_dwell": 3},
    },
    metadata={
        "perplexity": float(result.perplexity),
        "code_usage": len(np.unique(result.codes.numpy())) / 240 * 100,
    }
)

print(f"Complete evaluation bundle: {bundle_path}")
```

## Best Practices

1. **Always validate inputs**: Check for NaN, Inf, and shape mismatches before computing metrics
2. **Use comprehensive metrics**: Don't rely on accuracy alone - include calibration and behavioral metrics
3. **Generate evaluation bundles**: Create complete packages with visualizations for reproducibility
4. **Monitor calibration**: ECE and MCE are critical for deployment confidence
5. **Analyze behavioral patterns**: Transition rates and dwell times reveal model behavior
6. **Check metric consistency**: MCE should be ≥ ECE, all probabilities in [0,1]
7. **Document everything**: Include configuration, metadata, and quality reports
8. **Compare across runs**: Use bundle comparison tools to track improvements

This metrics system provides comprehensive evaluation capabilities that ensure production-ready models with proper uncertainty quantification and behavioral analysis.