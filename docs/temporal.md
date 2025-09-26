# Temporal Policies Documentation

The temporal smoothing system provides configurable policies for eliminating artifacts and enforcing realistic behavioral dynamics with min-dwell guarantees and hysteresis-based stability.

## Overview

Key features:
- **Policy pattern**: Pluggable algorithms (Median, HSMM)  
- **Min-dwell enforcement**: Eliminates 1-2 frame flickers
- **Hysteresis smoothing**: Enter/exit thresholds prevent oscillation
- **Runtime configuration**: Switch policies via config
- **Transition monotonicity**: Smoothing never increases transitions

## Architecture

```
Raw Labels (B, T) → Temporal Policy → Smoothed Labels (B, T)
                        ↓
                 Min-dwell Check → Hysteresis → State Persistence
                        ↓              ↓              ↓
                 Artifact Filter  Stability     Long-term States
```

## Core Components

### Abstract Base Class

```python
from conv2d.temporal.interface import TemporalPolicy

class TemporalPolicy(ABC):
    """Abstract base class for temporal smoothing policies."""
    
    @abstractmethod
    def smooth(self, labels: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to label sequences."""
        pass
    
    def _enforce_min_dwell(
        self,
        labels: np.ndarray,
        min_dwell: int = 3,
    ) -> np.ndarray:
        """Enforce minimum dwell time for all labels."""
        pass
```

### Concrete Implementations

#### Median Hysteresis Policy

```python
from conv2d.temporal.median import MedianHysteresisPolicy

policy = MedianHysteresisPolicy(
    min_dwell=5,
    window_size=7,
    enter_threshold=0.7,
    exit_threshold=0.3,
)

smoothed = policy.smooth(labels)  # (B, T) → (B, T)
```

#### HSMM Policy (Optional)

```python
from conv2d.temporal.hsmm import HSMMPolicy

policy = HSMMPolicy(
    min_dwell=3,
    duration_model="negative_binomial",
    transition_learnable=True,
)

smoothed = policy.smooth(labels, features=features)
```

## Median Hysteresis Algorithm

The core smoothing algorithm combines median filtering with hysteresis thresholds:

### Algorithm Flow

```python
def smooth(self, labels: np.ndarray) -> np.ndarray:
    """Apply median hysteresis smoothing."""
    
    # Step 1: Apply median filter
    median_filtered = self._apply_median_filter(labels)
    
    # Step 2: Apply hysteresis thresholds  
    hysteresis_filtered = self._apply_hysteresis(median_filtered)
    
    # Step 3: Enforce min-dwell constraint
    final_labels = self._enforce_min_dwell(hysteresis_filtered)
    
    return final_labels
```

### Median Filtering

```python
def _apply_median_filter(self, labels: np.ndarray) -> np.ndarray:
    """Apply median filter to eliminate noise spikes."""
    
    B, T = labels.shape
    filtered = np.copy(labels)
    
    # Apply median filter with configurable window
    half_window = self.window_size // 2
    
    for b in range(B):
        for t in range(half_window, T - half_window):
            window_start = t - half_window
            window_end = t + half_window + 1
            
            # Get median of window
            window_values = labels[b, window_start:window_end]
            filtered[b, t] = np.median(window_values)
    
    return filtered
```

### Hysteresis Thresholds

```python
def _apply_hysteresis(self, labels: np.ndarray) -> np.ndarray:
    """Apply enter/exit thresholds to prevent oscillation."""
    
    B, T = labels.shape
    smoothed = np.copy(labels)
    
    for b in range(B):
        current_state = labels[b, 0]
        
        for t in range(1, T):
            candidate_state = labels[b, t]
            
            if candidate_state != current_state:
                # State change candidate - check thresholds
                
                # Count confidence in new state
                confidence = self._compute_confidence(
                    labels[b], t, candidate_state
                )
                
                # Enter new state only if above threshold
                if confidence >= self.enter_threshold:
                    current_state = candidate_state
                # Exit current state only if below threshold
                elif confidence <= self.exit_threshold:
                    # Look for alternative state
                    current_state = self._find_alternative_state(
                        labels[b], t
                    )
            
            smoothed[b, t] = current_state
    
    return smoothed
```

### Min-Dwell Enforcement

```python
def _enforce_min_dwell(
    self, 
    labels: np.ndarray, 
    min_dwell: int = 3
) -> np.ndarray:
    """Eliminate segments shorter than min_dwell."""
    
    B, T = labels.shape
    enforced = np.copy(labels)
    
    for b in range(B):
        segments = self._find_segments(labels[b])
        
        for start, end, state in segments:
            duration = end - start
            
            if duration < min_dwell:
                # Segment too short - merge with neighbors
                replacement = self._choose_replacement(
                    labels[b], start, end, state
                )
                enforced[b, start:end] = replacement
    
    return enforced
```

## Usage Examples

### Basic Temporal Smoothing

```python
import numpy as np
from conv2d.temporal.median import MedianHysteresisPolicy

# Raw noisy behavioral labels
labels = np.array([
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0],  # Batch 1: noisy transitions
    [2, 2, 2, 1, 2, 2, 3, 3, 3, 3],  # Batch 2: brief interruption
])

# Create temporal policy
policy = MedianHysteresisPolicy(
    min_dwell=3,
    window_size=5,
    enter_threshold=0.6,
    exit_threshold=0.4,
)

# Apply smoothing
smoothed = policy.smooth(labels)

print("Original:", labels[0])
print("Smoothed:", smoothed[0])
# Original: [0 0 1 0 0 1 1 1 0 0]
# Smoothed: [0 0 0 0 0 1 1 1 1 1]  # Eliminates flickers
```

### Configuration-Based Policy Selection

```python
# Configuration-driven policy selection
config = {
    "temporal": {
        "type": "median_hysteresis",
        "min_dwell": 5,
        "window_size": 7,
        "enter_threshold": 0.7,
        "exit_threshold": 0.3,
    }
}

# Factory pattern for policy creation
def create_temporal_policy(config):
    """Create temporal policy from configuration."""
    
    temporal_config = config["temporal"]
    policy_type = temporal_config["type"]
    
    if policy_type == "median_hysteresis":
        return MedianHysteresisPolicy(**temporal_config)
    elif policy_type == "hsmm":
        return HSMMPolicy(**temporal_config)
    else:
        raise ValueError(f"Unknown temporal policy: {policy_type}")

policy = create_temporal_policy(config)
smoothed_labels = policy.smooth(raw_labels)
```

### Batch Processing

```python
# Process multiple sequences efficiently
sequences = [labels_t0, labels_t1, labels_t2]  # List of (B, T) arrays

policy = MedianHysteresisPolicy(min_dwell=3)
smoothed_sequences = []

for labels in sequences:
    smoothed = policy.smooth(labels)
    smoothed_sequences.append(smoothed)
```

## Configuration Parameters

### Median Hysteresis Parameters

```python
policy = MedianHysteresisPolicy(
    min_dwell=5,           # Minimum state duration (frames)
    window_size=7,         # Median filter window size (odd)
    enter_threshold=0.7,   # Confidence to enter new state
    exit_threshold=0.3,    # Confidence to exit current state
    verbose=False,         # Debug logging
)
```

**Parameter Guidelines:**
- `min_dwell`: 3-10 frames (shorter = more responsive, longer = more stable)
- `window_size`: 3-11 frames (must be odd, larger = more smoothing)
- `enter_threshold`: 0.5-0.8 (higher = harder to change state)
- `exit_threshold`: 0.2-0.5 (lower = easier to exit unstable state)

### HSMM Parameters

```python
policy = HSMMPolicy(
    min_dwell=3,                    # Minimum state duration
    max_dwell=50,                   # Maximum state duration
    duration_model="negative_binomial", # "poisson", "gaussian"
    transition_learnable=True,      # Learn transition matrix
    emission_learnable=False,       # Fixed emission (from clustering)
    n_iterations=100,               # EM algorithm iterations
)
```

## Quality Metrics

### Temporal Quality Assessment

```python
def assess_temporal_quality(labels_raw, labels_smoothed):
    """Assess temporal smoothing quality."""
    
    metrics = {}
    
    # Transition rate reduction
    transitions_raw = count_transitions(labels_raw)
    transitions_smoothed = count_transitions(labels_smoothed)
    metrics['transition_reduction'] = (
        transitions_raw - transitions_smoothed
    ) / transitions_raw
    
    # Min-dwell compliance
    metrics['min_dwell_violations'] = count_violations(
        labels_smoothed, min_dwell=3
    )
    
    # State preservation (no new states introduced)
    unique_raw = set(labels_raw.flatten())
    unique_smoothed = set(labels_smoothed.flatten())
    metrics['state_preservation'] = unique_smoothed <= unique_raw
    
    # Temporal consistency
    metrics['consistency'] = compute_consistency_score(
        labels_raw, labels_smoothed
    )
    
    return metrics

def count_transitions(labels):
    """Count state transitions across all sequences."""
    transitions = 0
    B, T = labels.shape
    
    for b in range(B):
        for t in range(1, T):
            if labels[b, t] != labels[b, t-1]:
                transitions += 1
    
    return transitions
```

### Stability Analysis

```python
def analyze_stability(labels_sequence, policy, n_runs=10):
    """Analyze temporal policy stability across runs."""
    
    results = []
    
    for run in range(n_runs):
        # Add small amount of noise
        noisy_labels = add_noise(labels_sequence, noise_level=0.1)
        
        # Apply temporal smoothing
        smoothed = policy.smooth(noisy_labels)
        
        # Collect metrics
        metrics = assess_temporal_quality(labels_sequence, smoothed)
        results.append(metrics)
    
    # Analyze stability
    stability = {
        'mean_transition_reduction': np.mean([r['transition_reduction'] for r in results]),
        'std_transition_reduction': np.std([r['transition_reduction'] for r in results]),
        'consistent_state_preservation': all(r['state_preservation'] for r in results),
    }
    
    return stability
```

## Advanced Features

### Context-Aware Smoothing

```python
def smooth_with_context(
    self,
    labels: np.ndarray,
    features: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply smoothing with additional context."""
    
    base_smoothed = self.smooth(labels)
    
    if confidence is not None:
        # Use confidence to weight smoothing decisions
        base_smoothed = self._confidence_weighted_smooth(
            labels, base_smoothed, confidence
        )
    
    if features is not None:
        # Use feature similarity for smoothing decisions
        base_smoothed = self._feature_guided_smooth(
            labels, base_smoothed, features
        )
    
    return base_smoothed
```

### Multi-Resolution Smoothing

```python
class MultiResolutionPolicy(TemporalPolicy):
    """Apply smoothing at multiple temporal scales."""
    
    def __init__(self, scales=[3, 7, 15]):
        self.scales = scales
        self.policies = [
            MedianHysteresisPolicy(window_size=scale)
            for scale in scales
        ]
    
    def smooth(self, labels: np.ndarray) -> np.ndarray:
        """Apply multi-scale smoothing."""
        
        smoothed = labels.copy()
        
        # Apply each scale progressively
        for policy in self.policies:
            smoothed = policy.smooth(smoothed)
        
        return smoothed
```

### Adaptive Thresholds

```python
def _compute_adaptive_thresholds(
    self,
    labels: np.ndarray,
    features: np.ndarray,
) -> Tuple[float, float]:
    """Compute adaptive enter/exit thresholds based on data."""
    
    # Analyze feature stability
    feature_variance = np.var(features, axis=1)  # Variance per timestep
    
    # High variance → higher thresholds (more conservative)
    # Low variance → lower thresholds (more responsive)
    base_enter = 0.7
    base_exit = 0.3
    
    variance_factor = np.clip(feature_variance.mean(), 0.5, 2.0)
    
    enter_threshold = base_enter * variance_factor
    exit_threshold = base_exit / variance_factor
    
    return enter_threshold, exit_threshold
```

## Integration Examples

### With FSQ Features

```python
from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.gmm import GMMClusterer
from conv2d.temporal.median import MedianHysteresisPolicy

# Process sequence of IMU data
x_sequence = torch.randn(10, 32, 9, 2, 100)  # (T, B, C, S, L)
labels_sequence = []

clusterer = GMMClusterer(random_state=42)

for t in range(10):
    # Encode with FSQ
    result = encode_fsq(x_sequence[t])
    
    # Cluster features
    labels = clusterer.fit_predict(result.features.numpy(), k=4)
    labels_sequence.append(labels)

# Stack temporal sequence
motif_sequence = np.array(labels_sequence).T  # (B, T)

# Apply temporal smoothing
policy = MedianHysteresisPolicy(min_dwell=3)
smoothed_sequence = policy.smooth(motif_sequence)

print(f"Transitions reduced: {count_transitions(motif_sequence)} → {count_transitions(smoothed_sequence)}")
```

### With Calibration

```python
from conv2d.metrics.calibration import CalibrationAnalyzer

# Apply temporal smoothing to confidence scores
raw_confidences = model.predict_proba(features)  # (B, T, K)
raw_labels = np.argmax(raw_confidences, axis=-1)  # (B, T)

# Smooth labels
policy = MedianHysteresisPolicy(min_dwell=5)
smoothed_labels = policy.smooth(raw_labels)

# Analyze calibration impact
analyzer = CalibrationAnalyzer()

# Calibration before smoothing
cal_before = analyzer.analyze(y_true, raw_labels, raw_confidences.max(axis=-1))

# Calibration after smoothing (need to align confidences)
smoothed_confidences = align_confidences(raw_confidences, smoothed_labels)
cal_after = analyzer.analyze(y_true, smoothed_labels, smoothed_confidences)

print(f"ECE before: {cal_before.ece:.3f}")
print(f"ECE after: {cal_after.ece:.3f}")
```

## Performance Optimization

### Vectorized Processing

```python
def smooth_vectorized(self, labels: np.ndarray) -> np.ndarray:
    """Vectorized implementation for large batches."""
    
    B, T = labels.shape
    
    # Vectorized median filtering
    smoothed = scipy.signal.medfilt(labels, kernel_size=(1, self.window_size))
    
    # Vectorized min-dwell enforcement
    for b in range(B):
        smoothed[b] = self._enforce_min_dwell_fast(smoothed[b])
    
    return smoothed.astype(labels.dtype)

def _enforce_min_dwell_fast(self, sequence: np.ndarray) -> np.ndarray:
    """Fast min-dwell enforcement using run-length encoding."""
    
    # Find runs
    diff_indices = np.where(np.diff(sequence) != 0)[0] + 1
    run_starts = np.concatenate(([0], diff_indices))
    run_ends = np.concatenate((diff_indices, [len(sequence)]))
    
    # Process short runs
    for start, end in zip(run_starts, run_ends):
        if end - start < self.min_dwell:
            # Replace with most common neighbor
            replacement = self._get_neighbor_replacement(
                sequence, start, end
            )
            sequence[start:end] = replacement
    
    return sequence
```

### Memory-Efficient Processing

```python
def smooth_streaming(
    self,
    labels_generator,
    chunk_size: int = 1000,
) -> Generator[np.ndarray, None, None]:
    """Process large sequences in streaming fashion."""
    
    buffer = []
    buffer_size = self.window_size + self.min_dwell
    
    for chunk in labels_generator:
        # Add to buffer
        if len(buffer) == 0:
            buffer = chunk.tolist()
        else:
            buffer.extend(chunk.tolist())
        
        # Process when buffer is full
        while len(buffer) >= chunk_size + buffer_size:
            # Extract chunk with overlap
            chunk_data = np.array(buffer[:chunk_size + buffer_size])
            
            # Apply smoothing
            smoothed_chunk = self.smooth(chunk_data.reshape(1, -1))[0]
            
            # Yield processed chunk (excluding overlap)
            yield smoothed_chunk[:chunk_size]
            
            # Keep overlap for next iteration
            buffer = buffer[chunk_size:]
    
    # Process remaining buffer
    if buffer:
        remaining = np.array(buffer).reshape(1, -1)
        yield self.smooth(remaining)[0]
```

## Best Practices

1. **Choose appropriate min_dwell**: 3-5 frames for responsive systems, 7-10 for stable systems
2. **Tune hysteresis thresholds**: Higher enter_threshold for stability, lower exit_threshold for responsiveness
3. **Validate no new states**: Temporal smoothing should never introduce new behavioral states
4. **Monitor transition rates**: Aim for 30-70% reduction in spurious transitions
5. **Test with real data**: Synthetic noise patterns may not reflect real behavioral dynamics
6. **Consider context**: Use feature information and confidence scores when available
7. **Profile performance**: Vectorized implementations are 10-50x faster for large batches
8. **Validate determinism**: Same input should always produce identical smoothed output

This temporal policy system provides robust, configurable behavioral smoothing that eliminates artifacts while preserving meaningful behavioral dynamics and state transitions.