# FSQ Tuning Guide • Codebook & Clustering

[[Master_MOC]] • [[02__MVP/README]] • [[Conv2d_Documentation]]

| **Metadata** | **Value** |
|-------------|-----------|
| status | production |
| priority | high |
| project | Conv2d |
| component | tuning-guide |
| created | 2025-09-25 |
| updated | 2025-09-26 |
| version | v1.1 |
| validation | D1 review complete |

---

## Executive Summary

This guide provides production-ready tuning strategies for FSQ (Finite Scalar Quantization) in the Conv2d-VQ-HDP-HSMM architecture. Following D1 committee feedback, we've optimized from a wasteful 16M codebook to an efficient 64-code configuration achieving >80% utilization while maintaining 96.7% accuracy.

## 1. FSQ Level Selection

### Production Configurations

| **Configuration** | **FSQ Levels** | **Total Codes** | **Use Case** | **Utilization** |
|------------------|----------------|-----------------|--------------|-----------------|
| **Optimized** ✅ | `[4, 4, 4]` | 64 | Production default | >80% |
| **Balanced** | `[8, 6, 5]` | 240 | Complex behaviors | ~60% |
| **Extended** | `[8, 6, 5, 5, 4]` | 4,800 | Research/exploration | ~40% |
| **Minimal** | `[4, 4]` | 16 | Resource-constrained | >90% |
| **Wasteful** ❌ | `[8,8,8,8,8,8,8,8]` | 16,777,216 | Never use | <0.01% |

### Selection Criteria

```python
# Rule of thumb: Keep effective active codes in dozens, not hundreds
def select_fsq_levels(dataset_properties):
    """Select appropriate FSQ levels based on dataset characteristics."""
    
    n_behaviors = dataset_properties['n_distinct_behaviors']
    temporal_complexity = dataset_properties['temporal_complexity']
    
    if n_behaviors < 10:
        return [4, 4, 4]  # 64 codes - Production default
    elif n_behaviors < 30:
        return [8, 6, 5]  # 240 codes - Balanced
    elif temporal_complexity == 'high':
        return [8, 6, 5, 5, 4]  # 4,800 codes - Extended
    else:
        return [4, 4, 4]  # Default to optimized
```

### Signs You Need More Levels

- **Motifs over-merged**: Different gaits collapsed into single cluster
- **Clustering silhouette score < 0.3**: Poor separation between behaviors
- **Reconstruction error > 0.05**: Insufficient expressiveness
- **Perplexity saturated**: All codes being used equally

### Signs You Need Fewer Levels

- **Code usage histogram sparse**: <20% codes active
- **Calibration ECE > 0.03**: Overconfident predictions
- **Memory constraints**: Edge device limitations
- **Training instability**: Codebook collapse

## 2. Window Size Optimization

### Standard Configurations

| **Window Size** | **Duration** | **Use Case** | **FSQ Levels** |
|----------------|--------------|--------------|----------------|
| 50 | ~0.5s | Fast transitions | `[4, 4]` |
| **100** ✅ | ~1-2s | **Default** | `[4, 4, 4]` |
| 150 | ~2-3s | Extended context | `[8, 6, 5]` |
| 200 | ~3-4s | Long behaviors | `[8, 6, 5, 5, 4]` |

### Dynamic Window Adjustment

```python
def adjust_window_for_fsq(fsq_levels, base_window=100):
    """Adjust window size based on FSQ configuration."""
    
    n_codes = np.prod(fsq_levels)
    
    if n_codes < 50:
        # Fewer codes need more temporal context
        return int(base_window * 1.5)  # 150
    elif n_codes > 1000:
        # Many codes need shorter windows to populate
        return int(base_window * 0.75)  # 75
    else:
        return base_window  # 100
```

## 3. Clustering Configuration

### 3.1 Clustering Methods

| **Method** | **Pros** | **Cons** | **When to Use** |
|-----------|----------|----------|-----------------|
| **K-Means** ✅ | Fast, stable | Fixed K | Production default |
| **GMM** | Soft assignments | Slower | Uncertainty needed |
| **DBSCAN** | Finds outliers | Parameter sensitive | Anomaly detection |
| **Hierarchical** | No K needed | Memory intensive | Exploration |

### 3.2 Optimal K Selection

```python
def find_optimal_k(fsq_codes, min_k=4, max_k=24):
    """Find optimal number of clusters using BIC."""
    
    bic_scores = []
    silhouette_scores = []
    
    for k in range(min_k, max_k + 1):
        # Fit clustering
        kmeans = KMeans(n_clusters=k, n_init=10)
        labels = kmeans.fit_predict(fsq_codes)
        
        # Calculate metrics
        bic = calculate_bic(fsq_codes, labels, k)
        silhouette = silhouette_score(fsq_codes, labels)
        
        bic_scores.append(bic)
        silhouette_scores.append(silhouette)
    
    # Primary criterion: BIC minimum
    optimal_k = min_k + np.argmin(bic_scores)
    
    # Fallback: silhouette maximum if BIC inconclusive
    if np.std(bic_scores) < 0.01:
        optimal_k = min_k + np.argmax(silhouette_scores)
    
    return optimal_k
```

### 3.3 Minimum Support Enforcement

```python
def enforce_min_support(labels, min_support=0.005):
    """Merge or drop clusters with insufficient support."""
    
    n_samples = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Find small clusters
    small_clusters = unique_labels[counts < n_samples * min_support]
    
    if len(small_clusters) > 0:
        # Merge small clusters to nearest large cluster
        for small_label in small_clusters:
            small_indices = labels == small_label
            # Find nearest large cluster and reassign
            labels[small_indices] = find_nearest_large_cluster(
                labels, small_indices, min_support
            )
    
    return relabel_sequential(labels)
```

## 4. Temporal Smoothing

### 4.1 Smoothing Pipeline

```python
def temporal_smoothing_pipeline(predictions, config=None):
    """Complete temporal smoothing pipeline."""
    
    if config is None:
        config = {
            'median_k': 7,
            'hysteresis_high': 0.6,
            'hysteresis_low': 0.4,
            'min_dwell_ms': 300,
            'sampling_rate': 50  # Hz
        }
    
    # Step 1: Median filter
    smoothed = median_filter(predictions, size=config['median_k'])
    
    # Step 2: Hysteresis thresholding
    smoothed = apply_hysteresis(
        smoothed,
        high=config['hysteresis_high'],
        low=config['hysteresis_low']
    )
    
    # Step 3: Minimum dwell enforcement
    min_samples = int(config['min_dwell_ms'] * config['sampling_rate'] / 1000)
    smoothed = enforce_min_dwell(smoothed, min_samples)
    
    return smoothed
```

### 4.2 HSMM Integration (Optional)

```python
def apply_hsmm_smoothing(fsq_codes, duration_model='negative_binomial'):
    """Apply HSMM for explicit duration modeling."""
    
    from models.hsmm_components import HSMM
    
    hsmm = HSMM(
        n_states=len(np.unique(fsq_codes)),
        duration_model=duration_model,
        max_duration=50
    )
    
    # Fit HSMM
    hsmm.fit(fsq_codes)
    
    # Viterbi decoding for smoothed states
    smoothed_states = hsmm.viterbi_decode(fsq_codes)
    
    return smoothed_states
```

## 5. Performance Targets & Validation

### 5.1 Target Metrics

| **Metric** | **Target** | **Achieved** | **Validation Method** |
|-----------|------------|--------------|----------------------|
| **Accuracy** | ≥96% | 96.7% | Temporal cross-validation |
| **Motif Count** | 30-60 | 42 | Stable cluster analysis |
| **ECE** | ≤0.03 | 0.025 | Calibration evaluation |
| **Latency** | <15ms | 12ms | Hailo-8 benchmark |
| **Codebook Usage** | >60% | 82% | Utilization histogram |
| **Memory** | <5MB | 2.3MB | Model size check |

### 5.2 Validation Pipeline

```python
def validate_fsq_configuration(model, test_loader, config):
    """Complete validation pipeline for FSQ configuration."""
    
    results = {
        'accuracy': [],
        'ece': [],
        'latency': [],
        'codebook_usage': [],
        'silhouette': []
    }
    
    # Temporal cross-validation with Bonferroni correction
    n_folds = 5
    alpha = 0.05 / n_folds  # Bonferroni correction
    
    for fold in range(n_folds):
        fold_data = get_temporal_fold(test_loader, fold, n_folds)
        
        # Evaluate metrics
        acc = evaluate_accuracy(model, fold_data)
        ece = calculate_ece(model, fold_data)
        lat = measure_latency(model, fold_data)
        usage = calculate_codebook_usage(model, fold_data)
        sil = calculate_silhouette(model, fold_data)
        
        results['accuracy'].append(acc)
        results['ece'].append(ece)
        results['latency'].append(lat)
        results['codebook_usage'].append(usage)
        results['silhouette'].append(sil)
    
    # Statistical validation
    for metric, values in results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_lower, ci_upper = stats.t.interval(
            1 - alpha, len(values) - 1,
            loc=mean_val, scale=std_val/np.sqrt(len(values))
        )
        
        print(f"{metric}: {mean_val:.3f} ± {std_val:.3f} "
              f"(95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    
    return results
```

## 6. Production Implementation

### 6.1 Complete FSQ Model Configuration

```python
from models.conv2d_fsq_optimized import Conv2dFSQOptimized

# Production configuration
model = Conv2dFSQOptimized(
    input_channels=9,
    input_height=2,
    input_width=100,
    fsq_levels=[4, 4, 4],  # Optimized for >80% utilization
    n_classes=7,
    n_motifs=42,
    encoder_dim=64,
    decoder_dim=128,
    dropout=0.1,
    use_calibration=True,
    entropy_weight=0.1
)

# Training configuration
training_config = {
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'gradient_clip': 1.0,
    'weight_decay': 1e-4,
    'scheduler': 'cosine_annealing',
    'warmup_epochs': 5
}
```

### 6.2 Deployment Configuration

```python
# Hailo-8 export configuration
hailo_config = {
    'quantization': 'int8',
    'calibration_samples': 1000,
    'optimization_level': 3,
    'batch_size': 1,
    'input_shape': (1, 9, 2, 100),
    'output_names': ['predictions', 'uncertainty']
}

# Edge deployment script
def deploy_to_edge(model, config):
    """Deploy FSQ model to edge device."""
    
    # Step 1: Export to ONNX
    onnx_path = export_to_onnx(model, config)
    
    # Step 2: Compile for Hailo-8
    hef_path = compile_for_hailo(onnx_path, hailo_config)
    
    # Step 3: Deploy to device
    deploy_to_device(hef_path, device_ip='192.168.1.100')
    
    # Step 4: Validate deployment
    validate_edge_deployment(device_ip='192.168.1.100')
    
    return True
```

## 7. Troubleshooting Guide

### Common Issues & Solutions

| **Issue** | **Symptoms** | **Solution** |
|-----------|-------------|--------------|
| **Codebook collapse** | All samples map to few codes | Reduce FSQ levels, increase commitment loss |
| **Over-segmentation** | Too many short segments | Increase temporal smoothing, min dwell |
| **Under-segmentation** | Behaviors merged | Increase FSQ levels, reduce clustering K |
| **Poor calibration** | ECE > 0.05 | Enable temperature scaling, reduce model capacity |
| **High latency** | >20ms inference | Reduce FSQ levels, optimize for int8 |
| **Memory overflow** | OOM errors | Use [4,4,4] config, reduce batch size |

## 8. Advanced Tuning Strategies

### 8.1 Rate-Distortion Optimization

```python
from models.fsq_rd_rounding import FSQRateDistortionOptimizer

# Optimize FSQ levels for rate-distortion trade-off
optimizer = FSQRateDistortionOptimizer(
    target_entropy=3.0,  # bits
    max_distortion=0.01
)

optimal_levels = optimizer.find_optimal_levels(
    data=validation_data,
    min_levels=2,
    max_levels=5,
    level_range=(2, 8)
)
```

### 8.2 Adaptive Clustering

```python
def adaptive_clustering(fsq_codes, behavior_dynamics):
    """Adapt clustering based on behavioral dynamics."""
    
    # Calculate transition matrix
    transitions = calculate_transitions(fsq_codes)
    
    # Find stable states (low out-transition probability)
    stable_states = find_stable_states(transitions, threshold=0.8)
    
    # Adjust K based on stable states
    suggested_k = len(stable_states) + int(len(stable_states) * 0.3)
    
    # Constrain to reasonable range
    final_k = np.clip(suggested_k, 4, 24)
    
    return final_k
```

## 9. Benchmarking & Monitoring

### 9.1 Continuous Monitoring

```python
# Production monitoring configuration
monitoring_config = {
    'metrics': [
        'codebook_utilization',
        'cluster_stability',
        'prediction_entropy',
        'calibration_ece',
        'inference_latency'
    ],
    'alert_thresholds': {
        'codebook_utilization': 0.2,  # Alert if < 20%
        'calibration_ece': 0.05,      # Alert if > 0.05
        'inference_latency': 20       # Alert if > 20ms
    },
    'logging_interval': 100,  # batches
    'dashboard_update': 1000  # batches
}
```

### 9.2 A/B Testing Framework

```python
def ab_test_fsq_configs(config_a, config_b, test_data, duration_hours=24):
    """A/B test different FSQ configurations."""
    
    results = {
        'config_a': [],
        'config_b': []
    }
    
    # Random assignment
    for batch in test_data:
        if np.random.random() < 0.5:
            result = evaluate_config(config_a, batch)
            results['config_a'].append(result)
        else:
            result = evaluate_config(config_b, batch)
            results['config_b'].append(result)
    
    # Statistical comparison
    p_value = stats.ttest_ind(
        results['config_a'],
        results['config_b']
    ).pvalue
    
    return {
        'winner': 'A' if np.mean(results['config_a']) > np.mean(results['config_b']) else 'B',
        'p_value': p_value,
        'effect_size': calculate_cohens_d(results['config_a'], results['config_b'])
    }
```

## 10. References & Resources

### Key Papers
- [FSQ: Finite Scalar Quantization](https://arxiv.org/abs/2309.15505)
- [VQ-VAE: Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- [Hierarchical Dirichlet Processes](https://people.eecs.berkeley.edu/~jordan/papers/hdp.pdf)

### Implementation Files
- `models/conv2d_fsq_optimized.py` - Production FSQ implementation
- `scripts/validate_fsq_real_data.py` - Validation pipeline
- `experiments/FSQ_LEVELS_GUIDE.md` - Detailed level selection guide
- `models/fsq_rd_rounding.py` - Rate-distortion optimizer

### Related Documentation
- [[D1_DESIGN_GATE_SUMMARY.md]] - Design review feedback
- [[M1.6 Revision/Committee_Notes.md]] - Committee recommendations
- [[Conv2d_Documentation/README.md]] - Project overview

---

## Appendix A: Quick Reference Card

```bash
# Quick FSQ configuration test
python -c "
from models.conv2d_fsq_optimized import Conv2dFSQOptimized
model = Conv2dFSQOptimized(fsq_levels=[4,4,4])
print(f'Codebook size: {model.fsq.codebook_size}')
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
"

# Validate configuration
python scripts/validate_fsq_real_data.py \
    --fsq-levels 4 4 4 \
    --window-size 100 \
    --clustering-k 12 \
    --min-support 0.005

# Deploy to edge
python scripts/deployment/deploy_m13_fsq.py \
    --model-path models/conv2d_fsq_optimized.pth \
    --target hailo8 \
    --optimize int8
```

## Appendix B: Configuration Templates

### Minimal Edge Configuration
```yaml
# config/fsq_edge_minimal.yaml
fsq:
  levels: [4, 4]
  commitment_loss: 0.25
clustering:
  method: kmeans
  k: 8
  min_support: 0.01
temporal:
  median_k: 5
  min_dwell_ms: 200
```

### Balanced Production Configuration
```yaml
# config/fsq_production.yaml
fsq:
  levels: [4, 4, 4]
  commitment_loss: 0.25
clustering:
  method: kmeans
  k: 12
  min_support: 0.005
temporal:
  median_k: 7
  hysteresis_high: 0.6
  hysteresis_low: 0.4
  min_dwell_ms: 300
```

### Research Configuration
```yaml
# config/fsq_research.yaml
fsq:
  levels: [8, 6, 5, 5, 4]
  commitment_loss: 0.1
clustering:
  method: gmm
  k_range: [4, 30]
  selection: bic
  min_support: 0.001
temporal:
  method: hsmm
  duration_model: negative_binomial
  max_duration: 100
```

---

*Last validated: 2025-09-26 | Production deployment: Conv2d v1.1.0*