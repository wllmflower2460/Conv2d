# FSQ Contract Guide

The FSQ (Finite Scalar Quantization) contract provides deterministic encoding with guaranteed shape/dtype enforcement for edge deployment safety.

## Overview

The FSQ contract ensures that:
- **Same input → identical codes** (100% deterministic)
- **Strict shape enforcement**: `(B,9,2,100) → outputs` with `float32`
- **Code usage validation**: All active levels used (no collapse)
- **Edge compatibility**: Hailo-8 safe operations only

## API Reference

### Core Function

```python
from conv2d.features.fsq_contract import encode_fsq

result = encode_fsq(
    x: Tensor,                    # Input (B,9,2,100) float32
    levels: List[int] = [8,6,5],  # Quantization levels
    embedding_dim: int = 64,      # Embedding dimension
    reset_stats: bool = False,    # Reset internal statistics
) -> CodesAndFeatures
```

### Return Type

```python
@dataclass
class CodesAndFeatures:
    codes: Tensor         # (B, embedding_dim) int32
    features: Tensor      # (B, feature_dim) float32  
    embeddings: Tensor    # (B, embedding_dim) float32
    perplexity: float     # Codebook usage metric
```

## Usage Examples

### Basic Encoding

```python
import torch
from conv2d.features.fsq_contract import encode_fsq

# Input: IMU data (batch_size, channels, sensors, timesteps)
x = torch.randn(32, 9, 2, 100, dtype=torch.float32)

# Encode with FSQ
result = encode_fsq(x, levels=[8, 6, 5], embedding_dim=64)

print(f"Codes shape: {result.codes.shape}")        # (32, 64)
print(f"Features shape: {result.features.shape}")  # (32, 256) 
print(f"Embeddings shape: {result.embeddings.shape}")  # (32, 64)
print(f"Perplexity: {result.perplexity:.2f}")      # 120.5
```

### Deterministic Verification

```python
# Same input should produce identical results
x = torch.randn(16, 9, 2, 100, dtype=torch.float32)

result1 = encode_fsq(x, reset_stats=True)
result2 = encode_fsq(x, reset_stats=True)

# Codes must be identical
assert torch.equal(result1.codes, result2.codes)
print("✓ Deterministic encoding verified")
```

### Code Usage Analysis

```python
# Monitor codebook utilization
result = encode_fsq(x, levels=[8, 6, 5])

# Check perplexity (higher = better usage)
if result.perplexity > 100:
    print("✓ Good codebook utilization")
else:
    print("⚠️ Potential codebook collapse")

# Analyze unique codes
unique_codes = torch.unique(result.codes)
total_possible = 8 * 6 * 5  # 240 codes
usage_rate = len(unique_codes) / total_possible
print(f"Code usage: {usage_rate:.1%}")
```

## Configuration Options

### Quantization Levels

```python
# Standard configuration (240 codes)
result = encode_fsq(x, levels=[8, 6, 5])

# High capacity (256 codes)
result = encode_fsq(x, levels=[4, 4, 4, 4])

# Compact (120 codes)
result = encode_fsq(x, levels=[8, 3, 5])
```

### Embedding Dimensions

```python
# Standard 64-dimensional embeddings
result = encode_fsq(x, embedding_dim=64)

# Higher capacity (more parameters)
result = encode_fsq(x, embedding_dim=128)

# Compact (faster inference)
result = encode_fsq(x, embedding_dim=32)
```

## Validation and Quality Checks

### Input Validation

```python
# Required input format
assert x.dtype == torch.float32, "Must be float32"
assert x.shape[1:] == (9, 2, 100), "Must be (B,9,2,100)"
assert torch.isfinite(x).all(), "No NaN/Inf values"
```

### Output Validation

```python
result = encode_fsq(x)

# Code validation
assert result.codes.dtype == torch.int32, "Codes must be int32"
assert result.codes.shape[0] == x.shape[0], "Batch size preserved"
assert result.codes.min() >= 0, "Codes non-negative"
assert result.codes.max() < 240, "Codes within bounds"

# Feature validation  
assert result.features.dtype == torch.float32, "Features must be float32"
assert torch.isfinite(result.features).all(), "Features finite"

# Embedding validation
assert result.embeddings.dtype == torch.float32, "Embeddings must be float32" 
assert result.embeddings.shape[1] == embedding_dim, "Embedding dim correct"
```

## Performance Characteristics

### Speed Benchmarks

| Batch Size | Inference Time | Memory Usage |
|------------|---------------|---------------|
| 1          | <10ms         | ~50MB        |
| 32         | <50ms         | ~200MB       |
| 256        | <200ms        | ~1GB         |

### Code Quality Metrics

```python
def analyze_codebook_health(result):
    """Analyze FSQ codebook quality."""
    
    # Perplexity (target: 50-200)
    perplexity = result.perplexity
    
    # Code distribution entropy
    unique, counts = torch.unique(result.codes, return_counts=True)
    probs = counts.float() / counts.sum()
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    
    # Usage statistics
    total_codes = 8 * 6 * 5  # For levels [8,6,5]
    active_codes = len(unique)
    usage_rate = active_codes / total_codes
    
    return {
        "perplexity": perplexity,
        "entropy": entropy.item(),
        "active_codes": active_codes,
        "usage_rate": usage_rate,
        "health": "good" if usage_rate > 0.4 else "warning"
    }
```

## Error Handling

### Common Issues

```python
try:
    result = encode_fsq(x)
except ValueError as e:
    if "shape mismatch" in str(e):
        print("Input must be (B,9,2,100)")
    elif "dtype" in str(e):
        print("Input must be float32")
    else:
        raise

# Graceful degradation
if result.perplexity < 10:
    print("Warning: Codebook collapse detected")
    # Could trigger reinitialization
```

### Debugging Tools

```python
# Enable detailed logging
import logging
logging.getLogger('conv2d.features').setLevel(logging.DEBUG)

# Trace intermediate values
result = encode_fsq(x, reset_stats=True)
print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
print(f"Feature range: [{result.features.min():.3f}, {result.features.max():.3f}]")
print(f"Code range: [{result.codes.min()}, {result.codes.max()}]")
```

## Edge Deployment

### ONNX Export

```python
# Create ONNX-compatible version
def fsq_for_onnx(x):
    result = encode_fsq(x, levels=[8, 6, 5])
    return result.codes, result.features

# Export
torch.onnx.export(
    fsq_for_onnx,
    torch.randn(1, 9, 2, 100, dtype=torch.float32),
    "fsq_encoder.onnx",
    input_names=["imu_input"],
    output_names=["codes", "features"],
    dynamic_axes={"imu_input": {0: "batch_size"}}
)
```

### Hailo-8 Compatibility

```python
# Ensure Hailo-8 compatible operations
# - No dynamic shapes (fixed batch=1)
# - Float32 inputs/outputs only
# - Standard conv2d/linear operations

x = torch.randn(1, 9, 2, 100, dtype=torch.float32)  # Fixed batch
result = encode_fsq(x)  # Uses only supported ops
```

## Best Practices

1. **Always validate inputs**: Check shape and dtype before encoding
2. **Monitor perplexity**: Values <50 indicate codebook issues
3. **Use fixed batch sizes**: For edge deployment consistency
4. **Reset stats appropriately**: Use `reset_stats=True` for reproducibility
5. **Test determinism**: Verify identical outputs for same inputs

## Integration Examples

### With Clustering

```python
from conv2d.clustering import GMMClusterer

# FSQ encoding
result = encode_fsq(x)

# Cluster the features
clusterer = GMMClusterer(random_state=42)
labels = clusterer.fit_predict(result.features.numpy(), k=4)
```

### With Temporal Smoothing

```python
from conv2d.temporal.median import MedianHysteresisPolicy

# Process sequence
sequence_results = []
for t in range(T):
    result = encode_fsq(x_sequence[t])
    sequence_results.append(result.codes)

codes_sequence = torch.stack(sequence_results)  # (T, B, 64)

# Apply temporal smoothing (needs clustering first)
# ... cluster codes to motifs, then smooth
```

This FSQ contract ensures deterministic, edge-safe encoding that forms the foundation of the entire behavioral analysis pipeline.