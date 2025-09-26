# Preprocessing Documentation

## NaN Interpolation Policy

The data quality handler provides configurable NaN interpolation with multiple strategies for handling missing data in sensor streams.

### Configuration

Configure the interpolation policy when creating a `DataQualityHandler`:

```python
from preprocessing.data_quality_handler import DataQualityHandler

# Default configuration
handler = DataQualityHandler(
    default_nan_strategy='interpolate',  # Primary strategy
    nan_threshold_warn=5.0,              # Warn if >5% NaN
    nan_threshold_error=20.0,            # Error if >20% NaN
    auto_select_strategy=True            # Auto-select based on data
)
```

### Interpolation Strategies

#### Primary NaN Strategies
- **`'interpolate'`** (default): Linear interpolation between valid points
- **`'mean'`**: Replace with channel mean
- **`'median'`**: Replace with channel median
- **`'zero'`**: Replace with zeros
- **`'drop'`**: Drop samples with NaN
- **`'raise'`**: Raise error on NaN

#### Edge NaN Handling Methods
For the `'interpolate'` strategy, edge NaNs are handled via:
- **`'extrapolate'`** (default): Linear extrapolation at edges
- **`'constant'`**: Hold nearest valid value
- **`'ffill'`**: Forward/backward fill

#### All-NaN Row Fallback
When entire rows are NaN:
- **`'zero'`**: Fill with zeros
- **`'mean'`**: Fill with channel mean from other samples
- **`'median'`**: Fill with channel median from other samples

### Usage Examples

```python
# Basic interpolation with default settings
data, report = handler.correct_data(
    data,
    name="imu_data"
)

# Custom interpolation with specific edge handling
result = handler._interpolate_nan(
    data,
    nan_fallback='median',      # Use median for all-NaN rows
    edge_method='constant'       # Hold edge values constant
)

# Override strategy for specific dataset
data, report = handler.correct_data(
    data,
    nan_strategy='mean',        # Override default
    name="noisy_sensor"
)
```

### Monitoring and Logging

The handler provides detailed logging of interpolation actions:

```
INFO: NaN interpolation counts: num_rows_interpolated=42, num_rows_mean_fallback=3, num_rows_all_nan_zero=1
INFO: Gap size histogram (sensor dropouts): 1-2:15, 2-5:8, 5-10:3, 10-20:1, 20-50:0, 50-100:0, >100:0
```

Gap size histogram helps diagnose sensor dropout patterns:
- **1-2 samples**: Typical single-point dropouts
- **2-5 samples**: Brief sensor interruptions  
- **5-10 samples**: Short connectivity issues
- **10+ samples**: Extended sensor failures requiring investigation

### Configuration in Config Files

To set the interpolation policy in configuration files:

```yaml
# config/training_config.yaml
data_quality:
  default_nan_strategy: interpolate
  nan_threshold_warn: 5.0
  nan_threshold_error: 20.0
  auto_select_strategy: true
  
  # Advanced options
  interpolation:
    edge_method: constant
    nan_fallback: median
```

```python
# In training scripts
from config.training_config import config

handler = DataQualityHandler(
    default_nan_strategy=config.data_quality.default_nan_strategy,
    nan_threshold_warn=config.data_quality.nan_threshold_warn,
    nan_threshold_error=config.data_quality.nan_threshold_error
)
```

### Performance Considerations

- **Interpolation** is fastest for scattered NaNs
- **Mean/Median** require full channel statistics
- **Zero** is fastest but may distort signals
- **Drop** reduces dataset size

The handler automatically tracks and logs performance metrics to help optimize strategy selection.

## Vectorized Distance Computation

The kinematic feature extractor uses optimized vectorized Euclidean distance computation with built-in safety checks.

### Guard Rails

The distance function includes automatic shape validation:

```python
assert x.shape == y.shape, f"Shape mismatch: x={x.shape}, y={y.shape}"
```

This ensures:
- Batch dimensions match
- Channel dimensions match  
- Time dimensions match

### Numerical Stability

Uses `torch.linalg.vector_norm` for improved numerical stability:
- Handles large values without overflow
- Maintains precision for small differences
- Supports automatic differentiation

## Stride Tricks Safety

When using sliding windows with stride tricks, the implementation includes safety measures:

```python
# WARNING: Do not write to this view; call .copy() before modifying
windows = as_strided(
    data,
    shape=(n_windows, window_size, n_channels),
    strides=(stride_samples * step_size, stride_samples, stride_channels),
    writeable=False  # Make read-only to prevent accidental mutations
)
# Return a copy to ensure safety
return windows.copy()
```

**Important**: The windowed views are read-only. Always work with the returned copy to avoid memory corruption.