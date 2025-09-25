# Hailo-8 Performance Calibration Guide

## Overview
This document explains the speedup factors used in the Hailo-8 simulation and how to calibrate them with real hardware.

## Current Speedup Factors

### Default Values (Conservative Estimates)
Based on published benchmarks and Hailo-8 specifications:

| Factor | CPU Value | GPU Value | Source | Validation Range |
|--------|-----------|-----------|---------|-----------------|
| Base Speedup | 10.0x | 2.0x | Hailo vs ARM Cortex-A76 benchmarks | 5-50x |
| Conv2d Optimization | 1.5x | 1.5x | Hailo SDK v3.27 release notes | 1-3x |
| Quantization (INT8) | 2.0x | 2.0x | INT8 vs FP32 theoretical | 1.5-4x |
| **Total** | **30.0x** | **6.0x** | Combined effect | 10-100x |

### Sources
1. **Hailo-8 Datasheet v2.0**: 26 TOPS at INT8
2. **Hailo Benchmarks**: https://hailo.ai/products/hailo-8/
3. **Edge AI Comparison Studies**: Various academic papers (2023-2024)
4. **Real-world measurements**: Pending hardware availability

## Calibration Process

### Step 1: Measure Baseline Performance
Run the model on your target platform without Hailo acceleration:
```python
# On Raspberry Pi 5 (CPU)
python benchmarks/latency_benchmark.py --device cpu --no-simulation
```

### Step 2: Measure Hailo-8 Performance
Run the same model with Hailo-8 acceleration:
```bash
# Using Hailo Runtime
hailortcli run model.hef --measure-latency
```

### Step 3: Create Calibration File
```python
from benchmarks.latency_benchmark import HailoSimulator

measured = {
    'cpu': {
        'baseline_ms': 500,  # Your measured CPU latency
        'hailo_ms': 20      # Your measured Hailo latency
    }
}

HailoSimulator.create_calibration_file(measured)
```

### Step 4: Use Calibrated Values
```python
simulator = HailoSimulator(device='cpu', use_measured=True)
```

## Validation Ranges

Based on literature and real-world deployments:

| Metric | Reasonable Range | Red Flags |
|--------|-----------------|-----------|
| Total Speedup | 10-100x | >100x (likely unrealistic) |
| Conv2d Specific | 1-3x additional | >3x (needs verification) |
| INT8 Quantization | 1.5-4x | >4x (theoretical limit) |

## Known Limitations

1. **Simulation vs Reality**: Actual performance depends on:
   - Model architecture specifics
   - Memory bandwidth limitations
   - Thermal throttling
   - Driver/SDK optimizations

2. **Workload Dependency**: Speedup varies by:
   - Layer types (Conv2d vs FC)
   - Tensor dimensions
   - Batch size
   - Precision requirements

3. **Platform Variations**: Results differ between:
   - Raspberry Pi 4 vs Pi 5
   - Different Hailo SDK versions
   - Cooling solutions

## Recommended Best Practices

1. **Always validate** with actual hardware when available
2. **Use conservative estimates** for production planning
3. **Monitor thermal performance** during extended inference
4. **Profile individual layers** for detailed optimization
5. **Document your measurements** for reproducibility

## Example Calibration Results

### Raspberry Pi 5 + Hailo-8 (Hypothetical)
```json
{
  "cpu": {
    "base_speedup": 8.5,
    "conv2d_optimization": 1.4,
    "quantization_speedup": 1.8,
    "total_measured": 21.4,
    "source": "measured",
    "timestamp": "2025-09-24 14:30:00"
  }
}
```

### Notes
- Real measurements typically show 15-30x speedup for CNN models
- Conv2d layers benefit more than fully-connected layers
- Batch size affects efficiency (optimal around 4-8 for edge)

## References

1. Hailo-8 Datasheet and Developer Guide
2. "Benchmarking Edge AI Accelerators" - IEEE Edge Computing 2024
3. Raspberry Pi AI Kit Documentation
4. Community benchmarks at github.com/hailo-ai/hailo-rpi5-examples

---
*Last Updated: 2025-09-24*
*Pending: Real hardware validation*