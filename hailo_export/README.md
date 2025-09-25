# Hailo-8 Export Pipeline

## Overview

This directory contains the export pipeline for deploying Conv2d-FSQ-HSMM models to Hailo-8 NPU accelerators. Updated per **M1.2 checkpoint requirements** with FSQ (Finite Scalar Quantization) replacing VQ-VAE to prevent codebook collapse.

## Architecture Status (M1.2)

### Current Model: Conv2d-FSQ-HSMM
- **Previous**: Conv2d-VQ-HDP-HSMM (VQ collapsed, HDP hurt performance)
- **Current**: Conv2d-FSQ-HSMM (FSQ stable, HDP removed)
- **Parameters**: 46,102 (optimal configuration)
- **Accuracy**: 78.12% (target: 85% for M1.3)

### Key Improvements
- ✅ **VQ Collapse Fixed**: FSQ cannot collapse by design
- ✅ **Ablation Complete**: HDP removed (degraded accuracy 48-71%)
- ✅ **Perplexity Stable**: 38.31 (within 50-200 target)
- ⚠️ **Calibration Pending**: ECE and conformal prediction for M1.3

## Performance Requirements

### M1.2 Targets
| Metric | Target | Status | Notes |
|--------|--------|--------|-------|
| Latency (P95) | <100ms | ⏳ Pending | Requires hardware validation |
| Core Inference | <15ms | ⏳ Pending | Hailo-8 profiling needed |
| Accuracy | ≥85% | ❌ 78.12% | Needs data augmentation |
| Model Size | <32MB | ✅ ~2MB | Well within limits |
| Quantization | INT8 | ✅ Ready | Calibration dataset included |

### M1.3 Requirements
- **Calibration**: ECE ≤3%, 90% conformal coverage
- **Accuracy**: 85% minimum on behavioral test set
- **Latency**: Validated <100ms on actual hardware

## Directory Structure

```
hailo_export/
├── README.md                          # This file
├── hailo8_hef_compilation_checklist.md  # Comprehensive checklist (v2.0)
├── export_fsq_for_hailo.py           # Main FSQ export script (NEW)
├── export_hailo.py                   # Legacy TCN-VAE export
├── export_best_model.py              # Model checkpoint exporter
└── outputs/                          # Generated export artifacts
    ├── conv2d_fsq_behavioral.onnx   # ONNX model
    ├── conv2d_fsq_behavioral_simplified.onnx
    ├── hailo/
    │   ├── calibration_data.npy     # INT8 calibration
    │   ├── compile_for_hailo8.sh    # Compilation script
    │   └── *.hef                    # Compiled Hailo model
    └── DEPLOYMENT_README.md         # Deployment instructions
```

## Quick Start

### 1. Export FSQ Model for Hailo

```bash
# Export FSQ model (no checkpoint = creates example model)
python export_fsq_for_hailo.py --output ./hailo_export_fsq

# With trained checkpoint
python export_fsq_for_hailo.py \
    --checkpoint models/conv2d_fsq_trained_*.pth \
    --output ./hailo_export_fsq
```

### 2. Compile for Hailo-8

```bash
cd hailo_export_fsq/hailo
./compile_for_hailo8.sh

# Output: conv2d_fsq_behavioral.hef
```

### 3. Deploy to Raspberry Pi

```bash
# Copy to Pi with Hailo-8
scp conv2d_fsq_behavioral.hef pi@raspberrypi:/opt/hailo/models/

# Test inference
ssh pi@raspberrypi
hailo run /opt/hailo/models/conv2d_fsq_behavioral.hef \
    --input test_imu_data.npy \
    --measure-latency
```

### 4. Integrate with EdgeInfer

```python
# In EdgeInfer service (pisrv_vapor_docker/EdgeInfer)
from hailo_platform import HailoRTService

class BehavioralInference:
    def __init__(self):
        self.model = HailoRTService.load_hef('conv2d_fsq_behavioral.hef')
        self.fsq_levels = [8,8,8,8,8,8,8,8]  # For post-processing
    
    def predict(self, imu_data):
        # Input: (1, 9, 2, 100) - IMU sensor data
        logits = self.model.run(imu_data)
        
        # Apply FSQ quantization in post-processing if needed
        # Apply calibration (M1.3 requirement)
        
        return logits
```

## Export Pipeline Features

### FSQ-Specific Optimizations
- **No VQ Codebook**: FSQ uses fixed grid, no learned parameters
- **HDP Removed**: Ablation showed 48-71% accuracy degradation
- **BatchNorm Fusion**: Conv2d+BN fused for efficiency
- **Static Shapes**: Fixed dimensions for Hailo compatibility

### Calibration Integration (M1.3)
```python
# Future integration with models/calibration.py
from models.calibration import CalibrationMetrics, ConformalPredictor

# After inference
calibrated_probs = calibrator.calibrate(logits)
confidence_intervals = conformal.predict_interval(calibrated_probs)
```

## Model Architecture Details

### Exported Layers
```
Input (1,9,2,100)
    ↓
Conv2d(9→32) + BN + ReLU + MaxPool
    ↓
Conv2d(32→64) + BN + ReLU + MaxPool  
    ↓
Conv2d(64→128) + BN + ReLU
    ↓
GlobalAvgPool → Flatten
    ↓
Linear(128→8)  # FSQ projection
    ↓
Linear(8→64) + ReLU + Linear(64→10)
    ↓
Output (1,10)  # Behavioral logits
```

### Not Exported
- FSQ quantization grid (applied post-inference)
- HSMM temporal modeling (server-side if needed)
- HDP clustering (removed per ablation)
- Calibration metrics (post-processing)

## Validation Checklist

### Before Export
- [x] Model in eval mode
- [x] BatchNorm fused with Conv2d
- [x] Dropout disabled (p=0)
- [x] No dynamic shapes
- [x] No unsupported operations

### After Export  
- [ ] ONNX validated and simplified
- [ ] Latency benchmarked (CPU baseline)
- [ ] Hailo compatibility checked
- [ ] Calibration dataset generated
- [ ] Compilation script created

### Deployment Validation
- [ ] HEF file < 32MB
- [ ] Latency < 100ms (P95)
- [ ] Core inference < 15ms
- [ ] Accuracy ≥ 85%
- [ ] ECE ≤ 3% (M1.3)

## Troubleshooting

### Common Issues

1. **Unsupported Operations**
   - Check `hailo8_hef_compilation_checklist.md` for allowed ops
   - Replace GELU/Swish with ReLU
   - Remove LayerNorm/GroupNorm

2. **Memory Overflow**
   - Reduce model channels
   - Use INT8 quantization
   - Split into sub-graphs if needed

3. **Low Accuracy After Quantization**
   - Increase calibration samples (>1000)
   - Use representative data distribution
   - Consider quantization-aware training

4. **Compilation Failures**
   - Verify all ops in supported list
   - Check tensor shapes are static
   - Ensure no training layers remain

## Performance Optimization Tips

1. **Channel Alignment**: Use multiples of 8/16 for channels
2. **Stride Patterns**: Powers of 2 (1, 2, 4)
3. **Kernel Sizes**: Standard sizes (3x3, 5x5, 7x7)
4. **Batch Size**: Always 1 for edge inference
5. **Quantization**: INT8 symmetric for best performance

## Files Reference

### Core Scripts
- `export_fsq_for_hailo.py` - Main FSQ export pipeline
- `hailo8_hef_compilation_checklist.md` - Comprehensive guide

### Generated Artifacts
- `*.onnx` - ONNX models (original and simplified)
- `calibration_data.npy` - INT8 calibration dataset
- `compile_for_hailo8.sh` - Auto-generated compilation script
- `*.hef` - Compiled Hailo executable

## Next Steps for M1.3

1. **Calibration Integration**
   ```bash
   # Integrate existing calibration.py
   python integrate_calibration.py \
       --model conv2d_fsq_behavioral.onnx \
       --calibration models/calibration.py
   ```

2. **Accuracy Improvement**
   - Data augmentation pipeline
   - Ensemble methods
   - Active learning for hard samples

3. **Hardware Validation**
   - Deploy to actual Hailo-8
   - Profile latency and throughput
   - Validate against requirements

## Resources

- [Hailo Dataflow Compiler](https://hailo.ai/developer-zone/documentation/)
- [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [M1.2 Checkpoint Report](../Shared/Conv2D/Agent_Reviews/M1_2_GATE_REVIEW.yaml)
- [FSQ Implementation](../models/conv2d_fsq_model.py)

## Version History

- **v2.0** (Sept 2025): FSQ implementation, HDP removed per M1.2
- **v1.0** (Aug 2025): Initial TCN-VAE export pipeline

---

*For M1.3 requirements and calibration integration, see the [M1.2 Committee Report](../Shared/Conv2D/Agent_Reviews/M1_2_COMMITTEE_REPORT.md)*