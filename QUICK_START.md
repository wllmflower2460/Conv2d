# Quick Start Guide - Conv2d-FSQ Project

## Prerequisites

```bash
# Python 3.8+ with CUDA support
python --version
nvidia-smi  # Verify GPU

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install torch torchvision numpy scikit-learn matplotlib pyyaml
```

## Quick Training Pipeline

### 1. Generate Quadruped Data (2 min)

```bash
python setup_quadruped_datasets.py
```

This creates:
- 15,000 samples of quadruped locomotion data
- 10 behavioral classes (stand, walk, trot, gallop, etc.)
- Proper train/val/test splits (70/15/15)

### 2. Train FSQ Model with QA (10-30 min)

```bash
python train_fsq_simple_qa.py
```

Expected output:
```
M1.5 TRAINING: FSQ Model with Preprocessing QA
Goal: Recover M1.0-M1.2 performance (78.12%)
================================================================================
PREPROCESSING QUALITY ASSURANCE
  ✅ QA Complete - Data is ready for training
------------------------------------------------------------
TRAINING
Epoch   0: Train=0.2341, Val=0.3521, Best=0.3521
Epoch  10: Train=0.5234, Val=0.5867, Best=0.5867
...
✅ SUCCESS: Approaching M1.0-M1.2 performance!
```

### 3. Evaluate Model (1 min)

```bash
python evaluate_m15_simple.py
```

This demonstrates:
- Synthetic vs real data performance gap
- Proper evaluation methodology
- Honest metrics reporting

## Expected Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Target Accuracy** | 78.12% | M1.0-M1.2 baseline |
| **Expected Range** | 70-80% | With proper training |
| **Random Baseline** | 10% | 10-class problem |
| **Training Time** | 10-30 min | GPU dependent |

## Common Commands

### Check Training Progress
```bash
# Watch training in real-time
tail -f training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check saved models
ls -la *.pth
```

### Use Existing TartanVO/MIT Cheetah Data
```bash
# TartanVO (drone IMU)
cd quadruped_data/tartanvo/TartanVO
# Place IMU files here

# MIT Cheetah (robot)
cd quadruped_data/mit_cheetah/Cheetah-Software
# Run simulation or place data
```

## Troubleshooting

### Low Accuracy (<50%)
- Check preprocessing QA report
- Verify data loading correctly
- Increase training epochs

### GPU Memory Issues
- Reduce batch size in train_fsq_simple_qa.py
- Use gradient accumulation

### No GPU Available
- Models will train on CPU (slower)
- Reduce batch size to 16

## Key Differences from M1.4

| Aspect | M1.4 (Wrong) | M1.5 (Fixed) |
|--------|-------------|--------------|
| **Data** | Synthetic patterns | Real behavioral data |
| **Splits** | Same data for train/test | Temporal separation |
| **Accuracy** | 99.95% (fake) | 70-80% (honest) |
| **Evaluation** | Memorization | True generalization |

## Next Steps

After successful training:

1. **Deploy to Edge**:
   ```bash
   python deploy_pipeline.py --model m15_fsq_best_qa.pth
   ```

2. **Hailo Compilation**:
   ```bash
   python hailo_export/export_fsq_for_hailo.py
   ```

3. **Test on Real Hardware**:
   - Deploy to Raspberry Pi 5
   - Test with Hailo-8 accelerator

## Documentation

- Full pipeline: `TRAINING_PIPELINE.md`
- Dataset details: `DATASET_DOCUMENTATION.md`
- Preprocessing: `preprocessing/README.md`
- M1.5 resolution: `M1_5_GATE_REVIEW_RESOLUTION.md`

## Support

For issues or questions:
- Check existing documentation
- Review M1.5 gate resolution for context
- Ensure preprocessing QA passes before training