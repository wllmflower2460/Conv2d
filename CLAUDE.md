# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Conv2d-VQ-HDP-HSMM** project, implementing a revolutionary architecture for behavioral synchrony analysis that bridges discrete and continuous models with full uncertainty quantification. 

**Key Achievement**: Complete implementation of the unified behavioral synchrony framework combining:
- Vector Quantization (VQ) for discrete behavioral codes
- Hierarchical Dirichlet Process (HDP) for automatic cluster discovery
- Hidden Semi-Markov Model (HSMM) for temporal dynamics
- Entropy-based uncertainty quantification for clinical deployment

The architecture achieves **78.12% accuracy** on quadruped behavioral recognition and is production-ready for deployment to Hailo-8 accelerated edge devices.

## High-Level Architecture

### Core Model Pipeline
```
IMU Data (B,9,2,100) → Conv2d Encoder → VQ Quantization → HDP Clustering → HSMM Dynamics
                              ↓              ↓                ↓              ↓
                         Features      Discrete Codes    Behaviors    Temporal States
                                              ↓                           ↓
                                    Entropy & Uncertainty Module
                                              ↓
                                  Confidence-Calibrated Output
```

### Key Architectural Components

1. **Vector Quantization (VQ)** (`models/vq_ema_2d.py`, `models/conv2d_vq_model.py`):
   - EMA-based codebook learning with 512 codes × 64 dimensions
   - Straight-through estimator for gradient flow
   - Perplexity monitoring for codebook utilization
   - Hailo-safe implementation

2. **Hierarchical Dirichlet Process (HDP)** (`models/hdp_components.py`):
   - Stick-breaking construction for non-parametric clustering
   - Automatic discovery of behavioral clusters
   - Temperature annealing for training stability
   - Hierarchical organization of behaviors

3. **Hidden Semi-Markov Model (HSMM)** (`models/hsmm_components.py`):
   - Explicit duration modeling (negative binomial, Poisson, Gaussian)
   - Forward-backward algorithm and Viterbi decoding
   - Input-dependent transition matrices
   - Realistic behavioral persistence modeling

4. **Entropy & Uncertainty** (`models/entropy_uncertainty.py`):
   - Shannon entropy for discrete states
   - Circular statistics for phase analysis
   - Mutual information I(Z;Φ) calculation
   - Confidence calibration with ECE and Brier scores

5. **Complete Integration** (`models/conv2d_vq_hdp_hsmm.py`):
   - Full pipeline from IMU to behavioral analysis
   - 313K parameters (compact and efficient)
   - Multiple prediction heads
   - Comprehensive uncertainty quantification

3. **Multi-Dataset Integration** (`preprocessing/enhanced_pipeline.py`):
   - Factory pattern for dataset selection (cross-species vs traditional HAR)
   - Support for WISDM, HAPT, PAMAP2, UCI-HAR, TartanIMU datasets
   - Unified preprocessing with configurable windowing (100 timesteps default)

4. **Deployment Pipeline**:
   - ONNX export for edge deployment (`export_best_model.py`)
   - Hailo-8 compilation scripts (`scripts/compile_tcn_vae_hailo8.py`)
   - CoreML conversion for iOS (`CoreML_Pipeline/scripts/convert_to_coreml.py`)

## Common Development Commands

### Conv2d-VQ-HDP-HSMM Commands

```bash
# Train the Conv2d-VQ model
python training/train_conv2d_vq.py

# Test complete architecture
python models/conv2d_vq_hdp_hsmm.py

# Test individual components
python models/vq_ema_2d.py           # Test VQ layer
python models/hdp_components.py      # Test HDP clustering
python models/hsmm_components.py     # Test HSMM dynamics
python models/entropy_uncertainty.py # Test uncertainty module

# Analyze learned behavioral codes
python analysis/codebook_analysis.py --checkpoint models/best_conv2d_vq_model.pth
```

### Legacy TCN-VAE Commands

```bash
# Basic training (50 epochs)
python training/train_tcn_vae.py

# Overnight optimized training (up to 500 epochs with early stopping)
python train_overnight.py

# Quadruped-specific training (completed with 78.12% accuracy)
python train_quadruped_overnight.py

# Enhanced multi-dataset training
python training/train_enhanced_overnight.py

# CPU-only training
python training/train_overnight_cpu.py
```

### Evaluation and Export

```bash
# Evaluate trained model with visualizations
python evaluation/evaluate_model.py

# Export to ONNX for edge deployment
python export_best_model.py

# Export for Hailo-8 compilation
python hailo_export/export_hailo.py

# Convert to CoreML for iOS
python CoreML_Pipeline/scripts/convert_to_coreml.py
```

### Hailo-8 Deployment

```bash
# Verify Hailo architecture (important for optimization)
python scripts/verify_hailo_architecture.py

# Compile TCN-VAE for Hailo-8
python scripts/compile_tcn_vae_hailo8.py

# Fixed YOLOv8 compilation (+25% FPS improvement)
python scripts/compile_yolov8_hailo8_fixed.py

# Deploy to T3.2a package
python t3_2a_deployment_package/execute_t3_2a.py
```

### Monitoring and Testing

```bash
# Monitor training progress
bash scripts/monitor_training.sh
tail -f enhanced_training.log

# Generate progress report
python scripts/progress_report.py

# Test 24-point pipeline
python test_24point_pipeline.py

# Run tensor caching tests
python tests/test_tensor_caching.py
```

### Data Pipeline Commands

```bash
# CVAT to SLEAP conversion
python sleap_integration/cvat_to_sleap_converter.py

# CVAT to YOLOv8 conversion  
python Pipeline_CVAT_Inguest/cvat_to_yolo8_converter.py

# Run full CVAT export pipeline
python CVAT_Export_Pipeline/run_full_pipeline.py

# Stanford Dogs keypoint analysis
python analyze_stanford_keypoints.py
```

## Key File Locations

### Models and Training
- `models/tcn_vae.py` - Core TCN-VAE architecture
- `models/vq_ema_2d.py` - VQ-VAE-2D implementation
- `models/device_attention.py` - Attention mechanisms
- `models/final_quadruped_tcn_vae.pth` - Production-ready quadruped model (78.12%)
- `models/quadruped_processor.pkl` - Preprocessing pipeline

### Configuration
- `config/training_config.py` - Training hyperparameters
- `configs/improved_config.py` - Enhanced configuration
- `configs/enhanced_dataset_schema.yaml` - Dataset schema

### Preprocessing
- `preprocessing/enhanced_pipeline.py` - Multi-dataset factory pattern
- `preprocessing/quadruped_pipeline.py` - Quadruped-specific preprocessing  
- `preprocessing/unified_pipeline.py` - Unified preprocessing utilities

### Deployment Assets
- `hailo_export/tcn_encoder_for_edgeinfer.onnx` - Edge-ready ONNX model
- `t3_2a_deployment_package/` - Sprint 3 T3.2a deployment package
- `CoreML_Pipeline/models/` - iOS-ready models

## Development Workflow

1. **Dataset Preparation**: Place datasets in appropriate directories or use preprocessing pipelines
2. **Training**: Use overnight training scripts for best results
3. **Evaluation**: Run evaluation script to generate confusion matrices and t-SNE visualizations
4. **Export**: Convert to ONNX/CoreML format for deployment
5. **Compilation**: Use Hailo scripts for edge device optimization
6. **Deployment**: Follow platform-specific deployment guides

## Performance Targets

- **Quadruped Model**: 78.12% accuracy achieved (86.8% of 90% target)
- **Model Size**: <10MB for edge deployment
- **Inference Speed**: Sub-100ms on Hailo-8
- **Training Time**: ~7 minutes to convergence on RTX 2060

## Important Notes

- Always verify Hailo architecture before compilation (`scripts/verify_hailo_architecture.py`)
- Use the factory pattern in `preprocessing/enhanced_pipeline.py` to switch between approaches
- Models are stored with checkpoints for recovery from interruptions
- The project uses Conv2d dimensions for Hailo-8 compatibility (Conv1d not supported)