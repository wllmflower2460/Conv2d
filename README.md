# 🧪 TCN-VAE Training Environment

**Complete training pipeline for behavioral analysis models with multi-dataset integration**  
**Focus**: Model development, dataset integration, training execution, and deployment preparation

[![Quadruped Model](https://img.shields.io/badge/Quadruped-78.12%25%20Ready-green.svg)](#quadruped-training-completed)
[![Enhanced Pipeline](https://img.shields.io/badge/Enhanced-5%20Datasets-blue.svg)](#enhanced-multi-dataset)
[![Deployment Ready](https://img.shields.io/badge/Deployment-Ready%20for%20Sprint%203-orange.svg)](#sprint-3-deployment)

## 🎯 Sprint 3 Deployment Status

This repository has produced **deployment-ready models and fixes** for Sprint 3:

### ✅ Completed Training Results
- **Quadruped Model**: 78.12% accuracy (86.8% of 90% target) - **Ready for production**
- **Enhanced Pipeline**: 5-dataset integration validated (WISDM+HAPT+existing)
- **Hailo8 Architecture Fix**: YOLOv8s performance improvement scripts (+25% FPS)

### 🚀 Ready for Cross-Repository Deployment
- Models → Copy to `tcn-vae-models/` for production storage
- Hailo fix → Deploy to `hailo_pipeline/` for GPUSrv compilation  
- EdgeInfer → Integrate with `pisrv_vapor_docker/EdgeInfer/` for Pi deployment

## 🏆 Training Achievements (2025-09-06)

### Quadruped Behavioral Recognition (COMPLETED ✅)
- **78.12% validation accuracy** on dog behavior classification
- **72.01% F1 score** for behavioral transition detection
- **Training duration**: 350+ epochs with early stopping
- **Model size**: <10MB optimized for edge deployment

### Enhanced Multi-Dataset Integration (VALIDATED ✅)
- **5 datasets integrated**: WISDM, HAPT, PAMAP2, UCI-HAR, TartanIMU
- **Architecture compatibility**: Fixed domain classifier and model forward pass
- **Cross-dataset generalization**: Validated preprocessing pipeline
- **Deployment-ready**: Enhanced pipeline architecture proven

## 🚀 Sprint 3 Deployment Assets

### Trained Models (Ready for Production)
```bash
# Ready to copy to tcn-vae-models/:
models/final_quadruped_tcn_vae.pth      # 78.12% quadruped model
models/quadruped_processor.pkl          # Preprocessing pipeline
models/enhanced_recovery_model.pth      # Enhanced pipeline checkpoint
```

### Hailo Architecture Fix (Ready for GPUSrv)
```bash  
# Ready to copy to hailo_pipeline/:
scripts/verify_hailo_architecture.py    # Architecture detection
scripts/compile_yolov8_hailo8_fixed.py  # Fixed compilation (+25% FPS)
HAILO_ARCHITECTURE_FIX.md               # Complete documentation
DEPLOYMENT_QUICK_REFERENCE.md           # Deployment commands
```

### Training Documentation  
```bash
# Sprint 3 readiness documentation:
TRAINING_RESULTS_SUMMARY_2025-09-06.md  # Complete session results
DEPLOYMENT_QUICK_REFERENCE.md           # Immediate deployment guide
```

## 📁 Repository Structure (Training Environment)

```
tcn-vae-training/
├── models/
│   ├── tcn_vae.py              # TCN-VAE model architecture
│   ├── __init__.py
│   └── *.pth, *.pkl            # Trained model checkpoints
├── training/
│   ├── train_tcn_vae.py        # Basic training script
│   ├── train_extended.py       # Extended training utilities
│   └── __init__.py
├── preprocessing/
│   ├── unified_pipeline.py     # Multi-dataset preprocessing
│   └── __init__.py
├── evaluation/
│   ├── evaluate_model.py       # Model evaluation and visualization
│   ├── confusion_matrix.png    # Performance visualization
│   └── tsne_features.png       # Latent space visualization
├── export/
│   ├── model_config.json       # Model configuration
│   └── tcn_encoder_for_edgeinfer.onnx  # ONNX export
├── configs/
│   └── improved_config.py      # Training hyperparameters
├── scripts/
│   ├── monitor_training.sh     # Training monitoring
│   ├── progress_report.py      # Progress reporting
│   └── setup_monitoring.sh     # Monitoring setup
├── train_overnight.py          # Optimized overnight training
├── export_best_model.py        # ONNX export pipeline
├── TRAINING_STATUS.md          # Training progress log
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🆕 Dual Pipeline Approach (Conv2d + Traditional HAR)

### Primary: Cross-Species Conv2d Pipeline for Hailo-8
Our breakthrough approach that transforms Conv1d operations to Conv2d for Hailo compatibility while enabling cross-species behavioral transfer learning.

**Key Innovation**: Using the height dimension (H) as a "device dimension" to represent paired relationships:
- `Shape: (Batch, Channels, Devices, Time)` where Devices=2 for phone+collar IMU
- Enables Hailo-8 hardware acceleration (Conv1d not supported)
- Natural representation for synchrony analysis
- Numerically equivalent to Conv1d (proven with >0.99 cosine similarity)

### Fallback: Traditional HAR Multi-Dataset Pipeline  
Proven approach with support for WISDM, HAPT, PAMAP2, UCI-HAR, and TartanIMU datasets as a reliable fallback option.

### Factory Pattern for Easy Switching
```python
from preprocessing.enhanced_pipeline import get_dataset

# Use the innovative Conv2d approach (default)
dataset = get_dataset(
    approach='cross_species',
    config_path='configs/enhanced_dataset_schema.yaml',
    mode='train',
    enforce_hailo_constraints=True
)

# Or use traditional HAR approach as fallback
dataset = get_dataset(
    approach='traditional_har',
    window_size=100,
    overlap=0.5,
    base_dataset_path='./datasets'
)
```

## 🏗️ Architecture

### TCN-VAE Model Components

1. **TCN Encoder**: Temporal convolutional layers with dilated convolutions
2. **Variational Component**: Encodes to latent space with reparameterization
3. **TCN Decoder**: Reconstructs original sequences
4. **Activity Classifier**: Supervised learning head
5. **Domain Classifier**: Adversarial training for domain invariance

### Key Features

- **9-axis IMU input**: Accelerometer (3D) + Gyroscope (3D) + Magnetometer (3D)
- **100 timestep windows**: ~1 second at 100Hz sampling rate
- **64-dimensional latent space**: Compact representations
- **Multi-objective training**: Reconstruction + Classification + Domain adaptation

## 📊 Datasets

The pipeline supports three datasets:

1. **PAMAP2**: 19 activities, chest-mounted IMU sensor
2. **UCI-HAR**: 6 activities, smartphone sensors
3. **TartanIMU** (synthetic): 4 activities, proof of concept

All datasets are normalized and windowed consistently for unified training.

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib seaborn
pip install onnx onnxruntime
```

### Basic Training

```bash
# Train with default configuration
python training/train_tcn_vae.py

# Overnight training with optimized settings
python train_overnight.py
```

### Evaluation

```bash
# Evaluate trained model
python evaluation/evaluate_model.py

# Export to ONNX for deployment
python export_best_model.py
```

## 🔧 Configuration

Training parameters are configurable in `configs/improved_config.py`:

```python
IMPROVED_CONFIG = {
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 5e-4,
    "beta": 0.3,           # KL divergence weight
    "lambda_act": 3.0,     # Activity classification weight
    "lambda_dom": 0.05,    # Domain adaptation weight
    "grad_clip_norm": 0.5,
    "weight_decay": 1e-4
}
```

## 📈 Training Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 72.13% |
| Training Time | ~7 minutes to convergence |
| Model Parameters | 1.1M |
| Latent Dimension | 64 |

### Training Timeline

- **Epoch 1**: 66.57% accuracy (immediate improvement)
- **Epoch 5**: 71.98% (exceeded previous baseline)
- **Epoch 9**: 72.13% (current best)

## 🎯 Model Architecture Details

### TCN Encoder
```python
TemporalConvNet(
    input_dim=9,
    hidden_dims=[64, 128, 256],
    kernel_size=3,
    dropout=0.2
)
```

### VAE Components
```python
# Encoder outputs
mu = fc_mu(features)      # Mean vector [batch, 64]
logvar = fc_logvar(features)  # Log variance [batch, 64]

# Reparameterization
z = mu + eps * exp(0.5 * logvar)
```

### Loss Function
```python
total_loss = reconstruction_loss + β*kl_loss + λ_act*activity_loss + λ_dom*domain_loss
```

## 📤 Deployment

The trained model can be exported for edge deployment:

```bash
python export_best_model.py
```

This generates:
- `tcn_encoder_for_edgeinfer.onnx`: ONNX model for inference
- `model_config.json`: Configuration metadata
- Hailo-8 compatible format for Raspberry Pi deployment

## 🔍 Evaluation & Visualization

The evaluation script provides:

1. **Classification Report**: Per-class precision, recall, F1-score
2. **Confusion Matrix**: Visual performance breakdown
3. **t-SNE Visualization**: Latent space clustering
4. **Cross-dataset Performance**: Domain adaptation effectiveness

## 🛠️ Advanced Usage

### Custom Dataset Integration

```python
from preprocessing.unified_pipeline import MultiDatasetHAR

# Add custom dataset loader
processor = MultiDatasetHAR(window_size=100, overlap=0.5)
processor.load_custom_dataset(data_path, labels)
```

### Hyperparameter Tuning

```python
# Modify training configuration
trainer = TCNVAETrainer(model, device, learning_rate=1e-3)
trainer.beta = 0.5  # Adjust VAE weight
trainer.lambda_act = 2.0  # Adjust classification weight
```

## 📝 Development Notes

### Data Preprocessing
- Z-score normalization per channel
- Sliding window approach with 50% overlap
- Label encoding for unified cross-dataset training

### Training Optimizations
- Gradient clipping prevents explosion
- Cosine annealing with warm restarts
- Progressive adversarial training for domain adaptation
- Early stopping with patience

### Model Export
- Static shapes for Hailo-8 compatibility
- ONNX opset 11 for edge deployment
- Validation against PyTorch reference

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PAMAP2 dataset authors for comprehensive activity data
- UCI Machine Learning Repository for HAR dataset
- PyTorch team for the deep learning framework
- Hailo team for edge AI acceleration

## 📞 Contact

For questions about this implementation or collaboration opportunities, please open an issue in the repository.

---

**Status**: ✅ Production Ready | **Accuracy**: 72.13% | **Deployment**: Edge-optimized ONNX