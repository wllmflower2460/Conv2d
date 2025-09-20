# TCN-VAE Training Pipeline Setup Guide

Complete setup instructions for the TCN-VAE training pipeline environment.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (tested with Python 3.10)
- **CUDA**: 11.0+ for GPU training (optional but recommended)
- **RAM**: 16GB+ recommended for full dataset training
- **Storage**: 10GB+ available space
- **OS**: Linux (tested), macOS, Windows

### Hardware Recommendations
- **GPU**: NVIDIA RTX 2060 or better for training
- **CPU**: 8+ cores for data preprocessing
- **SSD**: For faster data loading during training

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/wllmflower2460/TCN-VAE_Training_Pipeline-.git
cd tcn-vae-training
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n tcn-vae python=3.10
conda activate tcn-vae

# Using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

#### Core Requirements
```bash
# PyTorch (check https://pytorch.org for latest)
pip install torch torchvision torchaudio

# Core ML libraries
pip install numpy pandas scikit-learn matplotlib seaborn

# ONNX export support
pip install onnx onnxruntime

# Optional: GPU acceleration
pip install onnxruntime-gpu  # If you have ONNX GPU support
```

#### Complete Requirements File
Create `requirements.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
onnx>=1.12.0
onnxruntime>=1.12.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📂 Dataset Setup

### Directory Structure
Create the following dataset directory structure:
```
datasets/
├── pamap2/
│   └── PAMAP2_Dataset/
│       └── Protocol/
│           ├── subject101.dat
│           ├── subject102.dat
│           └── ...
├── uci_har/
│   └── UCI HAR Dataset/
│       ├── train/
│       │   ├── X_train.txt
│       │   └── y_train.txt
│       └── test/
│           ├── X_test.txt
│           └── y_test.txt
└── tartan_imu/
    └── (synthetic data generated automatically)
```

### Dataset Downloads

#### PAMAP2 Dataset
1. Download from: [PAMAP2 Dataset](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)
2. Extract to `datasets/pamap2/`
3. Ensure `.dat` files are in `datasets/pamap2/PAMAP2_Dataset/Protocol/`

#### UCI-HAR Dataset
1. Download from: [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
2. Extract to `datasets/uci_har/`
3. Verify train/test directories exist

#### TartanIMU (Optional)
- Generated automatically by the preprocessing pipeline
- No manual download required

### Quick Dataset Verification
```bash
python -c "
import os
datasets = ['datasets/pamap2/PAMAP2_Dataset/Protocol/', 
            'datasets/uci_har/UCI HAR Dataset/train/']
for d in datasets:
    exists = os.path.exists(d)
    print(f'{d}: {'✓' if exists else '✗'}')
"
```

## 🏃‍♂️ Quick Start

### 1. Verify Setup
```bash
# Test data loading
python -c "
from preprocessing.unified_pipeline import MultiDatasetHAR
processor = MultiDatasetHAR()
print('Preprocessing pipeline loaded successfully!')
"

# Test model loading
python -c "
from models.tcn_vae import TCNVAE
model = TCNVAE()
print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
"
```

### 2. Run Training
```bash
# Basic training (50 epochs)
python training/train_tcn_vae.py

# Overnight training (optimized, up to 500 epochs)
python train_overnight.py
```

### 3. Evaluate Results
```bash
# Evaluate trained model
python evaluation/evaluate_model.py

# Export to ONNX
python export_best_model.py
```

## 🔧 Configuration

### Training Configuration
Edit `configs/improved_config.py` to customize training:

```python
IMPROVED_CONFIG = {
    # Training schedule
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 5e-4,
    
    # Loss weights
    "beta": 0.3,          # VAE KL weight
    "lambda_act": 3.0,    # Activity classification
    "lambda_dom": 0.05,   # Domain adaptation
    
    # Regularization
    "dropout_rate": 0.3,
    "weight_decay": 1e-4,
    "grad_clip_norm": 0.5
}
```

### Environment Variables
```bash
# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optional: Optimize for your hardware
export OMP_NUM_THREADS=8
```

## 📊 Expected Output

### Training Progress
```
🌙 Starting Overnight TCN-VAE Training...
⏰ Started at: 2024-01-01 00:00:00
🎯 Target: Beat 57.68% validation accuracy
🔧 Device: cuda
🚀 GPU: NVIDIA GeForce RTX 2060

📊 Loading datasets...
PAMAP2: 15432 windows
UCI-HAR: 10299 windows  
TartanIMU: 2400 windows
Total windows: 28131
Training: 22504, Validation: 5627
Classes: 13

🔢 Parameters: 1,100,345

[2024-01-01 00:05:23] Epoch 1 Complete:
  Train: Loss=2.3456, Acc=0.6657
  Val:   Loss=2.1234, Acc=0.6657
  Time: 42.1s, LR: 0.000300
  Best: 0.6657

[2024-01-01 00:10:45] Epoch 5 Complete:
  Train: Loss=1.8765, Acc=0.7198
  Val:   Loss=1.9234, Acc=0.7198 🔥 NEW BEST!
  Time: 38.2s, LR: 0.000285
  Best: 0.7198
```

### Final Results
```
🏁 Training Session Complete!
⏱️ Total time: 0.12 hours
🎯 Best validation accuracy: 0.7213
📈 Improvement: +25.1%

✅ Model exported successfully for EdgeInfer integration!
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch size: `batch_size = 32` or `batch_size = 16`
- Use CPU training: Set `device = torch.device('cpu')`
- Clear GPU cache: Add `torch.cuda.empty_cache()` periodically

#### Missing Dataset Files
```
FileNotFoundError: No valid PAMAP2 data loaded
```
**Solutions:**
- Check dataset directory structure
- Verify file permissions
- Re-download datasets if corrupted

#### Import Errors
```
ModuleNotFoundError: No module named 'models.tcn_vae'
```
**Solutions:**
- Check Python path: `export PYTHONPATH="${PYTHONPATH}:/path/to/tcn-vae-training"`
- Run from project root directory
- Verify all `__init__.py` files exist

#### Poor Training Performance
```
Validation accuracy stuck at ~30%
```
**Solutions:**
- Check data preprocessing
- Verify dataset quality
- Adjust learning rate
- Increase training epochs
- Check loss weights

### Performance Optimization

#### GPU Utilization
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check PyTorch GPU usage
python -c "
import torch
print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
print(f'GPU Utilization: {torch.cuda.utilization()}%')
"
```

#### CPU Optimization
```python
# Set optimal thread count
import torch
torch.set_num_threads(8)  # Adjust for your CPU
```

#### Memory Management
```python
# Clear caches periodically
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()
```

### Logging and Monitoring

#### Training Logs
```bash
# Monitor training progress
tail -f logs/overnight_training.jsonl

# Parse training history
python scripts/progress_report.py
```

#### Model Checkpoints
```bash
# List saved models
ls -la models/*.pth

# Check model size
du -h models/best_overnight_tcn_vae.pth
```

## 🚀 Advanced Setup

### Development Environment
```bash
# Install development dependencies
pip install pytest black flake8 jupyter

# Code formatting
black .

# Linting
flake8 .

# Running tests
pytest tests/
```

### Docker Setup (Optional)
```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "train_overnight.py"]
```

```bash
# Build and run
docker build -t tcn-vae .
docker run --gpus all -v $(pwd)/models:/workspace/models tcn-vae
```

### Multi-GPU Training (Advanced)
```python
# Enable DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

## ✅ Verification Checklist

Before starting training, verify:

- [ ] Python 3.8+ installed
- [ ] PyTorch with CUDA support (if using GPU)
- [ ] All dependencies installed
- [ ] Datasets downloaded and properly structured
- [ ] GPU memory sufficient (8GB+ recommended)
- [ ] Project directory writable for model checkpoints
- [ ] Preprocessing pipeline runs without errors
- [ ] Model creation succeeds

## 🔗 Next Steps

After successful setup:

1. **Start with basic training**: Run `python training/train_tcn_vae.py`
2. **Monitor progress**: Check logs and validation accuracy
3. **Optimize hyperparameters**: Edit config files as needed
4. **Scale to overnight training**: Use `python train_overnight.py`
5. **Evaluate results**: Run evaluation and export scripts
6. **Deploy models**: Follow ONNX export for edge deployment

## 📞 Support

If you encounter issues:

1. Check this troubleshooting section
2. Review error messages carefully
3. Verify all prerequisites are met
4. Check GitHub issues for similar problems
5. Create new issue with detailed error information

---

**Ready to train? Run `python train_overnight.py` to start your journey to 72%+ accuracy! 🎯**