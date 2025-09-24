# Quick Deployment Reference

## 🚀 Hailo Architecture Fix Deployment

### On GPUSrv System

#### 1. Verify Current Architecture
```bash
cd /path/to/tcn-vae-training
python scripts/verify_hailo_architecture.py
```

#### 2. Apply Hailo8 Fix  
```bash
python scripts/compile_yolov8_hailo8_fixed.py
```

#### 3. Deploy to Pi
```bash
bash export/deploy_yolov8s_hailo8.sh
```

### Expected Results
- **Before**: 52.5 FPS (Hailo8L)
- **After**: ~65 FPS (Hailo8) 
- **Improvement**: 25% performance gain

## 📊 Training Models Usage

### Quadruped Model (Completed)
```python
# Load trained quadruped model
import joblib
processor = joblib.load('models/overnight_processor.pkl')

# Use for dog behavior classification
predictions = processor.transform(sensor_data)  # 78.12% accuracy
```

### Enhanced Multi-Dataset (In Progress)  
```python
# Monitor training or load when complete
import torch
checkpoint = torch.load('models/best_enhanced_working.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 🔧 Script Locations

- `scripts/verify_hailo_architecture.py` - Architecture verification
- `scripts/compile_yolov8_hailo8_fixed.py` - Fixed compilation  
- `train_quadruped_overnight.py` - Completed quadruped training
- `train_enhanced_working.py` - Enhanced training (running)

## ⚡ Quick Commands

```bash
# Check Hailo architecture
python scripts/verify_hailo_architecture.py

# Fix YOLOv8s compilation  
python scripts/compile_yolov8_hailo8_fixed.py

# Monitor enhanced training
tail -f enhanced_training.log
```

Status: **Ready for GPUSrv deployment** 🚀