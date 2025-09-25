# Hailo Architecture Fix: YOLOv8s Compilation Issue

**Date**: 2025-09-06  
**Issue**: YOLOv8s compiled for Hailo8L instead of Hailo8  
**Impact**: ~25% performance loss (52.5 FPS instead of ~65 FPS)  
**Solution**: Recompile with correct `--hw-arch hailo8` flag

---

## 🎯 Problem Summary

From the GPUSrv handoff document:
- **Current State**: YOLOv8s detection running at 52.5 FPS
- **Issue**: Model compiled for Hailo8L (lower performance variant)
- **Target**: Hailo8 (full NPU performance)
- **Expected Gain**: ~25% performance improvement → ~65 FPS

## 🔧 Root Cause Analysis

**Compilation Command Issue**:
```bash
# ❌ INCORRECT (what was likely used):
hailomz compile --hw-arch hailo8l  # Lower performance variant

# ✅ CORRECT (what should be used):
hailomz compile --hw-arch hailo8   # Full performance variant
```

**Architecture Differences**:
| Feature | Hailo8L | Hailo8 |
|---------|---------|--------|
| TOPS | ~13 TOPS | 26 TOPS |
| Performance | Baseline | +25% faster |
| Power | Lower | Higher (but within 5W budget) |
| Cost | Lower | Higher |

## 🚀 Solution Implementation

### Step 1: Verify Current State

On GPUSrv system:
```bash
# Check existing models
cd /path/to/models
ls -la *.hef

# Verify architecture (if hailo tools available)
hailo info yolov8s.hef | grep -i architecture
```

### Step 2: Download and Export YOLOv8s

```bash
# Install prerequisites
pip install ultralytics onnx

# Download and export
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    opset=11,
    batch=1,
    device='cpu',
    dynamic=False
)
"
```

### Step 3: Compile for Hailo8 (Critical Fix)

```bash
# CRITICAL: Use hailo8 not hailo8l
hailomz compile yolov8s.onnx \
    --hw-arch hailo8 \
    --performance \
    --optimization max \
    --batch-size 1 \
    --output yolov8s_hailo8_fixed.hef
```

**Key Parameters**:
- `--hw-arch hailo8` ← **CRITICAL FIX**
- `--performance` ← Enable performance mode
- `--optimization max` ← Maximum optimization

### Step 4: Deploy to Pi

```bash
# Transfer corrected HEF
scp yolov8s_hailo8_fixed.hef pi@raspberrypi.local:/home/pi/models/

# Update EdgeInfer configuration
ssh pi@raspberrypi.local
cd /home/pi/models
sudo systemctl stop edgeinfer
# Update config to use new HEF
sudo systemctl start edgeinfer
```

### Step 5: Validate Performance Improvement

```bash
# On Pi - benchmark new model
python3 /home/pi/hailo_tools/benchmark.py --model yolov8s_hailo8_fixed.hef

# Expected results:
# - FPS: ~65 (vs 52.5 baseline)
# - Latency: ~15ms (vs ~19ms)
# - TOPS usage: Similar but more efficient
```

## 📊 Expected Performance Improvements

### Before Fix (Hailo8L):
- **FPS**: 52.5
- **Latency**: ~19ms
- **Architecture**: hailo8l (suboptimal)
- **TOPS Utilization**: Inefficient

### After Fix (Hailo8):
- **FPS**: ~65 (+24% improvement)
- **Latency**: ~15ms (-21% reduction)
- **Architecture**: hailo8 (optimal)
- **TOPS Utilization**: Efficient

### Multi-Model Impact:
With parallel model execution:
- **TCN-VAE**: 20ms (unchanged)
- **YOLOv8s**: 15ms (improved from 19ms)
- **Human Pose**: 25ms (target)
- **Dog Pose**: 22ms (target)
- **Combined**: 42ms total (well under 50ms target)

## 🛠️ Automated Fix Scripts

### 1. Architecture Verification
```bash
python scripts/verify_hailo_architecture.py
```

### 2. Automated Recompilation
```bash
python scripts/compile_yolov8_hailo8_fixed.py
```

### 3. Pi Deployment
```bash
bash export/deploy_yolov8s_hailo8.sh
```

## 🔍 Validation Checklist

- [ ] HEF compiled with `--hw-arch hailo8` flag
- [ ] File size reasonable (~10-50MB)
- [ ] Deployment to Pi successful
- [ ] EdgeInfer configuration updated
- [ ] Performance benchmark shows ~25% improvement
- [ ] Combined latency under 50ms target
- [ ] System stability maintained

## 📈 Performance Monitoring

### Benchmark Commands:
```bash
# FPS measurement
python3 benchmark_fps.py --model yolov8s_hailo8_fixed.hef --duration 60

# Latency measurement  
python3 benchmark_latency.py --model yolov8s_hailo8_fixed.hef --samples 1000

# Resource utilization
htop & 
watch -n 1 'cat /sys/class/thermal/thermal_zone0/temp'
```

### Success Criteria:
- FPS > 60 (vs 52.5 baseline)
- Latency < 17ms (vs 19ms baseline)  
- Temperature < 70°C
- Combined multi-model latency < 50ms

## 🎯 Sprint 3 Impact

**Timeline**: Fix needed before Sprint 3 starts (2025-09-09)
**Blockers Resolved**: Ethogram visualization performance optimized
**Benefits**:
- 25% faster pose detection
- Better real-time responsiveness  
- Headroom for additional models
- Improved user experience

## 📝 Learning Capture

**Root Cause**: Incorrect hardware architecture flag in compilation
**Prevention**: Always verify `--hw-arch` parameter matches target hardware
**Validation**: Benchmark performance after every compilation
**Documentation**: Architecture differences between Hailo variants

---

## 🚀 Quick Fix Commands

For immediate resolution on GPUSrv:

```bash
# 1. Export YOLOv8s to ONNX
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='onnx', imgsz=640, simplify=True, opset=11, batch=1, dynamic=False)"

# 2. Compile for Hailo8 (NOT Hailo8L)
hailomz compile yolov8s.onnx --hw-arch hailo8 --performance --output yolov8s_hailo8_fixed.hef

# 3. Deploy to Pi
scp yolov8s_hailo8_fixed.hef pi@raspberrypi.local:/home/pi/models/

# 4. Validate performance
ssh pi@raspberrypi.local 'python3 /home/pi/hailo_tools/benchmark.py --expect-fps 65'
```

**Expected Result**: ~25% performance improvement, ~65 FPS detection rate

---

*Architecture Fix Status*: 📋 **DOCUMENTED** → Ready for GPUSrv implementation