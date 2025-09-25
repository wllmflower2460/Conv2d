# T3.2a Deployment Package
## Sprint 3: Pose Model Compilation

### Quick Execution
```bash
# On system with Hailo SDK:
cd scripts/
python compile_tcn_vae_hailo8.py
python compile_yolo_pose_hailo8_t3_2a.py
```

### Requirements
- Hailo SDK installed (hailomz command available)
- Internet connection (for YOLO model downloads)
- Python 3.7+ with requests module

### Expected Output
- tcn_vae_72pct_hailo8.hef
- yolov8n_pose_human_hailo8.hef  
- yolov8s_pose_dog_hailo8.hef

### T3.2a Acceptance Criteria
✅ 3 HEF files compiled with --hw-arch hailo8
✅ Each model achieving >40 FPS
✅ 25% performance improvement vs hailo8l

Generated: 2025-09-06T19:46:13.456008
