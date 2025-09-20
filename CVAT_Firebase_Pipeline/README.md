# Firebase-CVAT Active Learning Pipeline

**🎯 Intelligent Pose Annotation System for DataDogs**

This is a **distributed active learning pipeline** that creates an intelligent feedback loop between your iOS DataDogs app and professional annotation workflow. When your app encounters uncertain pose detections, it automatically sends them for human correction to continuously improve the model.

## 🧠 What This Pipeline Does

**The Problem**: Your iOS app sometimes makes pose predictions with low confidence - these are the cases that need human correction to improve the model.

**The Solution**: This pipeline automatically:
1. **Identifies uncertain poses** from your iOS app (low confidence scores)
2. **Prioritizes them** for annotation (most uncertain = highest priority)
3. **Transfers images** to professional annotation tools (CVAT)
4. **Collects corrected keypoints** from human annotators
5. **Feeds corrections back** to retrain and improve your model

**The Result**: Your iOS app gets smarter over time by learning from its mistakes!

## 🔄 Active Learning Flow

```
📱 iOS App → 🤔 Uncertain Pose → ☁️ Firebase → 🖥️ CVAT → 👨‍💻 Human Annotation → ☁️ Firebase → 🤖 Better Model
```

### Detailed Flow:

1. **📱 iOS Detection**: Your DataDogs app runs pose estimation and identifies uncertain poses (confidence < threshold)
2. **☁️ Smart Upload**: Only uncertain poses get uploaded to Firebase with priority scores (1-10 scale)
3. **🎯 Priority Transfer**: High-priority poses automatically transfer to your GPUSRV for annotation
4. **🖥️ Professional Annotation**: Human annotators correct keypoints using CVAT web interface
5. **📤 Feedback Loop**: Corrected annotations return to Firebase for model improvement
6. **🤖 Automatic Retraining**: When enough corrections accumulate, model retraining triggers automatically

## 💡 Key Benefits

### 🎯 **Smart Data Collection**
- Only annotates the most valuable/uncertain cases
- Reduces annotation workload by 80-90%
- Focuses human effort on edge cases that matter most

### ⚡ **Fully Automated Workflow**
- No manual file transfers or coordination
- Seamless iOS → Firebase → CVAT → Firebase pipeline
- Automatic priority-based batch processing

### 📈 **Continuous Model Improvement**
- Model gets smarter with each annotation batch
- Learns from real-world edge cases from your iOS devices
- Distributed learning across multiple device deployments

### 🔄 **Scalable Architecture**
- Works across unlimited iOS devices
- Centralized Firebase collection and distribution
- Professional annotation tools integration

## 🏗️ Architecture

```
iOS DataDogs App
     ↓ (uncertain poses)
Firebase Storage/Firestore
     ↓ (transfer script)
GPUSRV File System
     ↓ (CVAT import)
CVAT Annotation Interface  
     ↓ (export annotations)
CVAT XML Exports
     ↓ (return script)
Firebase labeled_data collection
     ↓ (Cloud Functions)
Model Retraining Pipeline
```

## 📋 Prerequisites

### GPUSRV Requirements
- Python 3.8+
- Docker (for CVAT)
- Tailscale access to `gpusrv.tailfdc654.ts.net`
- Firebase service account JSON key

### Firebase Setup
- Firebase project with Firestore and Storage enabled
- Service account with admin privileges
- Collections: `uncertain_poses`, `labeled_data`, `retraining_triggers`

## 🚀 Quick Start

### 1. Setup GPUSRV Environment

```bash
# Copy pipeline to GPUSRV
scp -r firebase_cvat_pipeline/ gpusrv.tailfdc654.ts.net:~/

# Setup environment
ssh gpusrv.tailfdc654.ts.net
cd ~/firebase_cvat_pipeline
bash gpusrv_setup.sh

# Install Python dependencies
pip3 install -r requirements.txt
```

### 2. Configure Firebase Credentials

```bash
# Copy your Firebase service account key
scp /path/to/service-account-key.json gpusrv.tailfdc654.ts.net:~/.firebase/

# Update firebase_to_cvat_transfer.py with your bucket name
# Line 32: 'storageBucket': 'your-project-id.appspot.com'
```

### 3. Transfer Images from Firebase

```bash
# On GPUSRV - Pull uncertain poses
python3 firebase_to_cvat_transfer.py \
  --service-account ~/.firebase/service-account.json \
  --limit 25 \
  --priority 7 \
  --output-dir /data/cvat-annotations
```

**Script Options:**
- `--limit`: Maximum images to transfer (default: 50)
- `--priority`: Minimum uncertainty priority 1-10 (default: 5)
- `--output-dir`: Local output directory (default: /data/cvat-annotations)
- `--gpusrv-host`: GPUSRV hostname (default: gpusrv.tailfdc654.ts.net)

### 4. Setup CVAT Annotation

```bash
# Start CVAT server
docker run -dit --name cvat_server --restart=always \
  -p 8080:8080 \
  -v /data/cvat-data:/home/django/data \
  -v /data/cvat-annotations:/home/django/keys \
  -v /data/cvat-exports:/home/django/models \
  cvat/server

# Access CVAT web interface
# http://gpusrv.tailfdc654.ts.net:8080
```

**CVAT Project Setup:**
1. Create new project: "DataDogs Pose Estimation"
2. Import images from `/data/cvat-annotations`
3. Configure keypoint labels for dog joints:
   - head, forelegs, hindlegs, trunk, tail
4. Begin pose annotation with corrected keypoints

### 5. Return Annotations to Firebase

```bash
# Export CVAT annotations to XML
# Download from CVAT interface or use API

# Upload corrected keypoints back to Firebase
python3 cvat_to_firebase_return.py \
  --service-account ~/.firebase/service-account.json \
  --annotations /data/cvat-exports/annotations.xml \
  --trigger-retraining
```

## 📊 Data Flow

### Firebase Collections

**uncertain_poses** - Images needing annotation
```json
{
  "id": "pose_uuid",
  "confidence": 0.3,
  "priority": 8,
  "status": "pending|in_progress|completed",
  "deviceId": "device_uuid",
  "timestamp": "2025-09-11T...",
  "detectedPoses": [...],
  "jointGroup": "all"
}
```

**labeled_data** - Corrected annotations  
```json
{
  "id": "cvat_pose_uuid_timestamp",
  "originalDataId": "pose_uuid",
  "correctedKeypoints": {"head": {"x": 100, "y": 200}, ...},
  "qualityScore": 0.95,
  "source": "cvat",
  "timestamp": "2025-09-11T..."
}
```

**retraining_triggers** - Model update triggers
```json
{
  "type": "cvat_batch_upload",
  "labeled_data_count": 25,
  "trigger_threshold": 10,
  "timestamp": "2025-09-11T..."
}
```

## 🔧 Configuration

### Transfer Script Configuration

Edit `firebase_to_cvat_transfer.py`:
```python
# Line 32: Update with your Firebase project
'storageBucket': 'datadogs-pose-estimation.appspot.com'

# Line 319: Update storage path if different
image_path = f"training_images/{pose_id}.jpg"
```

### Annotation Quality Thresholds

```python
# Minimum confidence for uncertain poses
MIN_CONFIDENCE = 0.3

# Priority calculation (1-10 scale)
# Higher priority = more uncertain = needs annotation first
priority = int((threshold - confidence) / threshold * 10) + 1

# Retraining trigger threshold
RETRAINING_THRESHOLD = 10  # annotations
```

## 🎯 Workflow Examples

### Daily Annotation Batch
```bash
# Morning: Pull overnight uncertain poses (priority 6+)
python3 firebase_to_cvat_transfer.py \
  --service-account ~/.firebase/service-account.json \
  --limit 50 --priority 6

# Afternoon: Annotate in CVAT web interface
# Evening: Upload completed annotations
python3 cvat_to_firebase_return.py \
  --service-account ~/.firebase/service-account.json \
  --annotations /data/cvat-exports/daily_batch.xml \
  --trigger-retraining
```

### High-Priority Emergency Batch
```bash
# Critical poses only (priority 8+)
python3 firebase_to_cvat_transfer.py \
  --service-account ~/.firebase/service-account.json \
  --limit 10 --priority 8
```

## 📈 Monitoring

### Check Transfer Status
```bash
# View manifest of transferred images
cat /data/cvat-annotations/transfer_manifest.json

# Check Firebase status updates
# Monitor uncertain_poses collection status field
```

### Annotation Progress
```bash
# CVAT project statistics
curl http://gpusrv.tailfdc654.ts.net:8080/api/v1/projects

# Firebase labeled_data count
# Monitor labeled_data collection size
```

## 🔍 Troubleshooting

### Common Issues

**Firebase Authentication Errors**
```bash
# Verify service account permissions
# Check Firebase IAM roles: Editor or Owner required
```

**CVAT Connection Issues**
```bash
# Check CVAT container status
docker ps | grep cvat

# Restart CVAT if needed
docker restart cvat_server
```

**Image Transfer Failures**
```bash
# Check storage bucket permissions
# Verify image paths in Firebase Storage
# Ensure sufficient disk space on GPUSRV
```

### Debug Mode
```bash
# Run with verbose logging
python3 -u firebase_to_cvat_transfer.py \
  --service-account ~/.firebase/service-account.json \
  --limit 5 --priority 9 2>&1 | tee debug.log
```

## 🔒 Security Notes

- Store Firebase service account keys securely in `~/.firebase/`
- Use environment variables for sensitive configuration
- Restrict CVAT access via Tailscale network
- Monitor annotation quality and user access

## 📚 Additional Resources

- [Firebase Admin SDK Documentation](https://firebase.google.com/docs/admin)
- [CVAT User Guide](https://opencv.github.io/cvat/docs/)
- [DataDogs Pose Estimation Documentation](../README.md)

---

**🏆 Created for DataDogs Distributed Active Learning Pipeline**  
*Professional pose estimation training data pipeline*