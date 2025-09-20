# CVAT to Training Pipeline Integration

Complete dual pipeline for converting CVAT web client annotations into TCN-VAE training format.
Supports both **pose estimation** (SLEAP) and **object detection** (YOLOv8) workflows.

## Dual Pipeline Overview

```
CVAT Web Client → XML Download → Dual Processing → TCN-VAE Training Data
                                      ↓
                              ┌───────────────────┐
                              │  Pose Pipeline    │  Object Detection Pipeline
                              │  (SLEAP)          │  (YOLOv8)
                              └───────────────────┘
```

## Pipeline Selection

### 🦴 SLEAP Pipeline (Pose Estimation)
**Best for**: Detailed movement analysis, gait studies, precise joint tracking
**Output**: 24-keypoint sequences `[seq_len, 24, 3]` (x, y, confidence)

### 🎯 YOLOv8 Pipeline (Object Detection)
**Best for**: Activity recognition, general movement patterns, computational efficiency
**Output**: Bounding box sequences `[seq_len, 5]` (center_x, center_y, width, height, confidence)

## Quick Start

### Option 1: SLEAP Pipeline (Pose Estimation)
```bash
# Full keypoint pose analysis
python run_full_pipeline.py \
    --cvat annotations.xml \
    --images /path/to/your/images \
    --output /path/to/pose/training/data
```

### Option 2: YOLOv8 Pipeline (Object Detection)
```bash
# Bounding box movement analysis
python run_yolo8_pipeline.py \
    --cvat annotations.xml \
    --images /path/to/your/images \
    --output /path/to/detection/training/data
```

### Option 3: Both Pipelines
```bash
# Run both pipelines from same CVAT data
python run_full_pipeline.py --cvat annotations.xml --images /path/to/images --output /path/to/pose_data
python run_yolo8_pipeline.py --cvat annotations.xml --images /path/to/images --output /path/to/detection_data
```

## Detailed Pipeline Flows

### 🦴 SLEAP Pipeline (3 Steps)

1. **CVAT Web Client** → Export annotations as XML to this folder
2. **Script #1**: `cvat_points_to_skeleton.py` → Convert individual points to skeleton format
3. **Script #2**: `cvat_to_sleap_converter.py` → Convert skeleton to SLEAP dataset
4. **Script #3**: `sleap_to_training_integration.py` → Convert SLEAP to TCN-VAE training format

### 🎯 YOLOv8 Pipeline (4 Steps)

1. **CVAT Web Client** → Export annotations as XML to this folder
2. **Script #1**: `cvat_points_to_skeleton.py` → Convert individual points to skeleton format *(shared)*
3. **Script #2**: `cvat_to_yolo8_converter.py` → Extract bounding boxes to YOLOv8 dataset
4. **Script #3**: Train YOLOv8 model *(optional)*
5. **Script #4**: `yolo8_to_training_integration.py` → Convert detections to TCN-VAE training format

## File Structure

```
CVAT_SLEAP_Export_Pipeline/
├── README.md                           # This file
│
├── Shared Scripts/
│   └── cvat_points_to_skeleton.py      # Step #1 (shared by both pipelines)
│
├── SLEAP Pipeline/
│   ├── cvat_to_sleap_converter.py      # Step #2 (SLEAP)
│   ├── sleap_to_training_integration.py # Step #3 (SLEAP)
│   └── run_full_pipeline.py            # SLEAP automation
│
├── YOLOv8 Pipeline/
│   ├── cvat_to_yolo8_converter.py      # Step #2 (YOLOv8)
│   ├── yolo8_to_training_integration.py # Step #4 (YOLOv8)
│   └── run_yolo8_pipeline.py           # YOLOv8 automation
│
├── Data/
│   ├── annotations.xml                 # Downloaded from CVAT
│   ├── test_skeleton.xml              # Test data
│   └── job_*_annotations_*.zip         # CVAT exports
```

## Data Requirements

### CVAT Annotations Must Include:

#### For SLEAP Pipeline:
- **Keypoints**: 24-point skeleton system (see below)
- **Visibility**: `visible`, `occluded`, `absent` attributes

#### For YOLOv8 Pipeline:
- **Bounding boxes**: `dog_bbox` rectangles
- **Attributes**: `breed_size`, `pose_type`, `difficult`, `crowd`

#### Both Pipelines Can Use:
- Individual point annotations (converted to skeletons automatically)
- Mixed annotation types in same XML file

## 24-Point Keypoint System (SLEAP)

### Head (5 points)
- `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`

### Torso (3 points)
- `throat`, `withers`, `center`

### Front Legs (6 points)
- `left_front_shoulder`, `left_front_elbow`, `left_front_paw`
- `right_front_shoulder`, `right_front_elbow`, `right_front_paw`

### Back Legs (6 points)
- `left_hip`, `left_knee`, `left_back_paw`
- `right_hip`, `right_knee`, `right_back_paw`

### Tail (4 points)
- `tail_base`, `tail_mid_1`, `tail_mid_2`, `tail_tip`

## Output Formats

### SLEAP Pipeline Output
```
sleap_training_output/
├── training_sequences.json             # JSON format sequences
├── training_sequences.pkl              # Pickle format (faster loading)
├── keypoint_dataloader.py              # PyTorch DataLoader template
├── sequences/                          # Individual NumPy arrays
│   ├── sequence_0000.npy              # [30, 24, 3] per sequence
│   └── ...
└── metadata/
    └── dataset_metadata.json           # Dataset information
```

### YOLOv8 Pipeline Output
```
yolo_training_output/
├── yolo_training_sequences.json        # JSON format sequences
├── yolo_training_sequences.pkl         # Pickle format (faster loading)
├── yolo_dataloader.py                  # PyTorch DataLoader template
├── sequences/                          # Individual NumPy arrays
│   ├── yolo_sequence_0000.npy         # [30, 5] per sequence
│   └── ...
└── metadata/
    └── yolo_integration_metadata.json  # Integration information
```

## Data Integration Examples

### SLEAP Pipeline Integration
```python
# In your TCN-VAE training script
from keypoint_dataloader import KeypointSequenceDataset
from torch.utils.data import DataLoader

# Load pose sequences
dataset = KeypointSequenceDataset('training_sequences.pkl')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in training loop
for batch in dataloader:
    keypoints = batch['keypoints']      # [batch, 30, 24, 2]
    confidence = batch['confidence']    # [batch, 30, 24]
    # ... your pose analysis code
```

### YOLOv8 Pipeline Integration
```python
# In your TCN-VAE training script
from yolo_dataloader import YOLODetectionDataset
from torch.utils.data import DataLoader

# Load detection sequences
dataset = YOLODetectionDataset('yolo_training_sequences.pkl')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in training loop
for batch in dataloader:
    bbox_features = batch['bbox_features']  # [batch, 30, 5]
    # Features: [center_x, center_y, width, height, confidence]
    # ... your movement analysis code
```

## Pipeline Configuration

### SLEAP Pipeline Parameters
- `--sequence-length`: Length of training sequences (default: 30)
- `--stride`: Overlapping stride for sequence extraction (default: 10)
- `--create-loader`: Generate PyTorch DataLoader template

### YOLOv8 Pipeline Parameters
- `--train-split`: Training data split (default: 0.8)
- `--val-split`: Validation data split (default: 0.1)
- `--confidence`: Detection confidence threshold (default: 0.5)
- `--train-yolo`: Actually train YOLOv8 model (default: use pretrained)

## Individual Script Usage

### Run Steps Individually (SLEAP)
```bash
# Step 1: Individual points → skeleton
python cvat_points_to_skeleton.py \
    --input annotations.xml \
    --output skeleton_annotations.xml

# Step 2: Skeleton → SLEAP
python cvat_to_sleap_converter.py \
    --cvat skeleton_annotations.xml \
    --images /path/to/images \
    --output dataset.slp

# Step 3: SLEAP → Training format
python sleap_to_training_integration.py \
    --sleap dataset.slp \
    --output /path/to/training/data \
    --create-loader
```

### Run Steps Individually (YOLOv8)
```bash
# Step 1: Individual points → skeleton (shared)
python cvat_points_to_skeleton.py \
    --input annotations.xml \
    --output skeleton_annotations.xml

# Step 2: Skeleton → YOLOv8 dataset
python cvat_to_yolo8_converter.py \
    --cvat skeleton_annotations.xml \
    --images /path/to/images \
    --output /path/to/yolo/dataset \
    --create-training-script

# Step 3: Train YOLOv8 (optional)
cd /path/to/yolo/dataset
python train_yolo8.py

# Step 4: YOLOv8 → Training format
python yolo8_to_training_integration.py \
    --yolo-dataset /path/to/yolo/dataset \
    --model /path/to/best.pt \
    --output /path/to/training/data \
    --create-loader
```

## Performance Comparison

| Aspect | SLEAP Pipeline | YOLOv8 Pipeline |
|--------|----------------|----------------|
| **Data Richness** | High (24 keypoints) | Medium (5 bbox features) |
| **Computation** | Heavy | Light |
| **Training Time** | Long | Short |
| **Precision** | Very High | Medium |
| **Use Cases** | Gait analysis, joint studies | Activity recognition, general movement |
| **File Sizes** | Large | Small |

## Troubleshooting

### Common Issues

1. **Missing keypoints (SLEAP)**: Pipeline handles missing points gracefully
2. **No bounding boxes (YOLOv8)**: Ensure `dog_bbox` labels exist in CVAT annotations
3. **SLEAP import errors**: Install with `conda install sleap -c sleap -c nvidia -c conda-forge`
4. **YOLOv8 import errors**: Install with `pip install ultralytics`
5. **Low sequence count**: Check that images are accessible and annotations are sufficient

### Quality Requirements

#### SLEAP Pipeline:
- Minimum 10 keypoints per image for skeleton creation
- Minimum 70% of frames valid for sequence inclusion
- At least 50% keypoints visible per frame

#### YOLOv8 Pipeline:
- At least one `dog_bbox` per image
- Bounding box coordinates within image bounds
- Minimum confidence threshold for detections

## TCN-VAE Training Pipeline Integration

Both pipelines are designed to integrate seamlessly with your existing TCN-VAE training scripts in `../training/`. Here's how to connect the outputs:

### Integration Architecture

```
CVAT Annotations → [SLEAP/YOLOv8 Pipeline] → Training Data → TCN-VAE Model → Behavioral Analysis
```

### Modifying Your TCN-VAE Training Scripts

#### For SLEAP Pipeline Integration:

```python
# In your existing training script (e.g., ../training/train_enhanced_overnight.py)
import sys
sys.path.append('../CVAT_SLEAP_Export_Pipeline')

from keypoint_dataloader import KeypointSequenceDataset
from torch.utils.data import DataLoader

class TCNVAETrainer:
    def __init__(self, model, device, learning_rate=1e-3):
        # Your existing initialization
        self.pose_dataloader = None

    def load_pose_data(self, pose_data_path):
        """Load SLEAP pose sequences"""
        dataset = KeypointSequenceDataset(f'{pose_data_path}/training_sequences.pkl')
        self.pose_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"Loaded pose data: {len(dataset)} sequences")
        print(f"Keypoint dimensions: {dataset.metadata['n_keypoints']} points × 2 coords")

    def train_with_pose_data(self):
        """Enhanced training loop with pose data"""
        for epoch in range(self.epochs):
            for batch in self.pose_dataloader:
                keypoints = batch['keypoints']      # [batch, seq_len, 24, 2]
                confidence = batch['confidence']    # [batch, seq_len, 24]

                # Reshape for TCN-VAE: [batch, seq_len, features]
                # Flatten keypoints: 24 points × 2 coords = 48 features
                pose_features = keypoints.reshape(keypoints.size(0), keypoints.size(1), -1)

                # Your existing TCN-VAE training code
                reconstructed, mu, logvar = self.model(pose_features)
                loss = self.compute_loss(reconstructed, pose_features, mu, logvar)

                # Include confidence weighting
                confidence_weights = confidence.mean(dim=2)  # [batch, seq_len]
                weighted_loss = loss * confidence_weights.unsqueeze(-1)

                # Backpropagation
                self.optimizer.zero_grad()
                weighted_loss.mean().backward()
                self.optimizer.step()
```

#### For YOLOv8 Pipeline Integration:

```python
# In your existing training script
import sys
sys.path.append('../CVAT_SLEAP_Export_Pipeline')

from yolo_dataloader import YOLODetectionDataset
from torch.utils.data import DataLoader

class TCNVAETrainer:
    def load_detection_data(self, detection_data_path):
        """Load YOLOv8 detection sequences"""
        dataset = YOLODetectionDataset(f'{detection_data_path}/yolo_training_sequences.pkl')
        self.detection_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"Loaded detection data: {len(dataset)} sequences")
        print(f"Feature dimensions: {dataset.metadata['feature_dim']} per frame")

    def train_with_detection_data(self):
        """Enhanced training loop with detection data"""
        for epoch in range(self.epochs):
            for batch in self.detection_dataloader:
                bbox_features = batch['bbox_features']  # [batch, seq_len, 5]
                # Features: [center_x, center_y, width, height, confidence]

                # Your existing TCN-VAE training code
                reconstructed, mu, logvar = self.model(bbox_features)
                loss = self.compute_loss(reconstructed, bbox_features, mu, logvar)

                # Extract confidence for weighting
                confidence = bbox_features[:, :, 4]  # [batch, seq_len]
                weighted_loss = loss * confidence.unsqueeze(-1)

                # Backpropagation
                self.optimizer.zero_grad()
                weighted_loss.mean().backward()
                self.optimizer.step()
```

### Multi-Modal Training (Both Pipelines)

```python
class MultiModalTCNVAETrainer(TCNVAETrainer):
    """Enhanced trainer using both pose and detection data"""

    def __init__(self, model, device, learning_rate=1e-3):
        super().__init__(model, device, learning_rate)
        self.pose_weight = 0.7      # Weight for pose loss
        self.detection_weight = 0.3  # Weight for detection loss

    def load_multimodal_data(self, pose_data_path, detection_data_path):
        """Load both pose and detection data"""
        self.load_pose_data(pose_data_path)
        self.load_detection_data(detection_data_path)

    def train_multimodal(self):
        """Train with both data modalities"""
        pose_iter = iter(self.pose_dataloader)
        detection_iter = iter(self.detection_dataloader)

        for epoch in range(self.epochs):
            try:
                # Get batches from both modalities
                pose_batch = next(pose_iter)
                detection_batch = next(detection_iter)

                # Process pose data (reshape to match detection dimensions)
                pose_features = pose_batch['keypoints'].reshape(
                    pose_batch['keypoints'].size(0),
                    pose_batch['keypoints'].size(1), -1
                )

                # Process detection data
                bbox_features = detection_batch['bbox_features']

                # Forward pass for both modalities
                pose_recon, pose_mu, pose_logvar = self.model(pose_features)
                detection_recon, det_mu, det_logvar = self.model(bbox_features)

                # Compute weighted combined loss
                pose_loss = self.compute_loss(pose_recon, pose_features, pose_mu, pose_logvar)
                detection_loss = self.compute_loss(detection_recon, bbox_features, det_mu, det_logvar)

                total_loss = (self.pose_weight * pose_loss +
                            self.detection_weight * detection_loss)

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            except StopIteration:
                # Reset iterators
                pose_iter = iter(self.pose_dataloader)
                detection_iter = iter(self.detection_dataloader)
```

### Modifying Existing Training Scripts

To integrate with your current training scripts in `../training/`, add these imports and modifications:

#### For `train_enhanced_overnight.py`:
```python
# Add at the top
from pathlib import Path
import sys
pipeline_path = Path(__file__).parent.parent / 'CVAT_SLEAP_Export_Pipeline'
sys.path.append(str(pipeline_path))

# Your choice of dataloader
from keypoint_dataloader import KeypointSequenceDataset  # For SLEAP
# OR
from yolo_dataloader import YOLODetectionDataset         # For YOLOv8

# In your main training function
def main():
    # Load your converted data
    if args.data_type == 'pose':
        dataset = KeypointSequenceDataset(args.data_path + '/training_sequences.pkl')
    elif args.data_type == 'detection':
        dataset = YOLODetectionDataset(args.data_path + '/yolo_training_sequences.pkl')

    # Continue with your existing training code...
```

### Command Line Integration

Update your training scripts to accept the new data formats:

```bash
# Train with SLEAP pose data
python ../training/train_enhanced_overnight.py \
    --data-type pose \
    --data-path /path/to/sleap/output \
    --epochs 100

# Train with YOLOv8 detection data
python ../training/train_enhanced_overnight.py \
    --data-type detection \
    --data-path /path/to/yolo/output \
    --epochs 100

# Multi-modal training
python ../training/train_enhanced_overnight.py \
    --data-type multimodal \
    --pose-data /path/to/sleap/output \
    --detection-data /path/to/yolo/output \
    --epochs 100
```

### Behavioral Analysis Applications

#### With SLEAP Data:
- **Gait Analysis**: Analyze limb coordination patterns
- **Joint Movement**: Study specific joint trajectories
- **Pose Classification**: Classify sitting, standing, walking, etc.
- **Abnormality Detection**: Identify irregular movement patterns

#### With YOLOv8 Data:
- **Activity Recognition**: Classify behaviors from movement patterns
- **Spatial Analysis**: Analyze movement in space
- **Trajectory Prediction**: Predict future movement paths
- **Interaction Detection**: Identify social behaviors

## Next Steps

### After Running SLEAP Pipeline:
1. **Test data**: Run `keypoint_dataloader.py`
2. **Integrate**: Modify TCN-VAE training scripts using examples above
3. **Analyze**: Use for detailed movement and gait analysis
4. **Validate**: Check keypoint temporal consistency

### After Running YOLOv8 Pipeline:
1. **Test data**: Run `yolo_dataloader.py`
2. **Integrate**: Modify TCN-VAE training scripts using examples above
3. **Analyze**: Use for activity recognition and general movement patterns
4. **Validate**: Check bounding box temporal consistency

### Combining Both Pipelines:
1. **Multi-modal analysis**: Use both pose and detection features
2. **Feature fusion**: Combine keypoint precision with detection efficiency
3. **Validation**: Cross-validate results between pipelines
4. **Ensemble methods**: Use both for robust behavioral analysis

## Requirements

### Core Requirements (Both Pipelines)
- Python 3.8+
- PyTorch
- NumPy
- Pillow (PIL)
- OpenCV (cv2)

### SLEAP Pipeline Additional
- SLEAP
- conda environment recommended

### YOLOv8 Pipeline Additional
- ultralytics
- YOLO models

## Links

- **Training Scripts**: `../training/`
- **CVAT Documentation**: See `../CVAT_ANNOTATION_GUIDE.md`
- **Skeleton Visualization**: `../dog_skeleton_24points_universal.html`
- **YOLOv8 Documentation**: https://docs.ultralytics.com/