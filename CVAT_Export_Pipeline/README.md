# CVAT to Training Pipeline Integration

Complete pipeline for converting CVAT web client annotations into TCN-VAE training format.

## Pipeline Overview

```
CVAT Web Client → XML Download → 3-Step Processing → TCN-VAE Training Data
```

### Step-by-Step Flow:

1. **CVAT Web Client** → Export annotations as XML to this folder
2. **Script #1**: `cvat_points_to_skeleton.py` → Convert individual points to skeleton format
3. **Script #2**: `cvat_to_sleap_converter.py` → Convert skeleton to SLEAP dataset
4. **Script #3**: `sleap_to_training_integration.py` → Convert SLEAP to TCN-VAE training format

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)
```bash
# Run all 3 steps automatically
python run_full_pipeline.py \
    --cvat annotations.xml \
    --images /path/to/your/images \
    --output /path/to/training/data
```

### Option 2: Run Steps Individually
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

## File Structure

```
CVAT_SLEAP_Export_Pipeline/
├── README.md                           # This file
├── cvat_points_to_skeleton.py          # Script #1
├── cvat_to_sleap_converter.py          # Script #2
├── sleap_to_training_integration.py    # Script #3
├── run_full_pipeline.py                # Automation script
├── annotations.xml                     # Downloaded from CVAT
├── test_skeleton.xml                   # Test data
└── job_*_annotations_*.zip             # CVAT exports
```

## 24-Point Keypoint System

The pipeline uses a 24-point anatomically correct skeleton:

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

## Output Format

The pipeline generates training-ready data:

```
training_output/
├── training_sequences.json             # JSON format sequences
├── training_sequences.pkl              # Pickle format (faster loading)
├── keypoint_dataloader.py              # PyTorch DataLoader template
├── sequences/                          # Individual NumPy arrays
│   ├── sequence_0000.npy
│   ├── sequence_0001.npy
│   └── ...
└── metadata/
    └── dataset_metadata.json           # Dataset information
```

### Data Format
- **Sequences**: `[sequence_length, n_keypoints, 3]`
  - Dimension 0: Time steps (default 30 frames)
  - Dimension 1: 24 keypoints
  - Dimension 2: `[x, y, confidence]`

## Integration with Training Pipeline

After running the pipeline, integrate with your TCN-VAE training:

```python
# In your training script
from keypoint_dataloader import KeypointSequenceDataset
from torch.utils.data import DataLoader

# Load your converted data
dataset = KeypointSequenceDataset('training_sequences.pkl')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in your existing TCN-VAE training loop
for batch in dataloader:
    keypoints = batch['keypoints']      # [batch, seq_len, n_keypoints, 2]
    confidence = batch['confidence']    # [batch, seq_len, n_keypoints]
    # ... your training code
```

## Pipeline Configuration

### Common Parameters

- `--sequence-length`: Length of training sequences (default: 30)
- `--stride`: Overlapping stride for sequence extraction (default: 10)
- `--create-loader`: Generate PyTorch DataLoader template

### CVAT Export Settings

In CVAT web client:
1. Use individual point annotations (not skeletons - avoids CVAT bug)
2. Export as "CVAT for images 1.1" format
3. Include visibility attributes: `visible`, `occluded`, `absent`

## Troubleshooting

### Common Issues

1. **Missing keypoints**: Pipeline handles missing points gracefully
2. **SLEAP import errors**: Install with `conda install sleap -c sleap -c nvidia -c conda-forge`
3. **Low sequence count**: Check that images are accessible and annotations have sufficient keypoints

### Quality Requirements

- Minimum 10 keypoints per image for skeleton creation
- Minimum 70% of frames valid for sequence inclusion
- At least 50% keypoints visible per frame

## Pipeline Statistics

The pipeline provides detailed statistics:

```
📊 Conversion Statistics:
  Frames processed: 150
  Sequences created: 45
  Valid keypoints: 2,847
  Missing keypoints: 753
  Keypoint validity: 79.10%
```

## Next Steps

1. **Test your data**: Run the generated `keypoint_dataloader.py`
2. **Integrate**: Modify your existing training scripts to use the new data format
3. **Validate**: Check sequence quality and temporal consistency
4. **Train**: Use with your TCN-VAE model in `../training/`

## Requirements

- Python 3.8+
- SLEAP
- OpenCV (cv2)
- PyTorch
- NumPy
- Pillow (PIL)

## Links

- **Training Scripts**: `../training/`
- **CVAT Documentation**: See `../CVAT_ANNOTATION_GUIDE.md`
- **Skeleton Visualization**: `../dog_skeleton_24points_universal.html`