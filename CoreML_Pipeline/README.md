# CoreML Pipeline for Dog Behavior Classification

## Overview

This pipeline converts PyTorch TCN-VAE models trained on 24-point dog pose data into CoreML format for iOS deployment. The model classifies dog behaviors from pose sequences captured over time.

## Model Architecture

- **Input**: 24 keypoints (x, y coordinates) over 100 frames
  - Shape: `[1, 48, 100]` where 48 = 24 keypoints × 2 coordinates
- **Output**: 21 behavior class probabilities
- **Model Size**: ~1.5MB (optimized for mobile)
- **Accuracy**: 94.46% validation accuracy on Stanford Dogs dataset

## Directory Structure

```
CoreML_Pipeline/
├── scripts/              # Conversion scripts
│   ├── export_for_coreml.py    # Linux/Windows: PyTorch → ONNX export
│   ├── convert_to_coreml.py    # Full CoreML conversion (requires macOS)
│   └── macos_convert.py        # macOS-specific conversion script
├── models/               # Exported models
│   ├── dog_behavior_classifier.onnx
│   ├── dog_behavior_classifier.pth
│   └── model_metadata.json
├── docs/                 # Documentation
│   ├── ios_integration.md
│   └── troubleshooting.md
├── ios_examples/         # Swift code examples
│   ├── DogBehaviorPredictor.swift
│   └── PoseProcessor.swift
└── README.md            # This file
```

## Prerequisites

### For ONNX Export (Linux/Windows/Mac)
```bash
pip install torch torchvision numpy
```

### For CoreML Conversion (macOS only)
```bash
pip install coremltools torch torchvision numpy
```

## Step-by-Step Conversion Process

### Step 1: Export PyTorch Model to ONNX (Any Platform)

```bash
cd scripts
python export_for_coreml.py
```

This creates:
- `models/dog_behavior_classifier.onnx` - ONNX model
- `models/dog_behavior_classifier.pth` - PyTorch checkpoint
- `models/model_metadata.json` - Model specifications

### Step 2: Convert ONNX to CoreML (macOS Required)

Transfer the ONNX file to a Mac, then run:

```bash
cd scripts
python macos_convert.py
```

Or use the Python console:

```python
import coremltools as ct

# Load and convert ONNX model
model = ct.convert(
    'models/dog_behavior_classifier.onnx',
    convert_to='mlprogram',
    minimum_deployment_target=ct.target.iOS15
)

# Add metadata
model.author = 'TCN-VAE Dog Behavior Model'
model.short_description = '24-point dog pose to behavior classification'
model.version = '1.0'

# Save as ML Package (recommended for iOS 15+)
model.save('models/DogBehaviorClassifier.mlpackage')
```

## 24-Point Dog Skeleton Structure

The model expects poses in this keypoint order:

```json
[
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "throat", "withers", "left_front_shoulder", "left_front_elbow", "left_front_paw",
  "right_front_shoulder", "right_front_elbow", "right_front_paw",
  "center", "left_hip", "left_knee", "left_back_paw",
  "right_hip", "right_knee", "right_back_paw",
  "tail_base", "tail_mid_1", "tail_mid_2", "tail_tip"
]
```

## Behavior Classes (21 total)

```
sit, down, stand, stay, lying, heel, come, fetch, drop, wait,
leave_it, walking, trotting, running, jumping, spinning, rolling,
playing, alert, sniffing, looking
```

## iOS Integration

### 1. Add Model to Xcode Project

1. Drag `DogBehaviorClassifier.mlpackage` into your Xcode project
2. Ensure "Add to targets" includes your app target
3. Xcode automatically generates the model class

### 2. Basic Usage in Swift

```swift
import CoreML
import Vision

// Load the model
guard let model = try? DogBehaviorClassifier(configuration: MLModelConfiguration()) else {
    print("Failed to load model")
    return
}

// Prepare input: 24 keypoints over 100 frames
let poseSequence = MLMultiArray(shape: [1, 48, 100], dataType: .float32)
// ... fill with pose data ...

// Make prediction
if let prediction = try? model.prediction(pose_sequence: poseSequence) {
    let behaviorProbs = prediction.behavior_probs
    // Get most likely behavior
    let behavior = prediction.classLabel
    print("Detected behavior: \(behavior)")
}
```

### 3. Real-time Processing Pipeline

See `ios_examples/DogBehaviorPredictor.swift` for a complete implementation.

## Data Preprocessing

Before feeding poses to the model:

1. **Normalize coordinates**: Scale x,y to [0,1] based on image dimensions
2. **Handle missing keypoints**: Set to (0,0) or use last known position
3. **Temporal buffering**: Accumulate 100 frames (≈3.3 seconds at 30fps)
4. **Format conversion**: Flatten to [x0,y0,x1,y1,...,x23,y23] per frame

## Performance Optimization

- **Model Quantization**: The model is already optimized to 1.5MB
- **Batch Processing**: Process multiple dogs simultaneously if needed
- **Frame Skipping**: Can sample every 2-3 frames instead of every frame
- **Sliding Window**: Use overlapping windows for continuous prediction

## Troubleshooting

### Common Issues

1. **"Model input shape mismatch"**
   - Ensure input is exactly [1, 48, 100]
   - Check keypoint ordering matches expected structure

2. **"Low confidence predictions"**
   - Verify keypoint normalization
   - Check if enough keypoints are visible
   - Ensure temporal window has sufficient movement

3. **"Model not loading in iOS"**
   - Verify minimum iOS deployment target (iOS 14+)
   - Check model file isn't corrupted
   - Ensure sufficient device memory

## Testing the Model

### Python Test Script
```python
from scripts.export_for_coreml import InferenceOnlyModel
import torch

# Load model
model = InferenceOnlyModel(...)
model.eval()

# Test with synthetic data
test_input = torch.randn(1, 48, 100)
output = model(test_input)
print(f"Predictions: {output}")
```

### iOS Test
Use the included `ios_examples/DogBehaviorPredictor.swift` for testing.

## Model Updates

To update the model with new training:

1. Train new model with `train_with_stanford_dogs.py`
2. Re-run the export pipeline
3. Update version number in metadata
4. Re-deploy to iOS app

## License & Attribution

- Model trained on Stanford Dogs dataset
- Uses 24-point CVAT annotation structure
- TCN-VAE architecture for temporal behavior analysis

## Support

For issues or questions:
- Check `docs/troubleshooting.md`
- Review `ios_examples/` for implementation references
- Ensure all preprocessing steps match training pipeline

---
Last Updated: September 2024
Model Version: 1.0
Pipeline Version: 1.0