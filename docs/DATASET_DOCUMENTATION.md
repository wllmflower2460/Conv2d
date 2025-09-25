# Dataset Documentation - Conv2d-FSQ Project

## Overview

This document describes the datasets, preprocessing procedures, and quality assurance measures used in the Conv2d-FSQ behavioral analysis project. Following the M1.4 gate failure, we've established rigorous data handling procedures to ensure scientific validity.

## Table of Contents

1. [Available Datasets](#available-datasets)
2. [Preprocessing Pipeline](#preprocessing-pipeline)
3. [Quality Assurance](#quality-assurance)
4. [Data Splits](#data-splits)
5. [Training Procedures](#training-procedures)

## Available Datasets

### 1. Quadruped Locomotion Dataset (Primary)

**Source**: Synthetic generation based on MIT Cheetah & Boston Dynamics Spot patterns  
**Location**: `./quadruped_data/processed/`  
**Samples**: 15,000 (10,500 train / 2,250 val / 2,250 test)  
**Classes**: 10 behavioral modes  

#### Behavioral Classes:
| ID | Behavior | Description | Frequency (Hz) |
|----|----------|-------------|----------------|
| 0 | stand | Standing/stationary | 0.5 |
| 1 | walk | Walking gait (4-beat) | 1.5 |
| 2 | trot | Trotting (diagonal pairs) | 3.0 |
| 3 | gallop | Galloping (rotary gallop) | 4.0 |
| 4 | turn_left | Turning left | 1.0 |
| 5 | turn_right | Turning right | 1.0 |
| 6 | jump | Jumping | 0.5 |
| 7 | pronk | Pronking (all legs together) | 2.0 |
| 8 | backup | Backing up | 1.0 |
| 9 | sit_down | Sitting down transition | 0.5 |

#### Data Format:
- **Shape**: (N, 9, 100)
  - N: Number of samples
  - 9: IMU channels (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z)
  - 100: Time steps (0.5 seconds at 200Hz)
- **Type**: Float32
- **Range**: Approximately [-35, 45] with gravity component

### 2. TartanVO Drone IMU Dataset

**Source**: https://github.com/castacks/TartanVO  
**Location**: `./quadruped_data/tartanvo/`  
**Status**: Repository cloned, manual data download required  

#### Setup Instructions:
```bash
# Clone repository (already done)
cd quadruped_data/tartanvo/TartanVO

# Download IMU data from TartanAir dataset
# Visit: https://theairlab.org/tartanair-dataset/
# Download trajectories with IMU data
# Extract to: quadruped_data/tartanvo/
```

### 3. MIT Cheetah Robot Dataset

**Source**: https://github.com/mit-biomimetics/Cheetah-Software  
**Location**: `./quadruped_data/mit_cheetah/`  
**Status**: Repository cloned, simulation available  

#### Setup Instructions:
```bash
# Clone repository (already done)
cd quadruped_data/mit_cheetah/Cheetah-Software

# Build simulation (requires dependencies)
mkdir build && cd build
cmake ..
make -j4

# Run simulation to generate IMU data
./sim/sim
```

### 4. Simple Behavioral Dataset (Fallback)

**Location**: `./evaluation_data/`  
**Samples**: 10,000 (6,000 train / 2,000 val / 2,000 test)  
**Classes**: 5 (walking, running, turning, standing, jumping)  

## Preprocessing Pipeline

### Stage 1: Data Loading

```python
from preprocessing.enhanced_pipeline import EnhancedCrossSpeciesDataset
from preprocessing.movement_diagnostics import QualityControl, QualityThresholds

# Initialize quality control
qc = QualityControl(
    thresholds=QualityThresholds(
        max_nan_percentage=5.0,
        min_signal_std=0.01,
        max_signal_std=50.0,
        min_codebook_usage=0.2
    ),
    strict_mode=False
)
```

### Stage 2: Data Validation

The preprocessing pipeline performs the following validation steps:

1. **Shape Validation**
   - Expected: (B, 9, 100) for raw data
   - Reshaped to: (B, 9, 2, 50) for Conv2d compatibility

2. **Data Quality Checks**
   - NaN detection and replacement
   - Inf value clipping
   - Zero-variance channel detection
   - Signal-to-noise ratio assessment

3. **Statistical Validation**
   - Mean and standard deviation bounds
   - Value range verification
   - Class distribution balance

### Stage 3: Data Transformation

```python
# Reshape for Conv2d (required for Hailo-8)
def reshape_for_conv2d(data):
    """
    Reshape from (B, C, T) to (B, C, 2, T/2)
    This maintains Hailo compatibility while preserving temporal structure
    """
    B, C, T = data.shape
    if T % 2 == 1:
        data = data[:, :, :-1]  # Make even
        T = T - 1
    return data.reshape(B, C, 2, T//2)
```

### Stage 4: Normalization

```python
# Per-channel normalization
def normalize_imu_data(data):
    """
    Normalize each IMU channel independently
    Preserves relative magnitudes within channels
    """
    for ch in range(data.shape[1]):
        mean = data[:, ch].mean()
        std = data[:, ch].std() + 1e-6
        data[:, ch] = (data[:, ch] - mean) / std
    return data
```

## Quality Assurance

### QA Thresholds

| Metric | Threshold | Action if Exceeded |
|--------|-----------|-------------------|
| NaN percentage | 5% | Replace with zeros |
| Inf values | 0 | Clip to [-1e6, 1e6] |
| Min signal std | 0.01 | Warning (potential dead channel) |
| Max signal std | 50.0 | Warning (potential noise) |
| Dead channels | 3 | Fail validation |
| Class imbalance | 1:10 | Warning, consider resampling |

### QA Implementation

```python
class QuadrupedDatasetWithQA(Dataset):
    def __init__(self, X, y, quality_control=None):
        self.qc = quality_control or QualityControl()
        
        # Run validation
        validation_results = self.validate_data(X)
        
        if not validation_results['pass']:
            raise ValueError(f"Data failed QA: {validation_results['failures']}")
        
        # Apply corrections
        X = self.apply_corrections(X, validation_results)
```

### QA Report Example

```
PREPROCESSING QUALITY ASSURANCE
============================================================
Dataset Statistics:
  ✓ Shape: (10500, 9, 100)
  ✓ Data type: torch.float32
  ✓ Value range: [-33.79, 45.31]
  ✓ Mean: 4.7759, Std: 17.5019

Quality Checks:
  ✓ NaN values: 0
  ✓ Inf values: 0
  ✓ Dead channels (std<0.001): 0

Class Distribution:
  Class 0: 1050 samples (10.0%)
  Class 1: 1050 samples (10.0%)
  ...
  
✅ QA Complete - Data is ready for training
```

## Data Splits

### Temporal Splitting (Required for Time Series)

```python
def create_temporal_splits(X, y, ratios=[0.7, 0.15, 0.15]):
    """
    Create train/val/test splits with temporal ordering
    Prevents data leakage in time series data
    """
    n = len(X)
    train_end = int(ratios[0] * n)
    val_end = int((ratios[0] + ratios[1]) * n)
    
    splits = {
        'train': (X[:train_end], y[:train_end]),
        'val': (X[train_end:val_end], y[train_end:val_end]),
        'test': (X[val_end:], y[val_end:])
    }
    
    # Verify no overlap
    assert_no_overlap(splits)
    return splits
```

### Split Verification

```python
def verify_no_data_leakage(splits):
    """
    Verify that there's no data leakage between splits
    Critical for valid evaluation
    """
    train_hashes = set(hash(x.tobytes()) for x in splits['train'][0])
    val_hashes = set(hash(x.tobytes()) for x in splits['val'][0])
    test_hashes = set(hash(x.tobytes()) for x in splits['test'][0])
    
    assert len(train_hashes & val_hashes) == 0
    assert len(train_hashes & test_hashes) == 0
    assert len(val_hashes & test_hashes) == 0
    
    return True
```

## Training Procedures

### 1. Load Data with QA

```python
# Setup
from setup_quadruped_datasets import QuadrupedDatasetManager

manager = QuadrupedDatasetManager()
X, y, behaviors = manager.create_simulated_quadruped_data(n_samples=15000)
save_dir = manager.save_quadruped_data(X, y, behaviors)
```

### 2. Create Datasets with Validation

```python
from train_fsq_simple_qa import QuadrupedDatasetWithQA

# Load splits
X_train = np.load(save_dir / "X_train_quadruped.npy")
y_train = np.load(save_dir / "y_train_quadruped.npy")

# Create dataset with QA
train_dataset = QuadrupedDatasetWithQA(
    X_train, y_train, 
    quality_control=qc
)
```

### 3. Model Configuration

```python
from models.conv2d_fsq_model import Conv2dFSQ

model = Conv2dFSQ(
    input_channels=9,
    hidden_dim=128,
    num_classes=10,
    fsq_levels=[8, 6, 5],  # 240 unique codes
    project_dim=None
)
```

### 4. Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 64 | Balance between GPU memory and gradient stability |
| Learning Rate | 0.001 | Standard for Adam optimizer |
| Scheduler | ReduceLROnPlateau | Adaptive learning rate reduction |
| Patience | 30 epochs | Allow model to converge |
| Max Epochs | 300 | Sufficient for complex behaviors |
| Gradient Clipping | 1.0 | Prevent exploding gradients |

### 5. Evaluation Metrics

```python
def evaluate_model(model, test_loader):
    """
    Comprehensive evaluation with honest metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'per_class_accuracy': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'random_baseline': 1.0 / n_classes,
        'improvement': accuracy - (1.0 / n_classes)
    }
    
    return metrics
```

## Best Practices

### DO's
- ✅ Always use temporal splits for time series data
- ✅ Verify no data leakage between splits
- ✅ Run preprocessing QA on all data
- ✅ Report honest metrics (expect 70-80% accuracy)
- ✅ Document data sources and transformations
- ✅ Save preprocessing parameters for reproducibility

### DON'Ts
- ❌ Never use same data for train and test
- ❌ Don't skip QA validation
- ❌ Avoid synthetic data for final evaluation
- ❌ Don't claim >95% accuracy without scrutiny
- ❌ Never mix temporal data randomly

## Reproducibility

All preprocessing steps are deterministic and reproducible:

```python
# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Save preprocessing configuration
config = {
    'qa_thresholds': qc.thresholds.__dict__,
    'normalization': 'per_channel',
    'reshape_dims': (9, 2, 50),
    'temporal_split_ratios': [0.7, 0.15, 0.15],
    'data_source': 'quadruped_locomotion'
}

with open('preprocessing_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## Validation Checklist

Before training, verify:
- [ ] Data loaded successfully
- [ ] QA validation passed
- [ ] No NaN or Inf values
- [ ] Proper shape for Conv2d
- [ ] Temporal splits applied
- [ ] No data leakage verified
- [ ] Class distribution acceptable
- [ ] Preprocessing config saved

## References

- MIT Cheetah: https://github.com/mit-biomimetics/Cheetah-Software
- TartanVO: https://github.com/castacks/TartanVO
- Movement Library: https://movement.neuroinformatics.dev/
- Hailo Documentation: https://hailo.ai/developer-zone/