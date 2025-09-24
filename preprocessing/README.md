# Preprocessing Pipeline Documentation

## Overview

The preprocessing pipeline for Conv2d-FSQ includes comprehensive quality assurance, data validation, and transformation procedures to ensure reliable behavioral analysis. This module handles multi-dataset preprocessing with QA validation for both human activity recognition and quadruped behavioral analysis.

## Pipeline Components

### Core Modules

1. **`enhanced_pipeline.py`** - Primary preprocessing with dual approach support (cross-species and HAR)
2. **`movement_diagnostics.py`** - Advanced quality control and validation system
3. **`movement_integration.py`** - Integration with Movement neuroinformatics library
4. **`kinematic_features.py`** - Kinematic feature extraction for behavioral analysis
5. **`data_augmentation.py`** - Augmentation strategies for behavioral data
6. **`unified_pipeline.py`** - Unified interface handling multiple HAR datasets
7. **`stanford_dogs_pipeline.py`** - Specialized pipeline for dog behavioral data

## Quality Assurance System

### QualityControl Class
Comprehensive quality control with multi-layer validation:

```python
from preprocessing.movement_diagnostics import QualityControl, QualityThresholds

# Initialize with custom thresholds
qa_thresholds = QualityThresholds(
    max_nan_percentage=5.0,
    min_signal_std=0.01,
    max_signal_std=50.0,
    min_codebook_usage=0.2,
    min_perplexity=4.0
)

qc = QualityControl(thresholds=qa_thresholds, strict_mode=False)
```

### MultiDatasetHAR Class
Central class for loading, preprocessing, and unifying multiple HAR datasets.

```python
class MultiDatasetHAR:
    def __init__(self, window_size=100, overlap=0.5)
```

**Parameters:**
- `window_size` (int): Length of sliding windows in timesteps (default: 100)
- `overlap` (float): Window overlap ratio 0.0-1.0 (default: 0.5)

## Supported Datasets

### 1. PAMAP2 Dataset
**Source**: Chest-mounted IMU sensor data  
**Activities**: 19 different activities (lying, sitting, walking, running, etc.)  
**Format**: Space-separated `.dat` files  
**Sensors**: 9-axis IMU (accelerometer + gyroscope + magnetometer)

```python
def load_pamap2(self, data_path):
    """Load PAMAP2 dataset - 9-axis IMU data from chest sensor"""
```

**Data Processing with QA:**
- Extracts chest sensor IMU data (columns 4-12)
- **QA Step**: Validates signal quality (SNR, variance)
- Filters out NaN values and transient activities (activity_id = 0)
- **QA Step**: Checks data consistency and gaps
- Maps activity IDs to descriptive labels
- **QA Step**: Verifies class distribution balance
- Handles multiple subject files in Protocol directory
- **QA Step**: Generates comprehensive quality report

**Activity Mapping:**
```python
{
    1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking',
    5: 'running', 6: 'cycling', 7: 'nordic_walking',
    9: 'watching_tv', 10: 'computer_work', 11: 'car_driving',
    12: 'ascending_stairs', 13: 'descending_stairs',
    16: 'vacuum_cleaning', 17: 'ironing', 18: 'folding_laundry',
    19: 'house_cleaning', 20: 'playing_soccer', 24: 'rope_jumping'
}
```

### 2. UCI-HAR Dataset
**Source**: Smartphone sensors (accelerometer + gyroscope)  
**Activities**: 6 basic activities  
**Format**: Space-separated `.txt` files with preprocessed features  
**Processing**: Converts 561 features to 9-axis pseudo-IMU format

```python
def load_uci_har(self, data_path):
    """Load UCI-HAR dataset - smartphone accelerometer + gyroscope"""
```

**Data Processing:**
- Loads train and test splits from separate files
- Uses first 6 features as pseudo-IMU signals
- Pads to 9 dimensions and replicates across 100 timesteps
- Combines train/test for unified training

**Activity Mapping:**
```python
{
    1: 'walking', 2: 'walking_upstairs', 3: 'walking_downstairs',
    4: 'sitting', 5: 'standing', 6: 'laying'
}
```

### 3. TartanIMU Dataset (Synthetic)
**Source**: Synthetic IMU data for proof of concept  
**Activities**: 4 motion patterns  
**Format**: Generated synthetic signals

```python
def load_tartan_imu(self, data_path):
    """Load TartanIMU sample data - simplified for proof of concept"""
```

**Synthetic Patterns:**
- **Stationary**: Low-noise signals around zero
- **Walking**: Periodic gait patterns with gravity
- **Running**: Higher amplitude periodic motion
- **Turning**: Angular velocity patterns with magnetic field variation

**Activity Mapping:**
```python
{
    0: 'stationary', 1: 'walking', 2: 'running', 3: 'turning'
}
```

## Data Processing Pipeline

### 1. Window Creation
```python
def create_windows(self, data, labels, dataset_name):
    """Sliding window approach with overlap"""
```

**Process:**
- Creates overlapping windows from continuous data
- Window size: 100 timesteps (~1 second at 100Hz)
- Overlap: 50% by default (50 timestep stride)
- Label assignment: Center label of each window
- Domain labels: Dataset identifier for each window

### 2. Normalization
**Method**: Z-score normalization per sensor channel
```python
# Applied per channel across all datasets
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_reshaped)
```

**Benefits:**
- Accounts for different sensor ranges across datasets
- Maintains relative patterns within each sensor type
- Enables unified training across heterogeneous data sources

### 3. Label Encoding
```python
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_combined)

domain_encoder = LabelEncoder()
domains_encoded = domain_encoder.fit_transform(domains_combined)
```

## Main Processing Method

### `preprocess_all()`
Central method that loads all datasets and creates unified training data.

```python
def preprocess_all(self):
    """Load all datasets and create unified training format"""
    # Returns: X_train, y_train, domains_train, X_val, y_val, domains_val
```

**Process Flow:**
1. **Load PAMAP2**: Extract chest IMU data, create windows
2. **Load UCI-HAR**: Convert features to pseudo-IMU, create windows  
3. **Load TartanIMU**: Generate synthetic data, create windows
4. **Combine Datasets**: Stack all windows and labels
5. **Encode Labels**: Create unified label space
6. **Normalize Features**: Z-score normalization per channel
7. **Train/Val Split**: Stratified split maintaining class balance

**Output Shapes:**
- `X_train/X_val`: (N, 100, 9) - windowed IMU data
- `y_train/y_val`: (N,) - encoded activity labels
- `domains_train/domains_val`: (N,) - encoded dataset labels

## Data Characteristics

### Combined Dataset Statistics
```python
Total windows: ~50,000-100,000 (depending on datasets available)
Training split: 80%
Validation split: 20%
Activity classes: 13 unified classes
Domain classes: 3 (pamap2, uci_har, tartan_imu)
```

### Window Statistics
- **Window size**: 100 timesteps
- **Overlap**: 50% (50 timestep stride)
- **Sampling rate equivalent**: ~100Hz
- **Time coverage**: ~1 second per window

### Feature Distribution
- **Accelerometer**: 3 channels (ax, ay, az) in m/s²
- **Gyroscope**: 3 channels (gx, gy, gz) in rad/s
- **Magnetometer**: 3 channels (mx, my, mz) in μT
- **Normalization**: Zero mean, unit variance per channel

## Usage Examples

### Basic Usage
```python
from preprocessing.unified_pipeline import MultiDatasetHAR

# Initialize processor
processor = MultiDatasetHAR(window_size=100, overlap=0.5)

# Load and preprocess all datasets
X_train, y_train, domains_train, X_val, y_val, domains_val = processor.preprocess_all()

print(f"Training samples: {X_train.shape[0]}")
print(f"Window shape: {X_train.shape[1:]}")  # (100, 9)
print(f"Number of classes: {len(np.unique(y_train))}")
```

### Custom Window Configuration
```python
# Smaller windows for faster processing
processor = MultiDatasetHAR(window_size=50, overlap=0.25)

# Larger windows for more context
processor = MultiDatasetHAR(window_size=200, overlap=0.75)
```

### Accessing Preprocessing Artifacts
```python
# After preprocessing, access fitted transformers
label_mapping = processor.label_encoder.classes_
domain_mapping = processor.domain_encoder.classes_
normalization_params = processor.scaler

# Save for inference pipeline
import pickle
with open('processor.pkl', 'wb') as f:
    pickle.dump(processor, f)
```

## Integration with Training

### DataLoader Creation
```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Create datasets
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(y_train),
    torch.LongTensor(domains_train)
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

### Model Input Format
The preprocessed data is compatible with the TCNVAE model:
```python
# Expected input shape: (batch_size, sequence_length, input_dim)
# Example: (64, 100, 9) for batch of 64 windows
```

## Dataset-Specific Notes

### PAMAP2 Considerations
- **Missing Data**: NaN values are filtered out
- **Activity Coverage**: Not all activities may be present in all subjects
- **Sensor Position**: Chest-mounted sensor provides stable orientation
- **File Structure**: Subject files in Protocol/ directory

### UCI-HAR Considerations  
- **Feature Engineering**: Original dataset uses 561 engineered features
- **Window Adaptation**: Converts pre-windowed data to our window format
- **Limited Sensors**: Only accelerometer and gyroscope available
- **Padding**: Magnetometer channels padded with zeros

### TartanIMU Considerations
- **Synthetic Data**: Not real sensor data, used for proof of concept
- **Pattern Generation**: Mathematically generated motion patterns
- **Limited Realism**: May not capture real-world sensor noise and variations
- **Extensibility**: Easy to modify for different synthetic patterns

## Error Handling

The preprocessing pipeline includes robust error handling:

```python
try:
    # Data loading with validation
    if valid_mask.sum() > 0:
        data_list.append(imu_data[valid_mask])
except Exception as e:
    print(f"Error loading {file_path}: {e}")
    continue
```

**Common Issues:**
- Missing dataset files: Graceful degradation
- Malformed data: Skip problematic samples
- Insufficient data: Raise informative errors
- Memory constraints: Chunked processing for large datasets

## Performance Optimization

### Memory Efficiency
- Processes datasets sequentially to reduce peak memory usage
- Uses numpy arrays for efficient numerical operations
- Applies transforms in-place where possible

### Processing Speed
- Vectorized operations with pandas and numpy
- Efficient window creation with stride operations
- Minimal data copying between processing steps

### Caching Potential
The preprocessing results can be cached:
```python
# Save preprocessed data
np.savez_compressed('preprocessed_data.npz',
                   X_train=X_train, y_train=y_train, domains_train=domains_train,
                   X_val=X_val, y_val=y_val, domains_val=domains_val)
```