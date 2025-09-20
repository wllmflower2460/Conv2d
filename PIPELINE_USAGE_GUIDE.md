# Pipeline Usage Guide
## How to Use the Dual Approach System

---

## Overview

This pipeline provides two complementary approaches for behavioral analysis:
1. **Cross-Species Conv2d** (Primary): Innovative approach for Hailo-8 deployment with cross-species transfer
2. **Traditional HAR** (Fallback): Proven multi-dataset approach for standard human activity recognition

Both approaches are accessible through a unified factory pattern, making it easy to switch between them for testing, comparison, or fallback scenarios.

---

## Quick Start

### Import the Factory Function
```python
from preprocessing.enhanced_pipeline import get_dataset
```

### Choose Your Approach
```python
# Option 1: Cross-Species Conv2d (Recommended for Hailo deployment)
dataset = get_dataset(approach='cross_species', ...)

# Option 2: Traditional HAR (Fallback for standard datasets)
dataset = get_dataset(approach='traditional_har', ...)
```

---

## Approach 1: Cross-Species Conv2d Pipeline

### When to Use
- ✅ Deploying to Hailo-8 hardware
- ✅ Analyzing human-dog behavioral synchrony
- ✅ Need phone+collar IMU fusion
- ✅ Want to leverage cross-species transfer learning
- ✅ Require static shapes for edge deployment

### Basic Usage
```python
from preprocessing.enhanced_pipeline import get_dataset

# Create dataset with Conv2d approach
train_dataset = get_dataset(
    approach='cross_species',
    config_path='configs/enhanced_dataset_schema.yaml',
    mode='train',
    enforce_hailo_constraints=True
)

val_dataset = get_dataset(
    approach='cross_species',
    config_path='configs/enhanced_dataset_schema.yaml',
    mode='val',
    enforce_hailo_constraints=True
)

# Get dataloaders
train_loader = train_dataset.get_dataloader(batch_size=32, shuffle=True)
val_loader = val_dataset.get_dataloader(batch_size=32, shuffle=False)

# Iterate through batches
for batch in train_loader:
    inputs = batch['input']           # Shape: (B, 9, 2, 100)
    human_labels = batch['human_label']
    dog_labels = batch['dog_label']
    has_dog = batch['has_dog_label']  # Boolean mask for valid dog labels
    
    # Your training code here
    break
```

### Advanced Configuration
```python
# Disable Hailo constraints for experimentation
dataset = get_dataset(
    approach='cross_species',
    config_path='configs/enhanced_dataset_schema.yaml',
    mode='train',
    enforce_hailo_constraints=False  # Allow dynamic shapes
)

# Custom dataloader settings
dataloader = dataset.get_dataloader(
    batch_size=64,
    shuffle=True,
    num_workers=8,  # Parallel data loading
    pin_memory=True  # For GPU training
)
```

### Data Shape Explanation
```python
# Input tensor shape: (B, C, H, T)
# B = Batch size
# C = 9 IMU channels (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z)
# H = 2 devices (phone, collar_imu)
# T = 100 timesteps (~1 second at 100Hz)

# Example:
# batch['input'].shape = (32, 9, 2, 100)
# - 32 samples in batch
# - 9 IMU channels
# - 2 devices (phone at index 0, collar at index 1)
# - 100 timesteps
```

---

## Approach 2: Traditional HAR Pipeline

### When to Use
- ✅ Working with standard HAR datasets (PAMAP2, UCI-HAR, WISDM, HAPT)
- ✅ Need proven baseline for comparison
- ✅ Don't require Hailo-specific optimizations
- ✅ Want to leverage existing dataset loaders

### Basic Usage
```python
from preprocessing.enhanced_pipeline import get_dataset

# Create HAR dataset processor
har_processor = get_dataset(
    approach='traditional_har',
    window_size=100,        # 1 second at 100Hz
    overlap=0.5,           # 50% sliding window overlap
    base_dataset_path='./datasets'
)

# Process all datasets
X_train, y_train, domains_train, X_val, y_val, domains_val = \
    har_processor.preprocess_all_enhanced()

print(f"Training samples: {X_train.shape}")
print(f"Validation samples: {X_val.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")
```

### Working with Specific Datasets
```python
# Load individual datasets
processor = get_dataset(approach='traditional_har')

# WISDM dataset
wisdm_data, wisdm_labels = processor.load_wisdm('./datasets/wisdm')

# HAPT dataset (includes transitions)
hapt_data, hapt_labels = processor.load_hapt('./datasets/hapt')

# Create windows
windows, labels, domains = processor.create_windows(
    wisdm_data, 
    wisdm_labels, 
    'wisdm'
)
```

---

## Switching Between Approaches

### A/B Testing Example
```python
def compare_approaches(config_path, har_data_path):
    """Compare both approaches on the same task"""
    
    # Approach 1: Conv2d
    conv2d_dataset = get_dataset(
        approach='cross_species',
        config_path=config_path,
        mode='train'
    )
    
    # Approach 2: Traditional HAR
    har_processor = get_dataset(
        approach='traditional_har',
        base_dataset_path=har_data_path
    )
    
    # Train models with each approach
    conv2d_model = train_with_conv2d(conv2d_dataset)
    har_model = train_with_har(har_processor)
    
    # Compare results
    return {
        'conv2d_accuracy': evaluate(conv2d_model),
        'har_accuracy': evaluate(har_model)
    }
```

### Fallback Pattern
```python
def get_data_with_fallback(config_path, har_path):
    """Try Conv2d first, fallback to HAR if needed"""
    try:
        # Try primary approach
        dataset = get_dataset(
            approach='cross_species',
            config_path=config_path,
            mode='train',
            enforce_hailo_constraints=True
        )
        print("✅ Using Conv2d approach for Hailo")
        return dataset
        
    except Exception as e:
        print(f"⚠️ Conv2d failed: {e}")
        print("📊 Falling back to traditional HAR")
        
        # Fallback to traditional approach
        processor = get_dataset(
            approach='traditional_har',
            base_dataset_path=har_path
        )
        return processor
```

---

## Validation and Testing

### Validate Hailo Compatibility
```python
from preprocessing.enhanced_pipeline import HailoDataValidator

# Only relevant for Conv2d approach
dataset = get_dataset(
    approach='cross_species',
    config_path='configs/enhanced_dataset_schema.yaml',
    mode='test'
)

validator = HailoDataValidator()

# Check data shapes
sample = dataset[0]
is_valid = validator.validate_tensor_shape(
    sample['input'].unsqueeze(0),
    dataset.config
)
print(f"Hailo shape compatibility: {'✅' if is_valid else '❌'}")

# Check model operations
from models.tcn_vae_hailo import HailoTCNVAE
model = HailoTCNVAE()
is_compatible = validator.validate_model_ops(model, dataset.config)
print(f"Model Hailo compatibility: {'✅' if is_compatible else '❌'}")
```

### Compare Data Distributions
```python
import matplotlib.pyplot as plt

# Load both approaches
conv2d_data = get_dataset(approach='cross_species', ...)
har_data = get_dataset(approach='traditional_har', ...)

# Visualize distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Conv2d distribution
sample_conv2d = conv2d_data[0]['input'].numpy()
axes[0].imshow(sample_conv2d.mean(axis=0), aspect='auto')
axes[0].set_title('Conv2d: Phone vs Collar IMU')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Device (0=Phone, 1=Collar)')

# HAR distribution  
sample_har = har_data.preprocess_all_enhanced()[0][0]
axes[1].plot(sample_har.mean(axis=0))
axes[1].set_title('Traditional HAR: Averaged Channels')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()
```

---

## Configuration Files

### Conv2d Approach Configuration (YAML)
```yaml
# configs/enhanced_dataset_schema.yaml
hailo_deployment:
  architecture_constraints:
    unsupported_ops: ['Conv1d', 'GroupNorm', 'LayerNorm']
    groups_allowed: 1
  io_specification:
    input_shape: [1, 9, 2, 100]  # Static shape for Hailo
    static_shape_required: true

data_pipeline:
  preprocessing:
    - step: normalization
      method: standard
    - step: augmentation
      enabled: true
      techniques:
        time_warping:
          enabled: true
          sigma: 0.2

cross_species_mapping:
  mapping_confidence_threshold: 0.7
  behavioral_correspondences:
    - source_activity: sitting
      target_behavior: sit
      confidence: 0.95
    - source_activity: lying
      target_behavior: down
      confidence: 0.90
```

### Traditional HAR Configuration (Python)
```python
# In code configuration
har_config = {
    'window_size': 100,
    'overlap': 0.5,
    'datasets': ['pamap2', 'uci_har', 'wisdm', 'hapt'],
    'canonical_labels': {
        'sit': 0, 'down': 1, 'stand': 2,
        'walking': 3, 'running': 4
    }
}

processor = get_dataset(
    approach='traditional_har',
    **har_config
)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Conv2d approach fails with shape mismatch
```python
# Solution: Check your config file
assert config['hailo_deployment']['io_specification']['input_shape'] == [1, 9, 2, 100]
```

#### Issue: Traditional HAR missing datasets
```python
# Solution: Create synthetic data as fallback
processor = get_dataset(approach='traditional_har')
# Will automatically generate synthetic data if real data not found
```

#### Issue: Memory issues with large batches
```python
# Solution: Reduce batch size or use gradient accumulation
dataloader = dataset.get_dataloader(
    batch_size=16,  # Smaller batch
    num_workers=2   # Fewer parallel workers
)
```

---

## Best Practices

1. **Start with Conv2d** for new projects targeting edge deployment
2. **Use Traditional HAR** for baseline comparisons and validation
3. **Test both approaches** when accuracy is critical
4. **Document which approach** was used for each model
5. **Version control** your configuration files
6. **Validate early** using the HailoDataValidator for Conv2d approach

---

## Performance Comparison

| Aspect | Conv2d Approach | Traditional HAR |
|--------|----------------|-----------------|
| **Hailo-8 Compatible** | ✅ Yes | ❌ No |
| **Cross-Species** | ✅ Yes | ❌ No |
| **Static Shapes** | ✅ Enforced | ❌ Dynamic |
| **Device Fusion** | ✅ Native | ⚠️ Manual |
| **Dataset Support** | ⚠️ Limited | ✅ Extensive |
| **Edge Deployment** | ✅ Optimized | ❌ Not optimized |
| **Proven Results** | ⚠️ New | ✅ Established |

---

## Next Steps

1. **For Production**: Use Conv2d approach with Hailo validation
2. **For Research**: Compare both approaches on your specific task
3. **For Development**: Start with Traditional HAR, migrate to Conv2d
4. **For Testing**: Implement both and A/B test

---

*"The best architecture is the one that ships. Use Conv2d for innovation, Traditional HAR for validation, and the factory pattern to switch between them seamlessly."*