# Configuration as a First-Class Citizen

The Conv2d-FSQ-HSMM project uses a robust configuration system combining **Hydra** for composition and **Pydantic** for validation, ensuring reproducibility and type safety.

## 🎯 Key Features

### 1. **Single Source of Truth**
- Base configuration: `conf/base.yaml`
- Environment overlays: `conf/env/{gpu,pi,hailo,local}.yaml`
- Experiment configs: `conf/exp/{quadruped,wisdm,pamap2}.yaml`
- Model configs: `conf/model/*.yaml`
- Training configs: `conf/training/*.yaml`

### 2. **Validation with Pydantic**
- Full type validation at startup (fail fast)
- Range constraints (e.g., learning rate > 0)
- Cross-field validation (e.g., max_duration > min_duration)
- Automatic type coercion and defaults

### 3. **Configuration Hashing**
- Deterministic SHA256 hash of config (excluding volatile keys)
- Every artifact folder includes the hash
- Ensures reproducibility tracking

## 📂 Configuration Structure

```
conf/
├── base.yaml                 # Base configuration (source of truth)
├── env/                      # Environment overlays
│   ├── local.yaml           # Local development (CPU)
│   ├── gpu.yaml             # GPU server (CUDA)
│   ├── pi.yaml              # Raspberry Pi 5
│   └── hailo.yaml           # Hailo-8 accelerator
├── exp/                      # Experiment configs
│   ├── quadruped.yaml       # Dog-human synchrony
│   ├── wisdm.yaml           # WISDM HAR dataset
│   └── pamap2.yaml          # PAMAP2 activity dataset
├── model/                    # Model architectures
│   └── conv2d_fsq_hsmm.yaml # Main model config
└── training/                 # Training strategies
    ├── standard.yaml        # Standard training
    └── fast.yaml           # Quick experiments
```

## 💻 Usage

### Basic Usage

```python
from conv2d.config import load_config

# Load base configuration
config = load_config()

# Access typed configuration
print(config.data.batch_size)  # 32
print(config.model.fsq.codebook_size)  # 240
print(config.training.epochs)  # 100
```

### With Overrides

```python
# Hydra-style overrides
config = load_config(
    overrides=[
        "env=gpu",              # Use GPU environment
        "exp=quadruped",        # Quadruped experiment
        "training.epochs=200",  # Override epochs
        "data.batch_size=64",   # Override batch size
    ]
)
```

### Command-Line Interface

```bash
# Run training with GPU environment and quadruped experiment
python -m conv2d.cli.hydra_train \
    env=gpu \
    exp=quadruped \
    training.epochs=200 \
    data.batch_size=64

# Override multiple settings
python -m conv2d.cli.hydra_train \
    env=hailo \
    exp=wisdm \
    model.fsq.levels=[4,4,4] \
    training=fast \
    +experiment.tags=[test,debug]
```

### Configuration Loader

```python
from conv2d.config import ConfigLoader

# Initialize loader
loader = ConfigLoader(config_dir="conf")

# Load with validation
config = loader.load(
    overrides=["env=gpu", "exp=quadruped"],
    validate=True,  # Pydantic validation
    compute_hash=True  # Add config hash
)

# Save with hash in filename
saved_path = loader.save("outputs", include_hash=True)
# Creates: outputs/config_a3f8b2c1.yaml
```

## 🔒 Validation Examples

### Automatic Validation

```python
# This will fail validation (negative batch size)
try:
    config = load_config(overrides=["data.batch_size=-1"])
except ValidationError as e:
    print(f"Validation failed: {e}")

# This will fail (max_duration <= min_duration)  
try:
    config = load_config(
        overrides=[
            "model.hsmm.min_duration=10",
            "model.hsmm.max_duration=5"
        ]
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Custom Validation

```python
from conv2d.config.models import DataConfig

# Pydantic model validation
data_config = DataConfig(
    dataset_name="custom",
    batch_size=32,
    window_size=100,
    step_size=150  # Will fail: step_size > window_size
)
```

## 🔑 Configuration Hashing

### Hash Computation

```python
from conv2d.config import compute_config_hash

# Compute deterministic hash
config_hash = compute_config_hash(config.model_dump())
print(f"Config hash: {config_hash}")  # e.g., "a3f8b2c1"

# Hash excludes volatile keys:
# - paths.output_dir
# - experiment.run_id
# - hardware.device_id
# - hydra settings
```

### Reproducibility Tracking

```python
from conv2d.config.utils import validate_config_reproducibility

# Check if config is reproducible
is_reproducible, issues = validate_config_reproducibility(config)

if not is_reproducible:
    print("Issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

## 🎯 Environment-Specific Configs

### GPU Environment
```yaml
# conf/env/gpu.yaml
device: cuda
mixed_precision: true
cuda:
  cudnn_benchmark: true
  allow_tf32: true
```

### Hailo-8 Environment
```yaml
# conf/env/hailo.yaml
device: hailo
hailo:
  arch: hailo8
  batch_size: 1  # Optimized for latency
  quantization:
    mode: int8
performance:
  target_latency_ms: 10
```

### Raspberry Pi Environment
```yaml
# conf/env/pi.yaml
device: cpu
raspberry_pi:
  model: Pi5
  num_threads: 4
resource_limits:
  max_memory_gb: 4
```

## 🔬 Experiment Configs

### Quadruped Behavioral Sync
```yaml
# conf/exp/quadruped.yaml
data:
  dataset_name: quadruped
  num_keypoints: 24
  num_subjects: 2
model:
  fsq_levels: [8, 6, 5]  # 240 behavioral codes
  hsmm:
    num_states: 32
```

### WISDM Human Activity
```yaml
# conf/exp/wisdm.yaml
data:
  dataset_name: wisdm
  num_activities: 6
  sampling_rate: 20.0
model:
  fsq_levels: [4, 4, 4]  # 64 codes
```

## 📊 Configuration Diffing

```python
from conv2d.config import get_config_diff

# Compare configurations
config1 = load_config(overrides=["training.epochs=100"])
config2 = load_config(overrides=["training.epochs=200"])

diff = get_config_diff(
    config1.model_dump(),
    config2.model_dump()
)

print(diff)
# Output: {'changed': {'training.epochs': {'old': 100, 'new': 200}}}
```

## 🚀 Hydra Features

### Multirun (Parameter Sweeps)

```bash
# Sweep over learning rates
python -m conv2d.cli.hydra_train \
    --multirun \
    training.optimizer.learning_rate=1e-4,5e-4,1e-3 \
    env=gpu

# Grid search over multiple parameters
python -m conv2d.cli.hydra_train \
    --multirun \
    model.fsq.levels='[4,4,4]','[6,5,4]','[8,6,5]' \
    training.optimizer.learning_rate=1e-4,1e-3 \
    data.batch_size=16,32,64
```

### Output Directory Structure

```
outputs/
├── quadruped_behavioral_sync/
│   ├── 2024-01-15_10-30-45/  # Run with timestamp
│   │   ├── config_a3f8b2c1.yaml  # Config with hash
│   │   ├── .hydra/
│   │   │   ├── config.yaml
│   │   │   └── overrides.yaml
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── tensorboard/
│   └── 2024-01-15_11-45-22/
│       └── ...
└── sweeps/
    └── learning_rate_sweep/
        ├── 0/  # lr=1e-4
        ├── 1/  # lr=5e-4
        └── 2/  # lr=1e-3
```

## 🔧 Best Practices

1. **Always use overlays** instead of modifying base.yaml
2. **Create experiment configs** for different datasets/tasks
3. **Include config hash** in artifact names
4. **Validate early** with Pydantic at startup
5. **Track volatile keys** separately (not in hash)
6. **Use typed access** via Pydantic models

## 📝 Creating New Configs

### New Experiment

```yaml
# conf/exp/my_dataset.yaml
# @package _global_

defaults:
  - override /model: conv2d_fsq_hsmm
  - override /training: standard

experiment:
  name: my_experiment
  tags: [custom, test]

data:
  dataset_name: my_dataset
  batch_size: 16
  
model:
  fsq_levels: [5, 5, 5]  # 125 codes
```

### New Environment

```yaml
# conf/env/tpu.yaml
# @package env

name: tpu
device: tpu
tpu:
  core_count: 8
  mixed_precision: bfloat16
```

## 🔍 Debugging

```python
# Print resolved configuration
from omegaconf import OmegaConf

loader = ConfigLoader()
config = loader.load(overrides=["env=gpu"])
loader.print_config(resolve=True)

# Access raw Hydra config
raw_config = loader.raw_config
print(OmegaConf.to_yaml(raw_config))

# Check what was overridden
print(raw_config.hydra.overrides.task)
```

The configuration system ensures that every experiment is reproducible, validated, and tracked with a unique hash for full traceability.