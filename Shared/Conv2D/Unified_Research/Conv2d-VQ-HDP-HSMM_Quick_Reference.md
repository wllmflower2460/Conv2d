# Conv2d-VQ-HDP-HSMM Quick Reference
*One-page reference for implementation*

## Core Formula
```
Input(B,9,2,T) → Encoder → VQ(discrete) → HDP(cluster) → HSMM(temporal) → Decoder
```

## Key Dimensions
- **B**: Batch size (32 typical)
- **C**: 9 IMU channels (acc_xyz, gyro_xyz, mag_xyz)  
- **H**: 2 devices (phone, collar)
- **T**: 100 timesteps (~1 second)
- **D**: 64 codebook dimension
- **K**: 512 codebook size
- **N**: 20 max clusters (HDP)
- **S**: 10 HSMM states

## Reusable Files from TCN-VAE Pipeline

```python
# Copy these directly - no changes needed
from models.tcn_vae_hailo import (
    HailoTemporalBlock,      # Conv2d temporal blocks
    DeviceAttention,         # Phone+IMU attention
    HailoTemporalConvNet    # TCN encoder/decoder
)

from preprocessing.enhanced_pipeline import (
    EnhancedCrossSpeciesDataset,  # Data loading
    HailoDataValidator,           # Validation tools
    get_dataset                   # Dataset factory
)

from config.training_config import TrainingConfig  # Baselines
```

## New Components to Build

### 1. VectorQuantizer (Priority 1)
```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        # Learnable codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        
    def forward(self, z):
        # Find nearest codebook entries
        distances = torch.cdist(z_flat, self.codebook.weight)
        indices = distances.argmin(dim=1)
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        return quantized, vq_loss, perplexity
```

### 2. HDPLayer (Priority 2)
```python
class HDPLayer(nn.Module):
    def __init__(self, input_dim=64, max_clusters=20):
        # Cluster centers
        self.centers = nn.Parameter(torch.randn(max_clusters, input_dim))
        # Stick-breaking weights
        self.v = nn.Parameter(torch.ones(max_clusters-1) * 0.5)
        
    def forward(self, z):
        # Compute cluster assignments
        weights = stick_breaking(self.v)
        assignments = gumbel_softmax(distances, tau=1.0)
        return clustered, cluster_info
```

### 3. HSMM (Priority 3)
```python
class HSMM(nn.Module):
    def __init__(self, num_states=10, max_duration=50):
        # State transitions
        self.trans_matrix = nn.Parameter(...)
        # Duration model
        self.duration_params = nn.Parameter(...)
        
    def forward(self, state_seq):
        # Forward-backward with durations
        return state_probs, durations
```

## Loss Components

```python
# Total Loss = Recon + βVQ + λClass + γHDP + δTemporal

recon_loss = F.mse_loss(x_recon, x_input)           # β=1.0
vq_loss = commit_loss + 0.25 * codebook_loss        # β=0.25  
class_loss = F.cross_entropy(logits, labels)        # λ=1.0
hdp_loss = -entropy(cluster_weights)                # γ=0.1
temporal_loss = -log_prob(state_transitions)        # δ=0.5
```

## Training Hyperparameters

```yaml
# Inherit from existing config, override these:
learning_rate: 1e-3          # Same as TCN-VAE
batch_size: 32               # Smaller for VQ stability
epochs: 300                  # More for discrete learning
patience: 50                 # Early stopping

# New VQ-specific:
commitment_cost: 0.25        # VQ commitment weight
decay: 0.99                  # EMA decay (optional)
tau_anneal: [1.0, 0.5]       # Gumbel temperature schedule
perplexity_threshold: 100   # Minimum codebook usage
```

## Validation Metrics

```python
# Monitor during training
metrics_to_log = {
    'loss/reconstruction': recon_loss.item(),
    'loss/vq': vq_loss.item(),
    'loss/total': total_loss.item(),
    'codebook/perplexity': perplexity.item(),  # >100 good
    'codebook/usage': used_codes / total_codes,  # >0.5 good
    'hdp/active_clusters': active_clusters,      # 5-15 typical
    'hsmm/mean_duration': mean_duration,         # 10-30 frames
    'accuracy/behavior': accuracy                # >85% target
}
```

## Hailo Compatibility Checks

```python
# Must avoid these operations:
UNSUPPORTED = ['Conv1d', 'GroupNorm', 'LayerNorm', 'Softmax']

# Use these alternatives:
Conv1d → Conv2d(kernel=(1,k))
GroupNorm → BatchNorm2d
LayerNorm → BatchNorm2d
Softmax → Sigmoid gates or Gumbel-Softmax

# Validate before export:
validator = HailoDataValidator()
is_valid = validator.validate_model_ops(model, config)
```

## Export for Deployment

```python
# Export only inference components
class InferenceModel(nn.Module):
    def __init__(self, full_model):
        self.encoder = full_model.encoder
        self.vq = full_model.vq.eval()  # No training ops
        self.classifier = full_model.classifier
        
    @torch.no_grad()
    def forward(self, x):
        encoded = self.encoder(x)
        indices = self.vq.get_indices(encoded)  # Just lookup
        quantized = self.vq.embed(indices)
        logits = self.classifier(quantized)
        return logits

# Export to ONNX
torch.onnx.export(
    InferenceModel(model),
    torch.randn(1, 9, 2, 100),
    "vq_hdp_hsmm.onnx",
    opset_version=11,
    do_constant_folding=True
)
```

## Testing Commands

```bash
# Quick component tests
python -m pytest tests/test_vq_layer.py -v
python -m pytest tests/test_hdp_clustering.py -v
python -m pytest tests/test_hsmm_dynamics.py -v

# Integration test
python test_integration.py --components all

# Hailo validation
python validate_hailo.py --model checkpoints/best.pth

# Training with tensorboard
python train_vq_hdp_hsmm.py --tb_logging
tensorboard --logdir runs/
```

## File Organization

```
Conv2d-VQ-HDP-HSMM/
├── models/
│   ├── vq_layer.py              # NEW: VQ implementation
│   ├── hdp_layer.py             # NEW: HDP clustering
│   ├── hsmm.py                  # NEW: HSMM dynamics
│   ├── conv2d_vq_hdp_hsmm.py   # NEW: Combined model
│   └── tcn_vae_hailo.py        # COPY: Base blocks
├── configs/
│   ├── vq_config.yaml          # NEW: VQ settings
│   └── enhanced_dataset.yaml   # COPY: Data config
└── train_vq_hdp_hsmm.py        # NEW: Training script
```

## Debugging Tips

| Problem | Check | Fix |
|---------|-------|-----|
| Low perplexity | Codebook usage | Reduce commitment cost |
| NaN loss | Gradient explosion | Reduce LR, add clipping |
| Poor reconstruction | VQ bottleneck | Increase codebook size |
| Single cluster | HDP collapse | Adjust concentration |
| Static states | HSMM stuck | Check transition init |
| Hailo error | Unsupported ops | Replace with Conv2d |

## Performance Targets

- **Training**: Converge in <24 hours on V100
- **Inference**: <15ms on Hailo-8
- **Accuracy**: >85% behavior classification
- **Codebook**: >50% utilization
- **Clusters**: 5-15 active
- **Memory**: <150MB peak

---

*Keep this handy during implementation!*
