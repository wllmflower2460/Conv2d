# Conv2d-VQ-HDP-HSMM Implementation Roadmap
*Created: 2025-09-19*
*Practical step-by-step guide for implementation*

## Quick Start Checklist

### Repository Setup
```bash
# Create new repository from existing pipeline
cd /Users/willflower/Developer/data-dogs/
cp -r TCN-VAE_Training_Pipeline- Conv2d-VQ-HDP-HSMM
cd Conv2d-VQ-HDP-HSMM
git init
git add .
git commit -m "Initial fork from TCN-VAE pipeline"
```

### Essential Files to Keep
```
✅ KEEP AS-IS:
- models/tcn_vae_hailo.py (HailoTemporalBlock, DeviceAttention)
- preprocessing/enhanced_pipeline.py (Dataset management)
- configs/enhanced_dataset_schema.yaml (YAML config)
- config/training_config.py (Training parameters)
- models/device_attention.py (If separate)

⚠️ MODIFY:
- models/tcn_vae.py → models/conv2d_vq_hdp_hsmm.py
- training/train_tcn_vae.py → training/train_vq_hdp_hsmm.py

❌ REMOVE/REPLACE:
- VAE-specific loss calculations
- Continuous latent space operations
```

## Component Implementation Order

### Week 1: Vector Quantization (Start Here!)

#### Day 1-2: Basic VQ Layer
```python
# models/vq_layer.py
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize uniformly
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
```

#### Day 3-4: Integration with Encoder
```python
# Modify existing encoder output
encoded = self.tcn_encoder(x)  # Your existing encoder
z = self.to_codebook_dim(encoded)  # New projection layer
quantized, vq_loss = self.vq_layer(z)  # Add VQ
```

#### Day 5: Test & Validate
- Check gradient flow
- Monitor codebook usage
- Validate reconstruction quality

### Week 2: HDP Clustering

#### Day 1-2: Implement Stick-Breaking
```python
def stick_breaking_weights(v, K):
    """Generate weights from stick-breaking construction"""
    batch_size = v.size(0)
    weights = torch.zeros(batch_size, K)
    remaining = torch.ones(batch_size)
    
    for k in range(K-1):
        weights[:, k] = v[:, k] * remaining
        remaining *= (1 - v[:, k])
    weights[:, -1] = remaining
    return weights
```

#### Day 3-4: HDP Layer Integration
```python
class HDPLayer(nn.Module):
    def __init__(self, input_dim, max_clusters=20):
        super().__init__()
        self.cluster_centers = nn.Parameter(torch.randn(max_clusters, input_dim))
        self.concentration = nn.Parameter(torch.tensor(1.0))
```

#### Day 5: Clustering Validation
- Visualize cluster assignments
- Monitor active clusters
- Check cluster coherence

### Week 3: HSMM Implementation

#### Day 1-2: Basic HSMM Structure
```python
class HSMM(nn.Module):
    def __init__(self, num_states=10):
        super().__init__()
        # Transition probabilities
        self.trans_matrix = nn.Parameter(torch.randn(num_states, num_states))
        # Duration parameters
        self.duration_mean = nn.Parameter(torch.ones(num_states) * 10)
        self.duration_std = nn.Parameter(torch.ones(num_states))
```

#### Day 3-4: Forward-Backward Algorithm
- Implement forward pass
- Add backward pass
- Include duration modeling

#### Day 5: Temporal Validation
- Check state transitions
- Validate duration distributions
- Test on sequences

### Week 4: Full Integration

#### Day 1-2: Combine All Components
```python
class Conv2dVQHDPHSMM(nn.Module):
    def __init__(self):
        # Combine all modules
        self.encoder = ...  # Existing
        self.vq = VectorQuantizer(...)
        self.hdp = HDPLayer(...)
        self.hsmm = HSMM(...)
        self.decoder = ...  # Existing
```

#### Day 3-4: Training Pipeline
- Combined loss function
- Training loop
- Validation metrics

#### Day 5: Testing & Optimization
- End-to-end testing
- Performance optimization
- Hailo export validation

## Code Templates

### Template 1: Minimal VQ Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalVQ(nn.Module):
    def __init__(self, num_codes=256, code_dim=32):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        
    def forward(self, z):
        # z shape: (B, D, H, W)
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        
        # Find nearest codes
        distances = torch.cdist(z_flat, self.codebook.weight)
        indices = distances.argmin(dim=1)
        
        # Quantize
        quantized = self.codebook(indices).reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        # Straight-through estimator
        return z + (quantized - z).detach()
```

### Template 2: Training Loop Modification
```python
def train_step(model, batch, optimizer):
    outputs = model(batch['input'])
    
    # Reconstruction loss (existing)
    recon_loss = F.mse_loss(outputs['reconstructed'], batch['input'])
    
    # VQ loss (new)
    vq_loss = outputs['vq_loss']
    
    # Classification loss (modified)
    class_loss = F.cross_entropy(outputs['behavior_logits'], batch['labels'])
    
    # Combined loss
    total_loss = recon_loss + 0.25 * vq_loss + class_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        'loss': total_loss.item(),
        'perplexity': outputs['perplexity'].item()
    }
```

### Template 3: Hailo Export Modification
```python
def export_for_hailo(model, dummy_input):
    """Export VQ-HDP-HSMM model for Hailo"""
    
    # Create inference-only version
    class InferenceModel(nn.Module):
        def __init__(self, full_model):
            super().__init__()
            self.encoder = full_model.encoder
            self.vq = full_model.vq
            self.classifier = full_model.behavior_classifier
            
        def forward(self, x):
            encoded = self.encoder(x)
            quantized = self.vq(encoded)
            # Skip HDP/HSMM for inference
            logits = self.classifier(quantized.mean(dim=[2,3]))
            return logits
    
    inference_model = InferenceModel(model)
    torch.onnx.export(inference_model, dummy_input, "model.onnx")
```

## Debugging Checklist

### VQ Issues
- [ ] Codebook collapse? → Check perplexity
- [ ] No gradient flow? → Verify straight-through estimator
- [ ] Poor reconstruction? → Adjust commitment cost

### HDP Issues
- [ ] All data in one cluster? → Check initialization
- [ ] Too many clusters? → Adjust concentration parameter
- [ ] Unstable training? → Use temperature annealing

### HSMM Issues
- [ ] States not changing? → Check transition matrix
- [ ] Unrealistic durations? → Adjust duration priors
- [ ] Computational bottleneck? → Use viterbi approximation

## Testing Commands

```bash
# Test VQ layer standalone
python -c "from models.vq_layer import VectorQuantizer; vq = VectorQuantizer(); print('VQ initialized')"

# Test with dummy data
python -c "
import torch
from models.conv2d_vq_hdp_hsmm import Conv2dVQHDPHSMM
model = Conv2dVQHDPHSMM()
dummy = torch.randn(4, 9, 2, 100)
output = model(dummy)
print(f'Output shape: {output[\"reconstructed\"].shape}')
"

# Validate Hailo compatibility
python scripts/validate_hailo_ops.py --model models/conv2d_vq_hdp_hsmm.py
```

## Key Metrics to Monitor

### During Training
```python
metrics = {
    'reconstruction_loss': [],     # Should decrease
    'vq_loss': [],                 # Should stabilize
    'perplexity': [],              # Should be > 50% of codebook size
    'active_clusters': [],         # Should be 5-15
    'state_transitions': [],       # Should show variety
    'mean_duration': [],           # Should match expected behavior
}
```

### Validation Metrics
- Behavioral classification accuracy
- Reconstruction quality (MSE, PSNR)
- Codebook utilization (perplexity)
- Cluster coherence (silhouette score)
- Temporal consistency (state persistence)

## Common Pitfalls & Solutions

### Pitfall 1: Codebook Not Learning
**Symptom**: Low perplexity, poor reconstruction
**Solution**: 
- Reduce learning rate for codebook
- Increase commitment cost
- Use EMA updates for codebook

### Pitfall 2: Mode Collapse in HDP
**Symptom**: All data assigned to few clusters
**Solution**:
- Add entropy regularization
- Use different initialization
- Anneal temperature parameter

### Pitfall 3: HSMM Too Slow
**Symptom**: Training takes forever
**Solution**:
- Use mini-batch forward-backward
- Implement in CUDA if needed
- Consider approximations

## Resources & References

### Your Existing Code
- Encoder: `models/tcn_vae_hailo.py:HailoTemporalConvNet`
- Dataset: `preprocessing/enhanced_pipeline.py:EnhancedCrossSpeciesDataset`
- Config: `configs/enhanced_dataset_schema.yaml`

### External Resources
- [VQ-VAE Tutorial](https://github.com/zalandoresearch/pytorch-vq-vae)
- [HDP Implementation](https://github.com/blei-lab/hdp)
- [HSMM in PyTorch](https://github.com/lindermanlab/ssm)

### Papers to Reference
- VQ-VAE: arXiv:1711.00937
- HDP: JASA 2006
- HSMM: IEEE TSP 2010

## Questions to Answer Before Starting

1. **Codebook Size**: How many behavioral primitives?
   - Start with 256-512
   - Monitor utilization
   - Adjust based on perplexity

2. **Training Strategy**: Staged or end-to-end?
   - Recommend: Stage 1 (Encoder+VQ), Stage 2 (Add HDP), Stage 3 (Add HSMM)
   - Benefit: Easier debugging
   - Trade-off: Longer total training

3. **Computational Budget**: How much complexity?
   - VQ: Minimal overhead
   - HDP: Moderate (depends on max_clusters)
   - HSMM: Potentially expensive (can optimize)

## Next Immediate Steps

1. **Create new repo and copy essentials** (30 min)
2. **Implement basic VQ layer** (2 hours)
3. **Test VQ with existing encoder** (1 hour)
4. **Validate gradient flow** (30 min)
5. **Run small training test** (1 hour)

Total Day 1 Goal: Working VQ layer integrated with existing encoder

---

*Status: Ready for implementation*
*First milestone: Working VQ layer by end of Day 1*
