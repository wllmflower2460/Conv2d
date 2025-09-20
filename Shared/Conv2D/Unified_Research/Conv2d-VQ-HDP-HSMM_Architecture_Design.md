# Conv2d-VQ-HDP-HSMM Architecture Design
*Created: 2025-09-19*
*Project: Advanced Behavioral Analysis with Discrete Representations*

## Executive Summary

The Conv2d-VQ-HDP-HSMM architecture represents a sophisticated evolution of the existing TCN-VAE training pipeline, introducing:
- **Vector Quantization (VQ)**: Discrete representation learning for interpretable behavioral primitives
- **Hierarchical Dirichlet Process (HDP)**: Automatic discovery of behavioral clusters
- **Hidden Semi-Markov Model (HSMM)**: Advanced temporal dynamics with duration modeling
- **Conv2d Base**: Maintains Hailo-8 compatibility through Conv2d operations

## Architecture Overview

### Core Innovation
Transform continuous latent representations into discrete behavioral codes that can be hierarchically organized and temporally modeled with explicit duration states.

```
Input (Phone+IMU) → Conv2d Encoder → VQ Layer → HDP Clustering → HSMM Dynamics → Decoder
     (B,9,2,T)         (Reused)      (Discrete)   (Hierarchical)   (Temporal)    (Reused)
```

## Component Breakdown

### 1. Vector Quantization Layer
**Purpose**: Convert continuous representations to discrete codes
**Key Features**:
- Learnable codebook of behavioral primitives
- Straight-through estimator for gradient flow
- Commitment loss for stable training
- Perplexity monitoring for codebook utilization

**Implementation Notes**:
```python
# Codebook size determines behavioral primitive vocabulary
codebook_size = 512  # Number of discrete codes
codebook_dim = 64    # Dimension of each code
commitment_cost = 0.25  # Balance reconstruction vs quantization
```

### 2. Hierarchical Dirichlet Process (HDP)
**Purpose**: Automatically discover optimal number of behavioral clusters
**Key Features**:
- Non-parametric Bayesian clustering
- Stick-breaking construction
- Infinite mixture model capability
- Hailo-compatible through fixed max_clusters

**Implementation Notes**:
```python
max_clusters = 20  # Static upper bound for Hailo
# Uses Gumbel-softmax for differentiable cluster assignment
# Monitors active clusters dynamically
```

### 3. Hidden Semi-Markov Model (HSMM)
**Purpose**: Model temporal dynamics with explicit duration states
**Key Features**:
- State duration modeling (unlike standard HMM)
- Captures behavioral persistence
- Forward-backward algorithm for inference
- Transition and emission probabilities

**Implementation Notes**:
```python
num_states = 10      # Number of hidden states
max_duration = 50    # Maximum state duration
# Gaussian duration model per state
# Learnable transition matrix
```

## Reusable Components from TCN-VAE Pipeline

### ✅ Can Reuse Directly:
1. **HailoTemporalBlock** (`models/tcn_vae_hailo.py`)
   - Conv2d-based temporal convolutions
   - Dilated causal convolutions
   - Already Hailo-validated

2. **DeviceAttention** (`models/device_attention.py`)
   - Phone+IMU dual-device processing
   - Cross-device attention mechanism
   - Synchrony measurement

3. **EnhancedCrossSpeciesDataset** (`preprocessing/enhanced_pipeline.py`)
   - YAML-driven configuration
   - Cross-species behavioral mappings
   - Data augmentation pipeline

4. **HailoDataValidator** (`preprocessing/enhanced_pipeline.py`)
   - Tensor shape validation
   - Operation compatibility checking
   - Static shape enforcement

5. **Training Infrastructure** (`training/`)
   - Monitoring scripts
   - Progress reporting
   - Overnight training setup

### ⚠️ Needs Modification:
1. **VAE Components** → Replace with VQ quantization
2. **Loss Functions** → Add VQ commitment + HDP regularization
3. **Decoder** → Adapt for discrete codebook reconstruction
4. **Evaluation Metrics** → Add perplexity, cluster quality

## Implementation Strategy

### Phase 1: VQ Integration (Week 1)
- [ ] Create `models/vq_components.py`
- [ ] Implement VectorQuantizer class
- [ ] Add commitment loss
- [ ] Test codebook learning
- [ ] Validate Hailo compatibility

### Phase 2: HDP Clustering (Week 2)
- [ ] Create `models/hdp_components.py`
- [ ] Implement HDPLayer with stick-breaking
- [ ] Add cluster assignment logic
- [ ] Test cluster discovery
- [ ] Monitor active clusters

### Phase 3: HSMM Dynamics (Week 3)
- [ ] Create `models/hsmm_components.py`
- [ ] Implement forward-backward algorithm
- [ ] Add duration modeling
- [ ] Test temporal coherence
- [ ] Validate state transitions

### Phase 4: Integration & Training (Week 4)
- [ ] Combine all components
- [ ] Create unified training script
- [ ] Implement combined loss
- [ ] Run validation experiments
- [ ] Export for Hailo deployment

## Project Structure

```
New-Conv2d-VQ-HDP-HSMM-Repo/
├── models/
│   ├── conv2d_vq_hdp_hsmm.py      # Main architecture
│   ├── vq_components.py            # VQ modules
│   ├── hdp_components.py           # HDP clustering
│   ├── hsmm_components.py          # HSMM temporal
│   └── [COPY] tcn_vae_hailo.py    # Base components
├── training/
│   ├── train_vq_hdp_hsmm.py       # Training script
│   └── [COPY] train_utils.py      # Reused utilities
├── losses/
│   ├── vq_losses.py               # VQ-specific
│   ├── hdp_losses.py              # HDP regularization
│   └── temporal_losses.py         # HSMM losses
├── preprocessing/
│   └── [COPY] enhanced_pipeline.py # Data pipeline
├── configs/
│   ├── vq_hdp_hsmm_config.yaml    # Model config
│   └── [COPY] enhanced_dataset_schema.yaml
└── evaluation/
    ├── behavioral_analysis.py      # New metrics
    └── [COPY] evaluate_model.py   # Base evaluation
```

## Key Design Decisions

### 1. Why VQ-VAE over Standard VAE?
- **Discrete representations** are more interpretable
- **Codebook** provides behavioral vocabulary
- **Better generalization** through quantization
- **Natural clustering** of similar behaviors

### 2. Why HDP over Fixed Clustering?
- **Automatic discovery** of cluster count
- **Hierarchical organization** of behaviors
- **Bayesian uncertainty** quantification
- **Flexible complexity** based on data

### 3. Why HSMM over HMM?
- **Duration modeling** captures behavior persistence
- **More realistic** for behavioral sequences
- **Better handling** of varying activity lengths
- **Explicit state duration** distributions

## Technical Challenges & Solutions

### Challenge 1: Gradient Flow through Discrete Operations
**Solution**: Straight-through estimator with stop-gradient

### Challenge 2: Dynamic Cluster Count vs Hailo Static Shapes
**Solution**: Fixed max_clusters with activity monitoring

### Challenge 3: HSMM Computational Complexity
**Solution**: Approximations using forward-backward with pruning

### Challenge 4: Training Stability
**Solution**: Staged training - encoder first, then VQ, then HDP/HSMM

## Performance Targets

### Model Quality
- Reconstruction MSE: < 0.1
- Codebook Perplexity: > 100 (good utilization)
- Behavior Classification: > 85% accuracy
- Active Clusters: 5-15 (reasonable granularity)

### Computational Efficiency
- Inference Latency: < 15ms (Hailo-8)
- Model Size: < 10MB (HEF format)
- Power Consumption: < 2W
- Memory Usage: < 150MB

## Integration with Existing Pipeline

### Data Flow
1. Use existing `EnhancedCrossSpeciesDataset`
2. Feed through new Conv2d-VQ-HDP-HSMM model
3. Reuse validation tools for Hailo checking
4. Export using modified ONNX pipeline

### Training Flow
1. Leverage existing monitoring infrastructure
2. Add VQ-specific metrics to logging
3. Use established checkpointing system
4. Apply same train/val/test splits

## Next Steps

1. **Setup New Repository**
   ```bash
   git clone existing_tcn_vae Conv2d-VQ-HDP-HSMM
   cd Conv2d-VQ-HDP-HSMM
   git checkout -b vq-hdp-hsmm-development
   ```

2. **Copy Reusable Components**
   ```bash
   # Copy core modules
   cp ../TCN-VAE_Training_Pipeline-/models/tcn_vae_hailo.py models/
   cp ../TCN-VAE_Training_Pipeline-/preprocessing/enhanced_pipeline.py preprocessing/
   cp -r ../TCN-VAE_Training_Pipeline-/configs/ .
   ```

3. **Implement VQ Layer First**
   - Start with basic vector quantization
   - Test on small dataset
   - Validate gradients flow correctly

4. **Incremental Development**
   - Add one component at a time
   - Validate each addition
   - Maintain Hailo compatibility throughout

## Questions for Clarification

### Architecture Questions
1. **Codebook Size**: 512 sufficient or need larger?
2. **Temporal Window**: Keep 100 timesteps or adjust?
3. **Device Dimension**: Stay with H=2 or plan for expansion?

### Training Questions
1. **Dataset Priority**: Which datasets to use first?
2. **Loss Weighting**: How to balance VQ vs classification?
3. **Training Schedule**: Staged or end-to-end?

### Deployment Questions
1. **Target Hardware**: Hailo-8 only or multiple targets?
2. **Latency Requirements**: Strict 10ms or flexible?
3. **Model Variants**: Single model or size variants?

## References & Resources

### Key Papers
- Van Den Oord et al. (2017) - "Neural Discrete Representation Learning" (VQ-VAE)
- Teh et al. (2006) - "Hierarchical Dirichlet Processes"
- Yu (2010) - "Hidden Semi-Markov Models"

### Existing Codebase
- TCN-VAE Pipeline: `/Users/willflower/Developer/data-dogs/TCN-VAE_Training_Pipeline-/`
- Conv2d Documentation: `CONV2D_ARCHITECTURE_DOCUMENTATION.md`
- Training Config: `config/training_config.py`

### Related Projects
- Original TCN-VAE implementation
- Hailo export pipeline
- Cross-species dataset management

## Notes & Observations

### From Exploration
- Excellent Conv2d foundation already in place
- Device attention mechanism is sophisticated
- YAML configuration system is very clean
- Hailo validation tools are comprehensive

### Opportunities
- VQ will provide interpretable behavioral codes
- HDP can discover natural behavioral hierarchies
- HSMM will capture realistic duration patterns
- Combined system could revolutionize behavioral analysis

### Risks
- Complexity may affect training stability
- Computational overhead needs monitoring
- Hailo compilation might need iterations
- Integration testing will be critical

---

*Last Updated: 2025-09-19*
*Status: Design Phase - Ready for Implementation*
