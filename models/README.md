# Model Components Documentation

This directory contains all model components for the Conv2d-VQ-HDP-HSMM architecture, implementing a unified framework for behavioral synchrony analysis.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────┐
│         Conv2d-VQ-HDP-HSMM Model        │
├─────────────────────────────────────────┤
│ 1. Conv2d Encoder (Feature Extraction)  │
│ 2. Vector Quantization (Discrete Codes) │
│ 3. HDP Clustering (Behavioral Groups)   │
│ 4. HSMM Dynamics (Temporal Modeling)    │
│ 5. Entropy Module (Uncertainty)         │
└─────────────────────────────────────────┘
```

## 📁 File Structure

### Core Components

#### `vq_ema_2d.py` - Vector Quantization with EMA
- **Purpose**: Learns discrete behavioral vocabulary
- **Key Features**:
  - EMA (Exponential Moving Average) codebook updates
  - 512 codes × 64 dimensions default
  - Straight-through estimator for gradients
  - Hailo-safe implementation (no unsupported ops)
- **Usage**:
```python
vq = VectorQuantizerEMA2D(num_codes=512, code_dim=64)
z_q, loss_dict, info = vq(z_e)  # z_e: (B, D, H, T)
```

#### `hdp_components.py` - Hierarchical Dirichlet Process
- **Purpose**: Automatic discovery of behavioral clusters
- **Key Features**:
  - Stick-breaking construction for non-parametric clustering
  - Hierarchical HDP for two-level clustering
  - Temperature annealing for stable training
  - Gumbel-Softmax for differentiable assignments
- **Components**:
  - `StickBreaking`: Generates cluster weights
  - `HDPLayer`: Single-level clustering
  - `HierarchicalHDP`: Two-level hierarchical clustering
- **Usage**:
```python
hdp = HDPLayer(input_dim=64, max_clusters=20)
cluster_assignments, info = hdp(features)  # features: (B, T, D)
```

#### `hsmm_components.py` - Hidden Semi-Markov Model
- **Purpose**: Models temporal dynamics with explicit duration
- **Key Features**:
  - Multiple duration distributions (negative binomial, Poisson, Gaussian)
  - Forward-backward algorithm for inference
  - Viterbi decoding for most likely path
  - Input-dependent transition matrices
- **Components**:
  - `DurationModel`: Models state durations
  - `HSMMTransitions`: State transition probabilities
  - `HSMM`: Complete model with inference
- **Usage**:
```python
hsmm = HSMM(num_states=10, observation_dim=64)
state_probs, info = hsmm(observations, return_viterbi=True)
```

#### `entropy_uncertainty.py` - Uncertainty Quantification
- **Purpose**: Quantifies model confidence for clinical deployment
- **Key Features**:
  - Shannon entropy for discrete distributions
  - Circular statistics for phase analysis
  - Mutual information I(Z;Φ) calculation
  - Confidence calibration (ECE, Brier scores)
  - Three-level confidence (high/medium/low)
- **Components**:
  - `EntropyUncertaintyModule`: Main uncertainty module
  - `CircularStatistics`: Helper for phase analysis
  - `summarize_window`: Quick analysis function
- **Usage**:
```python
entropy_module = EntropyUncertaintyModule(num_states=10)
uncertainty = entropy_module(state_posterior, phase_values)
confidence_level = uncertainty['confidence_level']  # 'high', 'medium', or 'low'
```

### Integrated Models

#### `conv2d_vq_model.py` - Conv2d-VQ Integration
- **Purpose**: Combines Conv2d encoder with VQ layer
- **Architecture**:
  - Conv2d encoder: IMU → continuous features
  - VQ layer: continuous → discrete codes
  - Conv2d decoder: discrete → reconstruction
  - Activity classifier head
- **Model Size**: ~305K parameters
- **Usage**:
```python
model = Conv2dVQModel(input_channels=9, input_height=2, num_codes=512)
outputs = model(imu_data)  # imu_data: (B, 9, 2, 100)
```

#### `conv2d_vq_hdp_hsmm.py` - Complete Architecture
- **Purpose**: Full integrated model with all components
- **Architecture Flow**:
  1. Conv2d Encoder → continuous features
  2. VQ → discrete behavioral codes
  3. HDP → automatic cluster discovery
  4. HSMM → temporal dynamics
  5. Entropy → uncertainty quantification
- **Model Size**: 313K parameters
- **Key Outputs**:
  - Behavioral codes with perplexity
  - Cluster assignments
  - Temporal states with durations
  - Confidence levels and intervals
- **Usage**:
```python
model = Conv2dVQHDPHSMM(
    input_channels=9,
    num_codes=512,
    max_clusters=20,
    num_states=10
)
outputs = model(imu_data, return_all_stages=True)
```

### Legacy Models

#### `tcn_vae.py` - Original TCN-VAE
The original TCN-VAE implementation that achieved 78.12% accuracy on quadruped behavioral recognition.

##### Classes

###### `TemporalConvNet`
Temporal Convolutional Network implementation with dilated convolutions.

```python
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2)
```

**Parameters:**
- `num_inputs` (int): Number of input channels (e.g., 9 for 9-axis IMU)
- `num_channels` (list): List of hidden channel dimensions [64, 128, 256]
- `kernel_size` (int): Convolution kernel size (default: 3)
- `dropout` (float): Dropout rate for regularization (default: 0.2)

###### `TCNVAE` (Main Model)
Complete TCN-VAE architecture combining encoder, decoder, and classification heads.

```python
class TCNVAE(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[64, 128, 256], latent_dim=64, 
                 sequence_length=100, num_activities=12)
```

**Components:**
1. **TCN Encoder**: Encodes temporal sequences to fixed-size representations
2. **VAE Components**: 
   - `fc_mu`: Linear layer for mean vector
   - `fc_logvar`: Linear layer for log variance
3. **Decoder**: Reconstructs original sequences from latent codes
4. **Activity Classifier**: Supervised learning head for activity recognition
5. **Domain Classifier**: Adversarial head for domain adaptation

## 🚀 Quick Start

### Test Individual Components
```bash
# Test VQ layer
python models/vq_ema_2d.py

# Test HDP clustering
python models/hdp_components.py

# Test HSMM dynamics
python models/hsmm_components.py

# Test entropy module
python models/entropy_uncertainty.py

# Test complete model
python models/conv2d_vq_hdp_hsmm.py
```

### Training Pipeline
```bash
# Train Conv2d-VQ model
python training/train_conv2d_vq.py

# Analyze learned codes
python analysis/codebook_analysis.py
```

## 📊 Key Metrics

### VQ Layer
- **Perplexity**: 100-150 (healthy codebook usage)
- **Usage**: 40-55% of codes actively used
- **Reconstruction**: Low MSE loss

### HDP Clustering
- **Active Clusters**: 5-10 automatically discovered
- **Entropy**: Measures cluster uncertainty
- **Balance**: Even distribution across clusters

### HSMM Dynamics
- **State Duration**: Mean ~2-5 timesteps
- **Transitions**: Realistic behavioral switches
- **Viterbi Path**: Most likely state sequence

### Uncertainty
- **Confidence Levels**: High (<0.3 entropy), Medium (0.3-0.6), Low (>0.6)
- **Mutual Information**: I(Z;Φ) for synchrony coherence
- **Calibration**: ECE and Brier scores for reliability

## 🔬 Research Significance

This architecture represents the first implementation of:
1. **Behavioral-Dynamical Coherence**: Mutual information I(Z;Φ) between discrete states and continuous phase
2. **Unified Framework**: Bridging Feldman's discrete and Kelso's continuous models
3. **Clinical Uncertainty**: Confidence-aware predictions for medical deployment
4. **Cross-species Analysis**: Separate tracking of human and animal behaviors

## 📚 References

- VQ-VAE: Van Den Oord et al. (2017) - Neural Discrete Representation Learning
- HDP: Teh et al. (2006) - Hierarchical Dirichlet Processes  
- HSMM: Yu (2010) - Hidden Semi-Markov Models
- Behavioral Synchrony: Feldman (2007), Kelso (1995), Leclère (2014)

## 🛠️ Development Notes

- All models support gradient flow for end-to-end training
- Hailo-safe implementations avoid unsupported operations
- Models can be used independently or in combination
- Comprehensive test functions included in each file

---

For detailed implementation documentation, see `IMPLEMENTATION_SUMMARY.md` in the project root.