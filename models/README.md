# Models Module Documentation

This module contains the core TCN-VAE model architecture and related components.

## Files Overview

### `tcn_vae.py`
The main model architecture file containing all neural network components.

#### Classes

##### `TemporalConvNet`
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

**Architecture:**
- Multiple `TemporalBlock` layers with increasing dilation
- Exponential dilation: 2^0, 2^1, 2^2, ... 
- Residual connections for gradient flow

##### `TemporalBlock`
Individual temporal block with dilated convolutions and residual connections.

```python
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2)
```

**Components:**
- Weight-normalized 1D convolutions
- Chomp padding for causal convolutions
- ReLU activations and dropout
- Residual skip connections

##### `Chomp1d`
Removes padding to ensure causal convolutions (future information doesn't leak).

```python
class Chomp1d(nn.Module):
    def __init__(self, chomp_size)
```

##### `TCNVAE` (Main Model)
Complete TCN-VAE architecture combining encoder, decoder, and classification heads.

```python
class TCNVAE(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[64, 128, 256], latent_dim=64, 
                 sequence_length=100, num_activities=12)
```

**Parameters:**
- `input_dim` (int): Input feature dimension (9 for 9-axis IMU)
- `hidden_dims` (list): TCN hidden dimensions
- `latent_dim` (int): Latent space dimension (64)
- `sequence_length` (int): Input sequence length (100)
- `num_activities` (int): Number of activity classes

**Components:**

1. **TCN Encoder**: Encodes temporal sequences to fixed-size representations
2. **VAE Components**: 
   - `fc_mu`: Linear layer for mean vector
   - `fc_logvar`: Linear layer for log variance
3. **Decoder**: Reconstructs original sequences from latent codes
4. **Activity Classifier**: Supervised learning head for activity recognition
5. **Domain Classifier**: Adversarial head for domain adaptation

**Forward Pass:**
```python
def forward(self, x, alpha=1.0):
    # Returns: x_recon, mu, logvar, activity_logits, domain_logits
```

**Key Methods:**

- `encode(x)`: Encodes input to latent distribution parameters
- `reparameterize(mu, logvar)`: VAE reparameterization trick
- `decode(z, sequence_length)`: Decodes latent codes to sequences

##### `ReverseLayerF`
Gradient reversal layer for adversarial domain adaptation.

```python
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): # Forward pass unchanged
    
    @staticmethod  
    def backward(ctx, grad_output): # Reverses gradients with scaling
```

## Model Architecture Flow

```
Input [B, 100, 9] 
    ↓
TCN Encoder [B, 256]
    ↓ 
VAE Latent [B, 64] (mu, logvar)
    ↓
Reparameterize [B, 64] (z)
    ↓
┌─ TCN Decoder → Reconstruction [B, 100, 9]
├─ Activity Classifier → Activity Logits [B, num_classes]  
└─ Domain Classifier → Domain Logits [B, 3]
```

## Training Objectives

The model optimizes multiple objectives simultaneously:

1. **Reconstruction Loss**: MSE between input and reconstructed sequences
2. **KL Divergence**: Regularizes latent space to unit Gaussian
3. **Activity Classification**: Cross-entropy for supervised learning
4. **Domain Classification**: Adversarial loss for domain invariance

## Model Checkpoints

### File Naming Convention
- `best_tcn_vae.pth`: Best validation accuracy model
- `best_overnight_tcn_vae.pth`: Best from overnight training (72.13%)
- `final_tcn_vae.pth`: Final epoch model
- `checkpoint_epoch_N.pth`: Periodic checkpoints

### Loading Models
```python
model = TCNVAE(input_dim=9, hidden_dims=[64, 128, 256], 
               latent_dim=64, num_activities=13)
model.load_state_dict(torch.load('models/best_overnight_tcn_vae.pth'))
model.eval()
```

## Performance Characteristics

### Model Size
- **Parameters**: ~1.1M trainable parameters
- **Memory**: ~50MB model size
- **Latent Space**: 64-dimensional continuous representation

### Inference Speed
- **Forward Pass**: ~10ms on GPU (RTX 2060)
- **Batch Processing**: ~100 sequences/second
- **Edge Deployment**: <50ms on Hailo-8

### Accuracy Breakdown
- **Overall Validation**: 72.13%
- **Cross-dataset Generalization**: Good domain adaptation
- **Activity Recognition**: Strong performance on walking, running, sitting

## Usage Examples

### Basic Inference
```python
import torch
from models.tcn_vae import TCNVAE

# Load model
model = TCNVAE(input_dim=9, latent_dim=64, num_activities=13)
model.load_state_dict(torch.load('best_overnight_tcn_vae.pth'))
model.eval()

# Inference
with torch.no_grad():
    imu_data = torch.randn(1, 100, 9)  # Batch of 1, 100 timesteps, 9 sensors
    recon, mu, logvar, activity_logits, domain_logits = model(imu_data)
    
    predicted_activity = activity_logits.argmax(dim=1)
    latent_embedding = mu  # Use mean for deterministic encoding
```

### Feature Extraction
```python
# Extract only latent features
mu, logvar = model.encode(imu_data)
features = mu  # 64-dimensional activity representation
```

### Activity Classification
```python
# Get activity probabilities
activity_probs = torch.softmax(activity_logits, dim=1)
confidence = activity_probs.max().item()
```

## Integration Notes

### With EdgeInfer
- Model encoder is exported to ONNX for edge deployment
- Normalization must match training preprocessing exactly
- Static input shapes required for Hailo-8 compilation

### With Training Pipeline
- Model supports progressive adversarial training
- Loss weighting balances reconstruction vs classification
- Compatible with cosine annealing learning rate schedules

### With Evaluation
- Model provides multiple outputs for comprehensive evaluation
- Latent space can be visualized with t-SNE
- Cross-dataset performance can be assessed via domain classification