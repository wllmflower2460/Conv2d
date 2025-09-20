# Conv2d-VQ-HDP-HSMM

A PyTorch implementation of Conv2d-VQ-HDP-HSMM for behavioral analysis on edge devices like Hailo-8. This model combines Temporal Convolutional Networks (TCN), Vector Quantization (VQ), Hierarchical Dirichlet Process (HDP) clustering, and Hidden Semi-Markov Models (HSMM) for real-time behavioral pattern recognition from dual-device IMU data.

## Features

- **Conv2d-Only Architecture**: Fully compatible with edge deployment constraints
- **Vector Quantization**: 512-code discrete latent space with straight-through estimator
- **HDP Clustering**: Hierarchical clustering with up to 20 adaptive clusters
- **HSMM Temporal Modeling**: Forward-backward algorithm for state sequence modeling
- **Real-time Processing**: Optimized for edge devices like Hailo-8
- **Behavioral Analysis**: Comprehensive monitoring and pattern detection

## Architecture

```
Input: (B, 9, 2, 100) - Dual-device IMU data
  ↓
TCN Encoder (Conv2d blocks with dilated convolutions)
  ↓
Vector Quantization (512 codes, dim=64)
  ↓
HDP Clustering (max 20 clusters)
  ↓
HSMM Temporal Dynamics (10 states, max duration 10)
  ↓
TCN Decoder (Conv2d blocks)
  ↓
Output: Reconstructed input + behavioral analysis
```

## Installation

```bash
# Clone the repository
git clone https://github.com/wllmflower2460/Conv2d.git
cd Conv2d

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
import torch
from conv2d_vq_hdp_hsmm import Conv2dVQHDPHSMM

# Initialize model
model = Conv2dVQHDPHSMM(
    input_channels=9,      # 3x3 IMU channels per device
    input_features=2,      # 2 devices
    sequence_length=100,   # Time steps
    latent_dim=64,         # Latent dimension
    num_embeddings=512,    # VQ codebook size
    max_clusters=20,       # Max HDP clusters
    num_states=10,         # HSMM states
    max_duration=10        # Max state duration
)

# Example input: dual-device IMU data
# Shape: (batch_size, 9, 2, 100)
x = torch.randn(8, 9, 2, 100)

# Forward pass
outputs = model(x)

# Extract behavioral analysis
behavioral_analysis = model.get_behavioral_analysis(outputs)
print(f"Active clusters: {behavioral_analysis['active_clusters']}")
print(f"Codebook utilization: {behavioral_analysis['codebook_utilization']:.3f}")
print(f"Average transition rate: {behavioral_analysis['avg_transition_rate']:.3f}")
```

## Components

### Vector Quantization (VQ)
- **Purpose**: Create discrete latent representations for robust behavioral encoding
- **Features**: Straight-through estimator, exponential moving average updates, perplexity monitoring
- **Configuration**: 512 codebook vectors, 64 dimensions

### Hierarchical Dirichlet Process (HDP)
- **Purpose**: Adaptive clustering without fixed number of clusters
- **Features**: Stick-breaking process, automatic cluster discovery, up to 20 clusters
- **Benefits**: Handles varying behavioral complexity across sessions

### Hidden Semi-Markov Model (HSMM)
- **Purpose**: Temporal sequence modeling with explicit state durations
- **Features**: Forward-backward algorithm, Viterbi decoding, duration modeling
- **Applications**: Capture behavioral transitions and persistence patterns

## Training Example

```python
import torch.optim as optim
from conv2d_vq_hdp_hsmm.utils import create_monitoring_suite, update_monitors

# Initialize model and monitoring
model = Conv2dVQHDPHSMM()
monitors = create_monitoring_suite(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    # Your data loading here
    data = load_imu_data()  # Shape: (B, 9, 2, 100)
    
    # Forward pass
    outputs = model(data)
    
    # Compute losses
    reconstruction_loss = torch.nn.MSELoss()(outputs['reconstructed'], data)
    outputs['reconstruction_loss'] = reconstruction_loss
    
    total_loss, loss_components = model.compute_total_loss(outputs)
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Update monitoring
    update_monitors(monitors, outputs)
```

## Edge Deployment

The model is specifically designed for edge deployment with the following constraints:

- **Conv2d Only**: No Conv1d, GroupNorm, or Softmax operations
- **Memory Efficient**: Optimized parameter usage
- **Real-time Capable**: Fast inference for streaming data
- **Hailo-8 Compatible**: Tested deployment constraints

### Deployment Checklist

```python
from conv2d_vq_hdp_hsmm.utils import EdgeDeploymentAnalyzer

# Analyze model for edge compatibility
analysis = EdgeDeploymentAnalyzer.analyze_model(model)
print(f"Edge ready: {analysis['edge_ready']}")
print(f"Total parameters: {analysis['total_parameters']:,}")
print(f"Estimated memory: {analysis['estimated_memory_mb']:.1f} MB")

# Benchmark inference speed
benchmark = EdgeDeploymentAnalyzer.benchmark_inference(
    model, input_shape=(1, 9, 2, 100)
)
print(f"Inference time: {benchmark['mean_inference_time_ms']:.2f} ms")
print(f"Throughput: {benchmark['throughput_fps']:.1f} FPS")
```

## Input Data Format

The model expects dual-device IMU data in the following format:

```python
input_shape = (batch_size, 9, 2, 100)
```

Where:
- **batch_size**: Number of samples in the batch
- **9 channels**: 3 accelerometer + 3 gyroscope + 3 magnetometer per device
- **2 devices**: Dual-device setup (e.g., wrist and ankle sensors)
- **100 time steps**: Temporal sequence length

## Monitoring and Analysis

The package includes comprehensive monitoring tools:

```python
from conv2d_vq_hdp_hsmm.utils import PerplexityMonitor, ClusterMonitor, StateTransitionAnalyzer

# Create monitors
perplexity_monitor = PerplexityMonitor()
cluster_monitor = ClusterMonitor(max_clusters=20)
transition_analyzer = StateTransitionAnalyzer(num_states=10)

# Update during training/inference
perplexity_monitor.update(perplexity, utilization, active_codes)
cluster_monitor.update(cluster_assignments)
transition_analyzer.update(state_sequence)

# Get statistics
perp_stats = perplexity_monitor.get_stats()
cluster_stats = cluster_monitor.get_stats()
transition_matrix = transition_analyzer.get_transition_matrix()
```

## Example Applications

1. **Activity Recognition**: Walking, running, sitting, standing detection
2. **Behavioral Pattern Analysis**: Routine identification and anomaly detection  
3. **Health Monitoring**: Gait analysis, movement quality assessment
4. **Sports Analytics**: Performance pattern recognition and optimization
5. **Assistive Technology**: Context-aware adaptive systems

## Testing

Run the test suite to verify installation and functionality:

```bash
# Run basic component tests
python tests/test_components.py

# Run full example with synthetic data
python example_usage.py
```

## Model Performance

- **Latent Space Size**: 64 dimensions
- **Codebook Size**: 512 discrete codes
- **Maximum Clusters**: 20 adaptive clusters
- **Temporal States**: 10 HSMM states
- **Parameters**: ~2M parameters (edge-optimized)
- **Inference Speed**: <10ms per sample (CPU)

## Requirements

- Python 3.7+
- PyTorch 1.10+
- NumPy 1.21+
- SciPy 1.7+

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{conv2d_vq_hdp_hsmm,
  title={Conv2d-VQ-HDP-HSMM: Behavioral Analysis on Edge Devices},
  author={Conv2d-VQ-HDP-HSMM Team},
  year={2024},
  url={https://github.com/wllmflower2460/Conv2d}
}
```

## Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## Support

For questions and support, please open an issue on GitHub or contact the development team.
