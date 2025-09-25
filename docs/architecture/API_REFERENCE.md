# API Reference

Comprehensive API documentation for the TCN-VAE training pipeline.

## Core Classes

### `TCNVAE`
**Location**: `models/tcn_vae.py`

Main TCN-VAE model class combining temporal convolutions with variational autoencoders.

#### Constructor
```python
TCNVAE(input_dim=9, hidden_dims=[64, 128, 256], latent_dim=64, 
       sequence_length=100, num_activities=12)
```

**Parameters:**
- `input_dim` (int): Input feature dimension (default: 9 for 9-axis IMU)
- `hidden_dims` (list): TCN hidden channel dimensions (default: [64, 128, 256])
- `latent_dim` (int): Latent space dimension (default: 64)
- `sequence_length` (int): Input sequence length (default: 100)
- `num_activities` (int): Number of activity classes (default: 12)

#### Methods

##### `forward(x, alpha=1.0)`
Forward pass through the complete model.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
- `alpha` (float): Gradient reversal strength for domain adaptation (default: 1.0)

**Returns:**
- `x_recon` (torch.Tensor): Reconstructed sequences (batch_size, sequence_length, input_dim)
- `mu` (torch.Tensor): Mean vectors (batch_size, latent_dim)
- `logvar` (torch.Tensor): Log variance vectors (batch_size, latent_dim)
- `activity_logits` (torch.Tensor): Activity classification logits (batch_size, num_activities)
- `domain_logits` (torch.Tensor): Domain classification logits (batch_size, 3)

##### `encode(x)`
Encode input sequences to latent distribution parameters.

**Parameters:**
- `x` (torch.Tensor): Input tensor (batch_size, sequence_length, input_dim)

**Returns:**
- `mu` (torch.Tensor): Mean vectors (batch_size, latent_dim)
- `logvar` (torch.Tensor): Log variance vectors (batch_size, latent_dim)

##### `decode(z, sequence_length)`
Decode latent codes back to sequences.

**Parameters:**
- `z` (torch.Tensor): Latent codes (batch_size, latent_dim)
- `sequence_length` (int): Target sequence length

**Returns:**
- `x_recon` (torch.Tensor): Reconstructed sequences (batch_size, sequence_length, input_dim)

##### `reparameterize(mu, logvar)`
VAE reparameterization trick for sampling from latent distribution.

**Parameters:**
- `mu` (torch.Tensor): Mean vectors (batch_size, latent_dim)
- `logvar` (torch.Tensor): Log variance vectors (batch_size, latent_dim)

**Returns:**
- `z` (torch.Tensor): Sampled latent codes (batch_size, latent_dim)

### `TemporalConvNet`
**Location**: `models/tcn_vae.py`

Temporal Convolutional Network implementation with dilated convolutions.

#### Constructor
```python
TemporalConvNet(num_inputs, num_channels, kernel_size=3, dropout=0.2)
```

**Parameters:**
- `num_inputs` (int): Number of input channels
- `num_channels` (list): List of hidden channel dimensions
- `kernel_size` (int): Convolution kernel size (default: 3)
- `dropout` (float): Dropout probability (default: 0.2)

#### Methods

##### `forward(x)`
Forward pass through temporal convolutional layers.

**Parameters:**
- `x` (torch.Tensor): Input tensor (batch_size, input_channels, sequence_length)

**Returns:**
- `output` (torch.Tensor): Encoded features (batch_size, output_channels, sequence_length)

### `MultiDatasetHAR`
**Location**: `preprocessing/unified_pipeline.py`

Multi-dataset preprocessing pipeline for human activity recognition.

#### Constructor
```python
MultiDatasetHAR(window_size=100, overlap=0.5)
```

**Parameters:**
- `window_size` (int): Sliding window size in timesteps (default: 100)
- `overlap` (float): Window overlap ratio 0.0-1.0 (default: 0.5)

#### Methods

##### `preprocess_all()`
Load and preprocess all supported datasets.

**Returns:**
- `X_train` (numpy.ndarray): Training data (N, window_size, input_dim)
- `y_train` (numpy.ndarray): Training labels (N,)
- `domains_train` (numpy.ndarray): Training domain labels (N,)
- `X_val` (numpy.ndarray): Validation data (N, window_size, input_dim)
- `y_val` (numpy.ndarray): Validation labels (N,)
- `domains_val` (numpy.ndarray): Validation domain labels (N,)

##### `load_pamap2(data_path)`
Load PAMAP2 dataset from specified path.

**Parameters:**
- `data_path` (str): Path to PAMAP2 dataset directory

**Returns:**
- `data` (numpy.ndarray): IMU sensor data (N, 9)
- `labels` (numpy.ndarray): Activity labels (N,)

##### `load_uci_har(data_path)`
Load UCI-HAR dataset from specified path.

**Parameters:**
- `data_path` (str): Path to UCI-HAR dataset directory

**Returns:**
- `data` (numpy.ndarray): Feature vectors converted to pseudo-IMU (N, 6)
- `labels` (numpy.ndarray): Activity labels (N,)

##### `create_windows(data, labels, dataset_name)`
Create sliding windows from continuous data.

**Parameters:**
- `data` (numpy.ndarray): Continuous sensor data (N, features)
- `labels` (numpy.ndarray): Corresponding labels (N,)
- `dataset_name` (str): Dataset identifier

**Returns:**
- `windows` (numpy.ndarray): Windowed data (N_windows, window_size, features)
- `window_labels` (numpy.ndarray): Window labels (N_windows,)
- `domain_labels` (numpy.ndarray): Domain labels (N_windows,)

### `TCNVAETrainer`
**Location**: `training/train_tcn_vae.py`

Training class for TCN-VAE models with multi-objective optimization.

#### Constructor
```python
TCNVAETrainer(model, device, learning_rate=1e-3)
```

**Parameters:**
- `model` (TCNVAE): Model instance to train
- `device` (torch.device): Training device (CPU/CUDA)
- `learning_rate` (float): Optimizer learning rate (default: 1e-3)

#### Attributes
- `beta` (float): VAE KL divergence weight (default: 1.0)
- `lambda_act` (float): Activity classification weight (default: 1.0)
- `lambda_dom` (float): Domain adaptation weight (default: 0.1)

#### Methods

##### `train_epoch(train_loader, epoch)`
Train model for one epoch.

**Parameters:**
- `train_loader` (DataLoader): Training data loader
- `epoch` (int): Current epoch number

**Returns:**
- `avg_loss` (float): Average training loss for the epoch

##### `validate(val_loader)`
Evaluate model on validation set.

**Parameters:**
- `val_loader` (DataLoader): Validation data loader

**Returns:**
- `avg_val_loss` (float): Average validation loss
- `activity_accuracy` (float): Activity classification accuracy

##### `vae_loss(recon_x, x, mu, logvar)`
Compute VAE loss (reconstruction + KL divergence).

**Parameters:**
- `recon_x` (torch.Tensor): Reconstructed sequences
- `x` (torch.Tensor): Original input sequences
- `mu` (torch.Tensor): Latent mean vectors
- `logvar` (torch.Tensor): Latent log variance vectors

**Returns:**
- `total_vae_loss` (torch.Tensor): Combined VAE loss
- `recon_loss` (torch.Tensor): Reconstruction loss component
- `kl_loss` (torch.Tensor): KL divergence component

## Utility Functions

### Model Evaluation

#### `evaluate_trained_model()`
**Location**: `evaluation/evaluate_model.py`

Comprehensive evaluation of trained TCN-VAE model.

**Returns:**
- `accuracy` (float): Overall validation accuracy
- `report` (str): Detailed classification report

#### `export_for_edgeinfer()`
**Location**: `evaluation/evaluate_model.py`

Export trained model for EdgeInfer deployment.

**Returns:**
- `config` (dict): Model configuration for deployment

### Model Export

#### `BestModelExporter`
**Location**: `export_best_model.py`

ONNX export pipeline for trained models.

##### Constructor
```python
BestModelExporter(model_dir="models", export_dir="export")
```

**Parameters:**
- `model_dir` (str): Directory containing trained models
- `export_dir` (str): Output directory for exported files

##### Methods

##### `export_complete_pipeline()`
Run complete export pipeline from PyTorch to ONNX.

**Returns:**
- `success` (bool): True if export completed successfully

##### `load_best_model()`
Load best performing trained model.

**Returns:**
- `model` (TCNVAE): Loaded model instance

##### `export_to_onnx(encoder)`
Export encoder component to ONNX format.

**Parameters:**
- `encoder` (torch.nn.Module): Encoder module to export

**Returns:**
- `success` (bool): Export success status
- `onnx_path` (str): Path to exported ONNX file

##### `validate_export(pytorch_encoder, onnx_path)`
Validate ONNX export against PyTorch reference.

**Parameters:**
- `pytorch_encoder` (torch.nn.Module): Original PyTorch model
- `onnx_path` (str): Path to ONNX model file

**Returns:**
- `validation_passed` (bool): True if validation successful

## Configuration

### Training Configuration
**Location**: `configs/improved_config.py`

```python
IMPROVED_CONFIG = {
    # Training schedule
    "epochs": int,           # Number of training epochs
    "batch_size": int,       # Training batch size
    "learning_rate": float,  # Initial learning rate
    
    # Loss weights
    "beta": float,           # VAE KL divergence weight
    "lambda_act": float,     # Activity classification weight
    "lambda_dom": float,     # Domain adaptation weight
    
    # Regularization
    "dropout_rate": float,   # Dropout probability
    "weight_decay": float,   # L2 regularization weight
    "grad_clip_norm": float, # Gradient clipping threshold
    
    # Learning rate schedule
    "lr_scheduler": dict,    # Scheduler configuration
    
    # Early stopping
    "patience": int,         # Epochs to wait for improvement
    "min_delta": float,      # Minimum improvement threshold
}
```

## Data Formats

### Input Data Format
All models expect input data in the following format:
- **Shape**: (batch_size, sequence_length, input_dim)
- **Type**: torch.FloatTensor
- **Range**: Normalized to zero mean, unit variance per channel

### Label Formats
- **Activity Labels**: torch.LongTensor with values 0 to (num_classes-1)
- **Domain Labels**: torch.LongTensor with values 0, 1, 2 for datasets

### Dataset Windows
- **Window Size**: 100 timesteps (~1 second at 100Hz)
- **Overlap**: 50% (50 timestep stride)
- **Channels**: 9 (accelerometer x3, gyroscope x3, magnetometer x3)

## Error Handling

### Common Exceptions

#### `FileNotFoundError`
Raised when dataset files are not found at expected paths.
```python
try:
    data, labels = processor.load_pamap2(data_path)
except FileNotFoundError as e:
    print(f"Dataset not found: {e}")
```

#### `ValueError`
Raised for invalid configuration or data format issues.
```python
try:
    model = TCNVAE(input_dim=-1)  # Invalid input dimension
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

#### `RuntimeError`
Raised for CUDA memory issues or model loading problems.
```python
try:
    model = model.to(device)
except RuntimeError as e:
    print(f"CUDA error: {e}")
    device = torch.device('cpu')  # Fallback to CPU
```

## Performance Considerations

### Memory Usage
- **Model Size**: ~50MB for standard configuration
- **Training Memory**: 4-8GB GPU memory recommended
- **Batch Size**: Adjust based on available GPU memory

### Training Speed
- **Epoch Time**: ~40 seconds on RTX 2060
- **Convergence**: ~10 epochs for good results
- **Full Training**: 1-2 hours for 200 epochs

### Inference Speed
- **Single Sample**: <1ms on GPU
- **Batch Inference**: ~100 samples/second
- **Edge Deployment**: <50ms on Hailo-8 accelerator

## Version Compatibility

### Dependencies
- **PyTorch**: 2.0.0+
- **NumPy**: 1.21.0+
- **Pandas**: 1.3.0+
- **Scikit-learn**: 1.0.0+
- **ONNX**: 1.12.0+

### CUDA Support
- **CUDA**: 11.0+ recommended
- **cuDNN**: 8.0+ for optimal performance
- **GPU Memory**: 6GB+ recommended for training

### ONNX Export
- **Opset Version**: 11 (Hailo-8 compatible)
- **Static Shapes**: Required for edge deployment
- **Validation**: Automatic against PyTorch reference

---

This API reference covers all public interfaces in the TCN-VAE training pipeline. For implementation details, refer to the source code and module-specific documentation.