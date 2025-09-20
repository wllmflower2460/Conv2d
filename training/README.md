# Training Module Documentation

This module contains training scripts and utilities for the TCN-VAE model.

## Files Overview

### `train_tcn_vae.py`
Basic training script with the core `TCNVAETrainer` class and training loop.

### `train_extended.py`
Extended training utilities and optimizations (if present).

## Classes

### `TCNVAETrainer`
Main trainer class that handles the complete training process for the TCN-VAE model.

```python
class TCNVAETrainer:
    def __init__(self, model, device, learning_rate=1e-3)
```

**Parameters:**
- `model` (TCNVAE): The model instance to train
- `device` (torch.device): CUDA or CPU device
- `learning_rate` (float): Adam optimizer learning rate

## Training Architecture

### Multi-Objective Loss Function
The trainer optimizes multiple objectives simultaneously:

```python
total_loss = vae_loss + λ_act * activity_loss + λ_dom * domain_loss
```

**Components:**
1. **VAE Loss**: Reconstruction + KL divergence
2. **Activity Loss**: Cross-entropy for supervised classification
3. **Domain Loss**: Adversarial loss for domain invariance

### Loss Implementation

#### VAE Loss
```python
def vae_loss(self, recon_x, x, mu, logvar):
    recon_loss = self.reconstruction_loss(recon_x, x)  # MSE
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + self.beta * kl_loss, recon_loss, kl_loss
```

**Components:**
- **Reconstruction Loss**: MSE between input and reconstructed sequences
- **KL Divergence**: Regularizes latent space to unit Gaussian
- **Beta Weighting**: Controls reconstruction vs. regularization trade-off

#### Progressive Adversarial Training
```python
# Progressive domain adaptation strength
p = float(batch_idx + epoch * len(train_loader)) / (10 * len(train_loader))
alpha = 2. / (1. + np.exp(-10 * p)) - 1
```

The domain adaptation strength gradually increases during training to allow the encoder to first learn good representations before applying domain adversarial training.

### Training Configuration

#### Default Loss Weights
```python
self.beta = 1.0          # VAE KL weight
self.lambda_act = 1.0    # Activity classification weight  
self.lambda_dom = 0.1    # Domain adaptation weight
```

#### Optimized Loss Weights (from overnight training)
```python
self.beta = 0.4          # Reduced for stability
self.lambda_act = 2.5    # Increased activity focus
self.lambda_dom = 0.05   # Minimal domain confusion
```

## Training Loop

### `train_epoch()`
Performs one complete training epoch.

```python
def train_epoch(self, train_loader, epoch):
    # Returns: average_loss
```

**Process:**
1. Set model to training mode
2. Iterate through batches
3. Apply progressive adversarial scaling
4. Forward pass through model
5. Compute multi-objective loss
6. Backward pass and optimization
7. Log progress every 100 batches

### `validate()`
Evaluates model on validation set.

```python
def validate(self, val_loader):
    # Returns: avg_val_loss, activity_accuracy
```

**Process:**
1. Set model to evaluation mode
2. Disable gradients for efficiency
3. Forward pass on validation data
4. Compute validation loss and accuracy
5. Return metrics for monitoring

## Training Features

### Gradient Management
- **Gradient Clipping**: Prevents gradient explosion during training
- **Weight Decay**: L2 regularization in optimizer
- **Learning Rate Scheduling**: Cosine annealing with warm restarts

### Monitoring and Logging
- **Batch-level Logging**: Progress every 100 batches
- **Epoch-level Metrics**: Loss and accuracy tracking
- **Best Model Saving**: Automatic checkpoint of best validation performance

### Early Stopping
```python
if val_accuracy > best_val_accuracy:
    best_val_accuracy = val_accuracy
    torch.save(model.state_dict(), 'models/best_tcn_vae.pth')
```

## Usage Examples

### Basic Training
```python
from training.train_tcn_vae import TCNVAETrainer
from models.tcn_vae import TCNVAE

# Initialize model and trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCNVAE(input_dim=9, latent_dim=64, num_activities=13)
trainer = TCNVAETrainer(model, device, learning_rate=1e-3)

# Training loop
for epoch in range(50):
    train_loss = trainer.train_epoch(train_loader, epoch)
    val_loss, val_accuracy = trainer.validate(val_loader)
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.4f}')
```

### Custom Loss Weights
```python
trainer = TCNVAETrainer(model, device)
trainer.beta = 0.5        # Adjust VAE weight
trainer.lambda_act = 2.0  # Increase classification focus
trainer.lambda_dom = 0.01 # Reduce domain adversarial weight
```

### Advanced Training Configuration
```python
# Custom optimizer settings
trainer.optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Learning rate scheduler
trainer.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    trainer.optimizer, T_0=20, T_mult=2
)
```

## Training Strategies

### Multi-Dataset Training
The trainer handles data from multiple datasets with domain labels:

```python
# Data format: (data, activity_labels, domain_labels)
for batch_idx, (data, activity_labels, domain_labels) in enumerate(train_loader):
    data = data.to(device).float()
    activity_labels = activity_labels.to(device).long()
    domain_labels = domain_labels.to(device).long()
```

### Progressive Learning
1. **Early Epochs**: Focus on reconstruction and basic activity recognition
2. **Middle Epochs**: Gradually introduce domain adversarial training
3. **Late Epochs**: Full multi-objective training with all loss components

### Regularization Techniques
- **Dropout**: Applied in TCN blocks (0.2-0.3 rate)
- **Weight Normalization**: In convolutional layers
- **Label Smoothing**: In activity classification loss
- **Data Augmentation**: Small noise injection to input data

## Performance Monitoring

### Training Metrics
- **Training Loss**: Combined multi-objective loss
- **Reconstruction Quality**: MSE between input and output
- **KL Divergence**: Latent space regularization
- **Activity Accuracy**: Classification performance
- **Domain Confusion**: Adversarial training effectiveness

### Validation Metrics
- **Validation Loss**: Held-out performance
- **Activity Accuracy**: Primary performance metric
- **Cross-Domain Performance**: Generalization across datasets

### Logging Format
```
Epoch: 10, Batch: 100, VAE Loss: 2.3456, Activity Loss: 0.8765, Domain Loss: 1.2345
Epoch 10: Train Loss: 2.1234, Val Loss: 2.3456, Val Accuracy: 0.7213
```

## Advanced Training Features

### Curriculum Learning
The training can be enhanced with curriculum learning:
- Start with easier samples (single-dataset)
- Gradually introduce cross-dataset complexity
- Progressive difficulty in activity recognition

### Transfer Learning
```python
# Load pretrained encoder
pretrained_model = torch.load('pretrained_tcn_encoder.pth')
model.tcn_encoder.load_state_dict(pretrained_model)

# Freeze encoder initially
for param in model.tcn_encoder.parameters():
    param.requires_grad = False
```

### Hyperparameter Optimization
Key hyperparameters for tuning:
- **Learning Rate**: 1e-4 to 1e-3 range
- **Beta (KL Weight)**: 0.1 to 1.0
- **Activity Weight**: 1.0 to 5.0
- **Domain Weight**: 0.01 to 0.1
- **Batch Size**: 32 to 128
- **Dropout Rate**: 0.1 to 0.5

## Integration with Main Training Scripts

### `train_overnight.py`
The overnight training script uses an enhanced version of the trainer:

```python
class OvernightTrainer(TCNVAETrainer):
    # Enhanced with:
    # - Better hyperparameters
    # - Detailed logging
    # - Checkpoint management
    # - ETA estimation
```

### Key Improvements in Overnight Training
1. **Conservative Learning Rate**: 3e-4 for stability
2. **Better Loss Balancing**: β=0.4, λ_act=2.5, λ_dom=0.05
3. **Gradient Clipping**: 0.8 max norm
4. **Label Smoothing**: 0.1 for activity classification
5. **Extended Patience**: 50 epochs for overnight runs

## Troubleshooting

### Common Training Issues

#### Gradient Explosion
**Symptoms**: Loss suddenly increases to very large values
**Solutions**: 
- Reduce learning rate
- Increase gradient clipping
- Reduce beta weight

#### Mode Collapse
**Symptoms**: KL loss goes to zero, poor reconstruction
**Solutions**:
- Increase beta weight gradually
- Use beta annealing schedule
- Check latent dimension size

#### Poor Activity Classification
**Symptoms**: Activity accuracy plateaus at low values
**Solutions**:
- Increase lambda_act weight
- Check data preprocessing
- Verify label encoding
- Add more supervised data

#### Domain Overfitting
**Symptoms**: Good single-dataset performance, poor cross-dataset
**Solutions**:
- Increase lambda_dom weight
- Check domain label distribution
- Balance dataset contributions

### Training Monitoring
```python
# Check for training issues
if torch.isnan(total_loss):
    print("NaN loss detected - reducing learning rate")
    
if val_accuracy < 0.1:
    print("Very low accuracy - check data preprocessing")
    
if kl_loss < 0.01:
    print("KL collapse - increase beta weight")
```

## Best Practices

### Training Setup
1. **Start Simple**: Begin with single dataset, single objective
2. **Gradual Complexity**: Add multi-objective losses progressively
3. **Monitor Closely**: Check all loss components, not just total loss
4. **Save Frequently**: Checkpoint every few epochs
5. **Validate Often**: Check validation performance regularly

### Hyperparameter Strategy
1. **Grid Search**: For key hyperparameters (lr, beta, lambda_act)
2. **Random Search**: For broader exploration
3. **Bayesian Optimization**: For efficient hyperparameter optimization
4. **Learning Rate Finding**: Use learning rate finder for optimal range

### Data Strategy
1. **Balanced Sampling**: Ensure equal representation of activities
2. **Domain Balance**: Balance datasets in each batch
3. **Augmentation**: Add sensor noise and temporal variations
4. **Validation Split**: Stratified split maintaining class balance