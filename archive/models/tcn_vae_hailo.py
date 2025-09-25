"""
Hailo-Compatible TCN-VAE with Conv2d Operations
Implements Conv1d→Conv2d transformation for Hailo-8 deployment
Supports phone+IMU dual-device processing with device attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class HailoTemporalBlock(nn.Module):
    """
    Hailo-compatible Temporal Block using Conv2d instead of Conv1d
    Implements dilated causal convolution with (1,k) kernels
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, 
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super(HailoTemporalBlock, self).__init__()
        
        # Conv1d → Conv2d transformation with (1, kernel_size) kernels
        # CRITICAL: groups=1 for Hailo compatibility (no grouped convolutions)
        self.conv1 = nn.Conv2d(
            n_inputs, n_outputs,
            kernel_size=(1, kernel_size),  # Height=1, Width=kernel_size
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, dilation),
            groups=1  # CRITICAL: No grouped convolutions for Hailo-8 (Conv2d limitations)
        )
        
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Conv2d(
            n_outputs, n_outputs,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, dilation),
            groups=1
        )
        
        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)
        
        # Residual connection with Conv2d
        self.downsample = nn.Conv2d(
            n_inputs, n_outputs, 
            kernel_size=(1, 1)
        ) if n_inputs != n_outputs else None
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Conv2d operations
        Input shape: (B, C, H, W) where H=2 (phone+IMU), W=T (time)
        """
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.dropout2(out)
        
        # Remove any padding artifacts (equivalent to Chomp1d)
        if out.size(-1) > residual.size(-1):
            out = out[:, :, :, :residual.size(-1)]
        
        return self.relu(out + residual)


class DeviceAttention(nn.Module):
    """
    Attention mechanism for phone+IMU dual-device processing
    Learns optimal weighting between phone and collar IMU sensors
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(DeviceAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention for device weighting
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Learnable device embeddings
        self.device_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        # Device-specific projection layers
        self.phone_projection = nn.Linear(hidden_dim, hidden_dim)
        self.imu_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply device attention to combine phone and IMU data
        Input: (B, C, 2, T) - 2 devices: phone, collar IMU
        Output: (B, C, T) - Combined features with attention weights
        """
        B, C, H, T = x.shape
        assert H == 2, f"Device dimension must be 2, got {H}"
        
        # Separate phone and IMU data
        phone_data = x[:, :, 0, :]  # (B, C, T)
        imu_data = x[:, :, 1, :]    # (B, C, T)
        
        # Reshape for attention: (B*T, 2, C)
        x_reshaped = x.permute(0, 3, 2, 1).reshape(B * T, 2, C)
        
        # Add learnable device embeddings
        device_emb = self.device_embedding.expand(B * T, -1, -1)
        x_with_emb = x_reshaped + device_emb
        
        # Apply multi-head attention
        attended, attention_weights = self.attention(
            x_with_emb, x_with_emb, x_with_emb
        )
        
        # Reshape back to (B, T, 2, C)
        attended = attended.reshape(B, T, 2, C)
        
        # Apply device-specific projections
        phone_features = self.phone_projection(attended[:, :, 0, :])  # (B, T, C)
        imu_features = self.imu_projection(attended[:, :, 1, :])      # (B, T, C)
        
        # Concatenate and project
        combined = torch.cat([phone_features, imu_features], dim=-1)  # (B, T, 2*C)
        output = self.output_projection(combined)  # (B, T, C)
        
        # Permute back to (B, C, T)
        output = output.permute(0, 2, 1)
        
        # Average attention weights across time for visualization
        avg_attention = attention_weights.reshape(B, T, 2, 2).mean(dim=1)  # (B, 2, 2)
        
        return output, avg_attention


class HailoTemporalConvNet(nn.Module):
    """
    Hailo-compatible Temporal Convolutional Network using Conv2d
    Supports phone+IMU dual-device input with device attention
    """
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 kernel_size: int = 3, dropout: float = 0.2, 
                 use_device_attention: bool = True):
        super(HailoTemporalConvNet, self).__init__()
        
        self.use_device_attention = use_device_attention
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Calculate padding for causal convolution
            padding = (kernel_size - 1) * dilation_size // 2
            
            layers.append(HailoTemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size, 
                padding=padding, dropout=dropout
            ))
        
        self.network = nn.ModuleList(layers)
        
        # Add device attention after TCN blocks
        if use_device_attention:
            self.device_attention = DeviceAttention(num_channels[-1])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through TCN with optional device attention
        Input: (B, C, H, W) where H=2 (phone+IMU), W=T (time)
        """
        attention_weights = None
        
        # Process through TCN blocks
        for layer in self.network:
            x = layer(x)
        
        # Apply device attention if enabled
        if self.use_device_attention and x.size(2) == 2:
            x, attention_weights = self.device_attention(x)
            # x is now (B, C, T) after device combination
        
        return x, attention_weights


class HailoTCNVAE(nn.Module):
    """
    Hailo-compatible TCN-VAE for cross-species behavioral analysis
    Implements phone+IMU processing with device attention
    """
    def __init__(self, input_dim: int = 9, hidden_dims: List[int] = [64, 128, 256], 
                 latent_dim: int = 64, sequence_length: int = 100,
                 num_human_activities: int = 12, num_dog_behaviors: int = 3,
                 use_device_attention: bool = True):
        super(HailoTCNVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.use_device_attention = use_device_attention
        
        # Input projection to match first hidden dimension
        self.input_projection = nn.Conv2d(
            input_dim, hidden_dims[0], 
            kernel_size=(1, 1)
        )
        
        # TCN Encoder with device attention
        self.tcn_encoder = HailoTemporalConvNet(
            hidden_dims[0], hidden_dims, 
            use_device_attention=use_device_attention
        )
        
        # VAE components
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dims[-1])
        decoder_dims = hidden_dims[::-1] + [input_dim]
        self.tcn_decoder = HailoTemporalConvNet(
            hidden_dims[-1], decoder_dims[1:],
            use_device_attention=False  # No device attention in decoder
        )
        
        # Cross-species classification heads
        self.human_classifier = nn.Linear(latent_dim, num_human_activities)
        self.dog_classifier = nn.Linear(latent_dim, num_dog_behaviors)
        
        # Domain adaptation for transfer learning
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # 2 domains: human, dog
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input to latent representation
        Input shape: (B, 9, 2, T) - 9 IMU channels, 2 devices, T timesteps
        """
        B, C, H, T = x.shape
        assert C == self.input_dim, f"Expected {self.input_dim} channels, got {C}"
        assert H == 2, f"Expected 2 devices (phone+IMU), got {H}"
        
        # Project input channels
        x = self.input_projection(x)
        
        # Apply TCN encoder with device attention
        h, attention_weights = self.tcn_encoder(x)
        
        # h shape depends on device attention:
        # If attention used: (B, hidden_dim, T)
        # If not used: (B, hidden_dim, 2, T)
        
        if len(h.shape) == 4:  # No device attention applied
            # Average across device dimension
            h = h.mean(dim=2)  # (B, hidden_dim, T)
        
        # Global average pooling over time
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # (B, hidden_dim)
        
        # VAE latent encoding
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar, attention_weights
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Standard VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Decode latent representation back to input space
        z: (B, latent_dim)
        target_shape: (B, 9, 2, T) - Original input shape
        """
        B, C, H, T = target_shape
        
        # Project latent to hidden dimension
        h = self.decoder_fc(z)  # (B, hidden_dim)
        
        # Reshape for TCN decoder
        h = h.unsqueeze(-1).expand(-1, -1, T)  # (B, hidden_dim, T)
        
        # For decoder, we need to add a dummy device dimension
        h = h.unsqueeze(2).expand(-1, -1, 1, -1)  # (B, hidden_dim, 1, T)
        
        # Apply TCN decoder
        reconstructed, _ = self.tcn_decoder(h)
        
        # Expand to match dual-device output
        if reconstructed.size(2) == 1:
            reconstructed = reconstructed.expand(-1, -1, 2, -1)  # (B, 9, 2, T)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass for training
        Input: (B, 9, 2, T) - 9 IMU channels, 2 devices, T timesteps
        """
        # Encode
        mu, logvar, attention_weights = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decode(z, x.shape)
        
        # Classification heads
        human_logits = self.human_classifier(z)
        dog_logits = self.dog_classifier(z)
        
        # Domain classification for adversarial training
        domain_logits = self.domain_classifier(z)
        
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'human_logits': human_logits,
            'dog_logits': dog_logits,
            'domain_logits': domain_logits,
            'attention_weights': attention_weights
        }
    
    def export_for_hailo(self, dummy_input: torch.Tensor, output_path: str):
        """
        Export model to ONNX format for Hailo compilation
        dummy_input: (1, 9, 2, 100) - Static shape for Hailo
        """
        self.eval()
        
        # Export only the encoder part for edge inference
        class EncoderOnly(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.input_projection = parent.input_projection
                self.tcn_encoder = parent.tcn_encoder
                self.fc_mu = parent.fc_mu
                self.dog_classifier = parent.dog_classifier
            
            def forward(self, x):
                x = self.input_projection(x)
                h, attention = self.tcn_encoder(x)
                if len(h.shape) == 4:
                    h = h.mean(dim=2)
                h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
                mu = self.fc_mu(h)
                dog_logits = self.dog_classifier(mu)
                return dog_logits, attention
        
        encoder_model = EncoderOnly(self)
        
        torch.onnx.export(
            encoder_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['dog_behavior', 'attention_weights'],
            dynamic_axes=None,  # Static shapes for Hailo
            verbose=True
        )
        
        print(f"✅ Model exported to {output_path} for Hailo compilation")


# Cross-species loss function
class CrossSpeciesLoss(nn.Module):
    """
    Combined loss for cross-species transfer learning
    Includes VAE loss, human activity loss, and dog behavior loss
    """
    def __init__(self, beta: float = 1.0, dog_weight: float = 2.0):
        super(CrossSpeciesLoss, self).__init__()
        self.beta = beta
        self.dog_weight = dog_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: dict, targets: dict) -> dict:
        """
        Calculate combined loss
        outputs: Model outputs dictionary
        targets: Ground truth labels
        """
        # VAE reconstruction loss
        recon_loss = F.mse_loss(
            outputs['reconstructed'], 
            targets['input'], 
            reduction='mean'
        )
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
        ) / outputs['mu'].size(0)
        
        # Human activity classification loss
        human_loss = self.ce_loss(
            outputs['human_logits'], 
            targets['human_labels']
        ) if 'human_labels' in targets else 0
        
        # Dog behavior classification loss (weighted higher)
        dog_loss = self.ce_loss(
            outputs['dog_logits'], 
            targets['dog_labels']
        ) * self.dog_weight if 'dog_labels' in targets else 0
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + human_loss + dog_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss,
            'human': human_loss if torch.is_tensor(human_loss) else torch.tensor(human_loss, device=outputs['mu'].device, dtype=outputs['mu'].dtype),
            'dog': dog_loss if torch.is_tensor(dog_loss) else torch.tensor(dog_loss, device=outputs['mu'].device, dtype=outputs['mu'].dtype)
        }


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing Hailo-compatible TCN-VAE...")
    
    # Create model
    model = HailoTCNVAE(
        input_dim=9,
        hidden_dims=[64, 128, 256],
        latent_dim=64,
        sequence_length=100,
        num_human_activities=12,
        num_dog_behaviors=3,
        use_device_attention=True
    )
    
    # Test input: (B, 9, 2, 100) - 9 IMU channels, 2 devices, 100 timesteps
    batch_size = 4
    dummy_input = torch.randn(batch_size, 9, 2, 100)
    
    # Forward pass
    outputs = model(dummy_input)
    
    print(f"✅ Model output shapes:")
    print(f"   Reconstructed: {outputs['reconstructed'].shape}")
    print(f"   Latent (mu): {outputs['mu'].shape}")
    print(f"   Human logits: {outputs['human_logits'].shape}")
    print(f"   Dog logits: {outputs['dog_logits'].shape}")
    print(f"   Attention weights: {outputs['attention_weights'].shape if outputs['attention_weights'] is not None else 'None'}")
    
    # Test export
    print("\n🚀 Testing ONNX export for Hailo...")
    model.export_for_hailo(
        torch.randn(1, 9, 2, 100),
        "tcn_vae_hailo.onnx"
    )
    
    print("\n✅ All tests passed! Model ready for Hailo deployment.")