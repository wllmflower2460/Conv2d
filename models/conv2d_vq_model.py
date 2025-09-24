"""
Conv2d-VQ Model: Integrates Conv2d encoder with Vector Quantization
Building on existing TCN-VAE architecture with VQ replacing VAE latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

# Import existing components
try:
    from .vq_ema_2d import VectorQuantizerEMA2D, VQHead2D
    from .tcn_vae import TemporalConvNet, TemporalBlock
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from vq_ema_2d import VectorQuantizerEMA2D, VQHead2D
    from tcn_vae import TemporalConvNet, TemporalBlock


class Conv2dEncoder(nn.Module):
    """
    Conv2d-based encoder for behavioral feature extraction
    Transforms (B, 9, 2, 100) IMU data to (B, code_dim, H_out, T_out)
    """
    
    def __init__(
        self,
        input_channels: int = 9,  # 9-axis IMU
        input_height: int = 2,    # Human + Dog devices
        hidden_channels: List[int] = [64, 128, 256],
        code_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        
        # Initial Conv2d to process spatial-temporal features
        self.initial_conv = nn.Conv2d(
            input_channels, 
            hidden_channels[0],
            kernel_size=(input_height, 5),  # Full height, 50ms temporal window
            padding=(0, 2)
        )
        
        # Temporal processing layers (keeping height=1 after initial conv)
        layers = []
        in_channels = hidden_channels[0]
        
        for out_channels in hidden_channels[1:]:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=(1, kernel_size),
                        padding=(0, kernel_size//2),
                        dilation=(1, 1)
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(dropout)
                )
            )
            in_channels = out_channels
        
        self.temporal_layers = nn.ModuleList(layers)
        
        # Project to code dimension
        self.to_codes = nn.Conv2d(
            hidden_channels[-1], 
            code_dim,
            kernel_size=1  # 1x1 conv for channel projection
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 9, 2, 100) - Batch, Channels, Devices, Time
        Returns:
            z: (B, code_dim, 1, T_out) - Ready for VQ
        """
        # Initial spatial-temporal processing
        z = self.initial_conv(x)  # (B, 64, 1, ~100)
        
        # Temporal feature extraction
        for layer in self.temporal_layers:
            z = layer(z)
        
        # Project to code dimension
        z = self.to_codes(z)  # (B, code_dim, 1, T_out)
        
        return z


class Conv2dDecoder(nn.Module):
    """
    Decoder to reconstruct from quantized codes
    Mirrors encoder architecture
    """
    
    def __init__(
        self,
        code_dim: int = 64,
        hidden_channels: List[int] = [256, 128, 64],
        output_channels: int = 9,
        output_height: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Project from codes
        self.from_codes = nn.Conv2d(code_dim, hidden_channels[0], kernel_size=1)
        
        # Temporal upsampling layers
        layers = []
        in_channels = hidden_channels[0]
        
        for out_channels in hidden_channels[1:]:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=(1, kernel_size),
                        padding=(0, kernel_size//2)
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(dropout)
                )
            )
            in_channels = out_channels
        
        self.temporal_layers = nn.ModuleList(layers)
        
        # Final reconstruction with spatial upsampling
        self.final_conv = nn.ConvTranspose2d(
            hidden_channels[-1],
            output_channels,
            kernel_size=(output_height, 5),
            padding=(0, 2)
        )
        
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: (B, code_dim, 1, T) - Quantized features
        Returns:
            x_recon: (B, 9, 2, 100) - Reconstructed IMU data
        """
        z = self.from_codes(z_q)
        
        for layer in self.temporal_layers:
            z = layer(z)
        
        x_recon = self.final_conv(z)  # (B, 9, 2, ~100)
        
        return x_recon


class Conv2dVQModel(nn.Module):
    """
    Complete Conv2d-VQ architecture for behavioral analysis
    Replaces VAE with VQ for discrete representation learning
    """
    
    def __init__(
        self,
        input_channels: int = 9,
        input_height: int = 2,
        num_codes: int = 256,  # Reduced from 512 per advisor recommendation
        code_dim: int = 64,
        hidden_channels: List[int] = [64, 128, 256],
        num_activities: int = 12,
        dropout: float = 0.2,
        vq_decay: float = 0.99,
        commitment_cost: float = 0.4  # Increased from 0.25 per advisor recommendation
    ):
        super().__init__()
        
        # Encoder
        self.encoder = Conv2dEncoder(
            input_channels=input_channels,
            input_height=input_height,
            hidden_channels=hidden_channels,
            code_dim=code_dim,
            dropout=dropout
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizerEMA2D(
            num_codes=num_codes,
            code_dim=code_dim,
            decay=vq_decay,
            commitment_cost=commitment_cost
        )
        
        # Decoder
        self.decoder = Conv2dDecoder(
            code_dim=code_dim,
            hidden_channels=hidden_channels[::-1],  # Reverse for decoder
            output_channels=input_channels,
            output_height=input_height,
            dropout=dropout
        )
        
        # Activity classifier head (operates on quantized features)
        self.activity_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),
            nn.Linear(code_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_activities)
        )
        
        # Behavioral state predictor (for future HDP-HSMM integration)
        self.state_predictor = nn.Sequential(
            nn.Conv2d(code_dim, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)  # 16 potential behavioral states
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 9, 2, 100) - IMU data
            return_all: Whether to return intermediate outputs
        
        Returns:
            Dictionary containing:
                - reconstructed: Reconstructed input
                - activity_logits: Activity classification
                - vq_loss: Vector quantization loss
                - perplexity: Codebook utilization metric
                - indices: Quantized code indices
        """
        # Encode
        z_e = self.encoder(x)  # (B, code_dim, 1, T_out)
        
        # Vector Quantize
        z_q, loss_dict, vq_info = self.vq(z_e)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        # Activity classification
        activity_logits = self.activity_classifier(z_q)
        
        # Behavioral states (for future use)
        states = self.state_predictor(z_q)
        
        # Prepare output
        outputs = {
            'reconstructed': x_recon,
            'activity_logits': activity_logits,
            'vq_loss': loss_dict['vq'],
            'commitment_loss': loss_dict['commitment'],
            'perplexity': vq_info['perplexity'],
            'codebook_usage': vq_info['usage'],
            'indices': vq_info['indices'],
            'behavioral_states': states
        }
        
        if return_all:
            outputs.update({
                'encoded': z_e,
                'quantized': z_q,
                'vq_info': vq_info,
                'loss_dict': loss_dict
            })
        
        return outputs
    
    def encode_only(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get encoded and quantized representations"""
        z_e = self.encoder(x)
        z_q, _, info = self.vq(z_e)
        return z_e, z_q, info['indices']
    
    def get_codebook_stats(self) -> Dict:
        """Get statistics about codebook usage"""
        with torch.no_grad():
            # Get current codebook
            codebook = self.vq.embedding  # (num_codes, code_dim)
            
            # Calculate pairwise distances between codes
            distances = torch.cdist(codebook, codebook)
            
            # Exclude self-distances
            mask = torch.eye(self.vq.num_codes, device=codebook.device)
            distances = distances + mask * 1e10
            
            min_distances = distances.min(dim=1)[0]
            
            return {
                'num_codes': self.vq.num_codes,
                'code_dim': self.vq.code_dim,
                'mean_min_distance': min_distances.mean().item(),
                'std_min_distance': min_distances.std().item(),
                'ema_cluster_sizes': self.vq.ema_cluster_size.tolist()[:20]  # First 20
            }


def test_conv2d_vq_model():
    """Test the integrated Conv2d-VQ model"""
    print("Testing Conv2d-VQ Model...")
    
    # Test parameters
    B, C, H, T = 4, 9, 2, 100  # Batch, IMU channels, Devices, Time
    
    # Initialize model
    model = Conv2dVQModel(
        input_channels=C,
        input_height=H,
        num_codes=512,
        code_dim=64,
        num_activities=12
    )
    
    # Test input
    x = torch.randn(B, C, H, T)
    
    # Forward pass
    model.train()
    outputs = model(x, return_all=True)
    
    # Check outputs
    assert outputs['reconstructed'].shape == x.shape, \
        f"Reconstruction shape mismatch: {outputs['reconstructed'].shape} vs {x.shape}"
    assert outputs['activity_logits'].shape == (B, 12), \
        f"Activity logits shape: {outputs['activity_logits'].shape}"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Reconstructed shape: {outputs['reconstructed'].shape}")
    print(f"✓ Activity logits shape: {outputs['activity_logits'].shape}")
    print(f"✓ VQ Loss: {outputs['vq_loss']:.4f}")
    print(f"✓ Perplexity: {outputs['perplexity']:.2f}")
    print(f"✓ Codebook usage: {outputs['codebook_usage']:.2%}")
    
    # Test gradient flow
    total_loss = (
        F.mse_loss(outputs['reconstructed'], x) +  # Reconstruction
        outputs['vq_loss'] +  # VQ loss
        F.cross_entropy(outputs['activity_logits'], torch.randint(0, 12, (B,)))  # Classification
    )
    
    total_loss.backward()
    print("✓ Gradient flow verified")
    
    # Test encode-only mode
    z_e, z_q, indices = model.encode_only(x)
    print(f"✓ Encode-only: z_e shape={z_e.shape}, indices shape={indices.shape}")
    
    # Get codebook stats
    stats = model.get_codebook_stats()
    print(f"✓ Codebook stats: {stats['num_codes']} codes, "
          f"mean min distance={stats['mean_min_distance']:.3f}")
    
    print("\nConv2d-VQ Model tests passed!")
    return model


if __name__ == "__main__":
    model = test_conv2d_vq_model()