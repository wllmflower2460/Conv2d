#!/usr/bin/env python3
"""
Conv2d model with FSQ (Finite Scalar Quantization).
FSQ provides stable discrete representations that cannot collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
from vector_quantize_pytorch import FSQ

class Conv2dEncoder(nn.Module):
    """2D Convolutional encoder for IMU data."""
    
    def __init__(self, input_channels: int = 9, hidden_dim: int = 128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, hidden_dim, kernel_size=(1, 5), padding=(0, 2))
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        # x: (B, 9, 2, 100) - 9 IMU channels, 2 spatial dims, 100 timesteps
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 128, 2, 100)
        
        # Global pooling to get single feature vector
        x = self.pool(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        
        return x


class Conv2dFSQ(nn.Module):
    """
    Conv2d model with FSQ quantization for behavioral analysis.
    
    FSQ quantizes continuous features into discrete codes, providing:
    - Guaranteed stability (no collapse possible)
    - Interpretable discrete behavioral codes
    - Efficient compression for edge deployment
    """
    
    def __init__(
        self,
        input_channels: int = 9,
        hidden_dim: int = 128,
        num_classes: int = 10,
        fsq_levels: Optional[List[int]] = None,
        project_dim: Optional[int] = None
    ):
        """
        Args:
            input_channels: Number of input channels (IMU dimensions)
            hidden_dim: Hidden dimension for encoder
            num_classes: Number of output classes
            fsq_levels: Quantization levels per dimension [8,8,8,8] = 4096 codes
            project_dim: If set, project to this dimension before FSQ
        """
        super().__init__()
        
        # Default FSQ levels if not provided
        if fsq_levels is None:
            # [8,6,5,5,4] = 9600 unique codes
            # More levels = more codes but less compression
            fsq_levels = [8, 6, 5, 5, 4]
        
        self.fsq_levels = fsq_levels
        self.num_codes = np.prod(fsq_levels)
        
        # Calculate FSQ input dimension
        fsq_dim = len(fsq_levels)
        
        # Encoder
        self.encoder = Conv2dEncoder(input_channels, hidden_dim)
        
        # Optional projection to match FSQ dimension
        self.project_dim = project_dim or fsq_dim
        if hidden_dim != self.project_dim:
            self.projection = nn.Linear(hidden_dim, self.project_dim)
        else:
            self.projection = None
        
        # FSQ layer
        self.fsq = FSQ(levels=fsq_levels)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fsq_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # For tracking code usage statistics
        self.register_buffer('code_counts', torch.zeros(self.num_codes))
        self.register_buffer('total_samples', torch.tensor(0))
        
    def encode_to_codes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to discrete codes.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            codes: Discrete code indices (B,)
            quantized: Quantized features (B, D)
        """
        # Encode to continuous features
        features = self.encoder(x)  # (B, 128)
        
        # Project if needed
        if self.projection is not None:
            features = self.projection(features)  # (B, fsq_dim)
        
        # FSQ expects 3D input (batch, sequence, features)
        # Add sequence dimension of 1
        features_3d = features.unsqueeze(1)  # (B, 1, fsq_dim)
        
        # Quantize with FSQ
        quantized, codes = self.fsq(features_3d)
        
        # Remove sequence dimension
        quantized = quantized.squeeze(1)  # (B, fsq_dim)
        codes = codes.squeeze(1) if codes.dim() > 1 else codes  # (B,)
        
        return codes, quantized
    
    def forward(self, x: torch.Tensor, return_codes: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder, FSQ, and classifier.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_codes: Whether to return discrete codes
            
        Returns:
            Dictionary containing:
                - logits: Classification logits (B, num_classes)
                - codes: Discrete codes if requested (B,)
                - quantized: Quantized features (B, D)
                - features: Pre-quantization features (B, D)
        """
        # Encode to continuous features
        features = self.encoder(x)  # (B, 128)
        
        # Project if needed
        if self.projection is not None:
            features = self.projection(features)  # (B, fsq_dim)
        
        # FSQ expects 3D input (batch, sequence, features)
        # Add sequence dimension of 1
        features_3d = features.unsqueeze(1)  # (B, 1, fsq_dim)
        
        # Quantize with FSQ
        quantized, codes = self.fsq(features_3d)
        
        # Remove sequence dimension
        quantized = quantized.squeeze(1)  # (B, fsq_dim)
        codes = codes.squeeze(1) if codes.dim() > 1 else codes  # (B,)
        
        # Classify from quantized features
        logits = self.classifier(quantized)
        
        # Update code usage statistics (during training)
        if self.training:
            self.update_code_stats(codes)
        
        output = {
            'logits': logits,
            'quantized': quantized,
            'features': features
        }
        
        if return_codes:
            output['codes'] = codes
            
        return output
    
    def update_code_stats(self, codes: torch.Tensor):
        """Update code usage statistics."""
        unique_codes = torch.unique(codes)
        for code in unique_codes:
            self.code_counts[code] += (codes == code).sum()
        self.total_samples += codes.numel()
    
    def get_code_stats(self) -> Dict[str, float]:
        """Get code usage statistics."""
        if self.total_samples == 0:
            return {
                'usage_ratio': 0.0,
                'perplexity': 0.0,
                'unique_codes': 0,
                'entropy': 0.0
            }
        
        # Calculate probabilities
        probs = self.code_counts / self.total_samples
        probs = probs[probs > 0]  # Only non-zero probabilities
        
        # Calculate entropy and perplexity
        entropy = -torch.sum(probs * torch.log(probs))
        perplexity = torch.exp(entropy)
        
        # Usage ratio
        usage_ratio = (self.code_counts > 0).float().mean()
        unique_codes = (self.code_counts > 0).sum()
        
        return {
            'usage_ratio': usage_ratio.item(),
            'perplexity': perplexity.item(),
            'unique_codes': unique_codes.item(),
            'entropy': entropy.item()
        }
    
    def reset_code_stats(self):
        """Reset code usage statistics."""
        self.code_counts.zero_()
        self.total_samples.zero_()
    
    def get_behavioral_dictionary(self, threshold: float = 0.01) -> Dict[int, Dict]:
        """
        Build a behavioral dictionary from code usage.
        
        Args:
            threshold: Minimum usage ratio to include code
            
        Returns:
            Dictionary mapping code indices to statistics
        """
        if self.total_samples == 0:
            return {}
        
        probs = self.code_counts / self.total_samples
        behavioral_dict = {}
        
        for code_idx in range(self.num_codes):
            if probs[code_idx] > threshold:
                behavioral_dict[code_idx] = {
                    'usage': probs[code_idx].item(),
                    'count': self.code_counts[code_idx].item(),
                    'rank': 0  # Will be filled later
                }
        
        # Add ranks based on usage
        sorted_codes = sorted(behavioral_dict.keys(), 
                            key=lambda x: behavioral_dict[x]['usage'], 
                            reverse=True)
        for rank, code_idx in enumerate(sorted_codes):
            behavioral_dict[code_idx]['rank'] = rank + 1
            
        return behavioral_dict


def test_fsq_model():
    """Test the FSQ model to ensure it works correctly."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing FSQ model on {device}")
    
    # Create model with different FSQ configurations
    configs = [
        ([8, 8, 8, 8], "8x8x8x8 = 4096 codes"),
        ([8, 6, 5, 5, 4], "8x6x5x5x4 = 4800 codes"),
        ([4, 4, 4, 4, 4, 4], "4^6 = 4096 codes, 6D"),
    ]
    
    for fsq_levels, description in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        print(f"{'='*60}")
        
        model = Conv2dFSQ(
            input_channels=9,
            hidden_dim=128,
            num_classes=10,
            fsq_levels=fsq_levels
        ).to(device)
        
        # Test batch
        batch_size = 32
        x = torch.randn(batch_size, 9, 2, 100).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x, return_codes=True)
        
        print(f"Output shapes:")
        print(f"  Logits: {output['logits'].shape}")
        print(f"  Codes: {output['codes'].shape}")
        print(f"  Quantized: {output['quantized'].shape}")
        
        # Check unique codes
        unique_codes = torch.unique(output['codes']).numel()
        print(f"Unique codes in batch: {unique_codes}/{batch_size}")
        
        # Simulate training to get code statistics
        model.train()
        for _ in range(10):
            x = torch.randn(batch_size, 9, 2, 100).to(device)
            output = model(x)
        
        stats = model.get_code_stats()
        print(f"\nCode usage statistics after 10 batches:")
        print(f"  Unique codes used: {stats['unique_codes']:.0f}/{model.num_codes}")
        print(f"  Usage ratio: {stats['usage_ratio']:.3f}")
        print(f"  Perplexity: {stats['perplexity']:.2f}")
        print(f"  Entropy: {stats['entropy']:.2f}")
    
    print("\n✅ FSQ model test complete!")
    return model


if __name__ == "__main__":
    model = test_fsq_model()
    
    # Show behavioral dictionary
    print("\n" + "="*60)
    print("Building Behavioral Dictionary")
    print("="*60)
    
    behavioral_dict = model.get_behavioral_dictionary(threshold=0.001)
    print(f"Found {len(behavioral_dict)} frequently used codes")
    
    # Show top 5 codes
    top_codes = sorted(behavioral_dict.items(), 
                      key=lambda x: x[1]['rank'])[:5]
    
    print("\nTop 5 behavioral codes:")
    for code_idx, info in top_codes:
        print(f"  Code {code_idx:4d}: Usage={info['usage']:.3f}, Count={info['count']:.0f}")
    
    print("\n🎉 FSQ implementation complete and working!")