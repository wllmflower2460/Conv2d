"""
Device Attention Module for Phone+IMU Dual-Sensor Processing
Implements attention mechanism to optimally combine phone and collar IMU data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class PhoneIMUAttention(nn.Module):
    """
    Advanced attention mechanism for phone+IMU sensor fusion
    Learns activity-specific weighting between phone and collar sensors
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, 
                 dropout: float = 0.1, temperature: float = 1.0):
        super(PhoneIMUAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention between devices
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True  
        )
        
        # Learnable device embeddings
        self.phone_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.imu_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Device-specific transformations
        self.phone_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.imu_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism for device selection
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, phone_features: torch.Tensor, 
                imu_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine phone and IMU features with attention
        
        Args:
            phone_features: (B, T, C) - Phone sensor features
            imu_features: (B, T, C) - Collar IMU features
            
        Returns:
            combined: (B, T, C) - Combined features
            attention_weights: (B, 2) - Device attention weights
        """
        B, T, C = phone_features.shape
        
        # Add device embeddings
        phone_emb = self.phone_embedding.expand(B, T, -1)
        imu_emb = self.imu_embedding.expand(B, T, -1)
        
        phone_with_emb = phone_features + phone_emb
        imu_with_emb = imu_features + imu_emb
        
        # Self-attention within each device
        phone_self, _ = self.self_attention(
            phone_with_emb, phone_with_emb, phone_with_emb
        )
        imu_self, _ = self.self_attention(
            imu_with_emb, imu_with_emb, imu_with_emb
        )
        
        # Cross-attention between devices
        phone_to_imu, _ = self.cross_attention(
            phone_self, imu_self, imu_self
        )
        imu_to_phone, _ = self.cross_attention(
            imu_self, phone_self, phone_self
        )
        
        # Apply device-specific transformations
        phone_transformed = self.phone_transform(phone_to_imu)
        imu_transformed = self.imu_transform(imu_to_phone)
        
        # Calculate gating weights
        global_phone = phone_transformed.mean(dim=1)  # (B, C)
        global_imu = imu_transformed.mean(dim=1)      # (B, C)
        
        gate_input = torch.cat([global_phone, global_imu], dim=-1)
        gate_weights = self.gate(gate_input)  # (B, 2)
        
        # Apply gated combination
        phone_weighted = phone_transformed * gate_weights[:, 0:1].unsqueeze(1)
        imu_weighted = imu_transformed * gate_weights[:, 1:2].unsqueeze(1)
        
        # Combine features
        combined = torch.cat([phone_weighted, imu_weighted], dim=-1)
        output = self.output_projection(combined)
        
        return output, gate_weights


class TemporalDeviceAttention(nn.Module):
    """
    Temporal attention that considers device importance over time
    Useful for activities where device relevance changes dynamically
    """
    def __init__(self, hidden_dim: int, window_size: int = 10):
        super(TemporalDeviceAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        
        # Temporal convolution for local attention
        self.temporal_conv = nn.Conv1d(
            hidden_dim * 2,  # Phone + IMU
            hidden_dim,
            kernel_size=window_size,
            padding=window_size // 2
        )
        
        # Attention score computation
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # 2 devices
        )
        
        # Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, phone_features: torch.Tensor,
                imu_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention across devices
        
        Args:
            phone_features: (B, C, T) - Phone features
            imu_features: (B, C, T) - IMU features
            
        Returns:
            combined: (B, C, T) - Combined features
            attention_scores: (B, 2, T) - Temporal attention scores
        """
        B, C, T = phone_features.shape
        
        # Concatenate devices
        combined = torch.cat([phone_features, imu_features], dim=1)  # (B, 2*C, T)
        
        # Apply temporal convolution
        temporal_features = self.temporal_conv(combined)  # (B, C, T)
        
        # Compute attention scores for each timestep
        temporal_features_t = temporal_features.permute(0, 2, 1)  # (B, T, C)
        attention_logits = self.attention_fc(temporal_features_t)  # (B, T, 2)
        attention_scores = F.softmax(attention_logits / 0.1, dim=-1)  # Temperature scaling
        
        # Apply attention
        attention_scores_t = attention_scores.permute(0, 2, 1)  # (B, 2, T)
        
        phone_weighted = phone_features * attention_scores_t[:, 0:1, :]
        imu_weighted = imu_features * attention_scores_t[:, 1:2, :]
        
        # Combine and normalize
        output = phone_weighted + imu_weighted
        output_t = output.permute(0, 2, 1)  # (B, T, C)
        output_norm = self.layer_norm(output_t)
        output = output_norm.permute(0, 2, 1)  # (B, C, T)
        
        return output, attention_scores_t


class CrossSpeciesAttention(nn.Module):
    """
    Attention mechanism specifically designed for cross-species transfer
    Learns mappings between human and dog behavioral patterns
    """
    def __init__(self, hidden_dim: int, num_species: int = 2):
        super(CrossSpeciesAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_species = num_species
        
        # Species-specific embeddings
        self.species_embeddings = nn.Embedding(num_species, hidden_dim)
        
        # Behavioral pattern matching
        self.pattern_matcher = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Transfer learning adapter
        self.species_adapter = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_species)
        ])
        
        # Cache for species index tensors to avoid repeated allocation
        # Structure: {(species_idx: int, device_type: str, device_index: int): torch.Tensor}
        #   - Key: tuple of (species_idx, device.type, device.index)
        #   - Value: torch.Tensor containing [species_idx] on the specified device
        self._species_tensor_cache = {}
        
    def _get_species_tensor(self, species_idx: int, device: torch.device) -> torch.Tensor:
        """
        Get cached species index tensor for the given device.
        Creates and caches the tensor on first use for each device.
        
        Args:
            species_idx: Species index (0=human, 1=dog)
            device: Target device for the tensor
            
        Returns:
            Cached tensor containing [species_idx] on the specified device
        """
        # Use device.type and device.index for efficient cache key (avoids string conversion)
        cache_key = (species_idx, device.type, device.index)
        if cache_key not in self._species_tensor_cache:
            # Create and cache the tensor for this species/device combination
            self._species_tensor_cache[cache_key] = torch.tensor(
                [species_idx], device=device, dtype=torch.long
            )
        return self._species_tensor_cache[cache_key]
        
    def forward(self, features: torch.Tensor, 
                source_species: int, 
                target_species: int) -> torch.Tensor:
        """
        Apply cross-species attention for transfer learning
        
        Args:
            features: (B, C, T) - Input features
            source_species: Source species index (0=human, 1=dog)
            target_species: Target species index
            
        Returns:
            adapted_features: (B, C, T) - Species-adapted features
        """
        B, C, T = features.shape
        
        # Get species embeddings using cached tensors
        source_tensor = self._get_species_tensor(source_species, features.device)
        target_tensor = self._get_species_tensor(target_species, features.device)
        
        source_emb = self.species_embeddings(source_tensor).expand(B, -1)  # (B, C)
        target_emb = self.species_embeddings(target_tensor).expand(B, -1)  # (B, C)
        
        # Compute behavioral pattern similarity
        features_global = features.mean(dim=-1)  # (B, C)
        
        pattern_input = torch.cat([
            features_global + source_emb,
            features_global + target_emb
        ], dim=-1)  # (B, 2*C)
        
        similarity = torch.sigmoid(self.pattern_matcher(pattern_input))  # (B, 1)
        
        # Apply species-specific adaptation
        features_t = features.permute(0, 2, 1)  # (B, T, C)
        adapted = self.species_adapter[target_species](features_t)
        adapted = adapted.permute(0, 2, 1)  # (B, C, T)
        
        # Weight by similarity
        output = features + similarity.unsqueeze(-1) * (adapted - features)
        
        return output


class HierarchicalDeviceAttention(nn.Module):
    """
    Hierarchical attention for multi-scale temporal patterns
    Captures both short-term and long-term device importance
    """
    def __init__(self, hidden_dim: int, num_scales: int = 3):
        super(HierarchicalDeviceAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Multi-scale attention modules
        self.scale_attentions = nn.ModuleList([
            TemporalDeviceAttention(hidden_dim, window_size=2**(i+2))
            for i in range(num_scales)
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, phone_features: torch.Tensor,
                imu_features: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Apply hierarchical multi-scale attention
        
        Args:
            phone_features: (B, C, T) - Phone features
            imu_features: (B, C, T) - IMU features
            
        Returns:
            fused: (B, C, T) - Multi-scale fused features
            scale_attentions: List of attention maps at each scale
        """
        scale_outputs = []
        scale_attentions = []
        
        # Apply attention at each scale
        for scale_attention in self.scale_attentions:
            output, attention = scale_attention(phone_features, imu_features)
            scale_outputs.append(output)
            scale_attentions.append(attention)
        
        # Concatenate scale outputs
        B, C, T = phone_features.shape
        concatenated = torch.cat(scale_outputs, dim=1)  # (B, num_scales*C, T)
        
        # Fuse scales
        concatenated_t = concatenated.permute(0, 2, 1)  # (B, T, num_scales*C)
        fused_t = self.scale_fusion(concatenated_t)  # (B, T, C)
        fused = fused_t.permute(0, 2, 1)  # (B, C, T)
        
        return fused, scale_attentions


if __name__ == "__main__":
    # Test device attention modules
    print("🧪 Testing Device Attention Modules...")
    
    B, T, C = 4, 100, 64  # Batch, Time, Channels
    
    # Test PhoneIMUAttention
    print("\n1. Testing PhoneIMUAttention...")
    attention = PhoneIMUAttention(hidden_dim=C)
    phone_feat = torch.randn(B, T, C)
    imu_feat = torch.randn(B, T, C)
    
    combined, weights = attention(phone_feat, imu_feat)
    print(f"   Input shapes: Phone {phone_feat.shape}, IMU {imu_feat.shape}")
    print(f"   Output shape: {combined.shape}")
    print(f"   Attention weights: {weights.shape}")
    print(f"   ✅ PhoneIMUAttention test passed!")
    
    # Test TemporalDeviceAttention
    print("\n2. Testing TemporalDeviceAttention...")
    temporal_attention = TemporalDeviceAttention(hidden_dim=C)
    phone_feat_t = torch.randn(B, C, T)
    imu_feat_t = torch.randn(B, C, T)
    
    combined_t, attention_scores = temporal_attention(phone_feat_t, imu_feat_t)
    print(f"   Input shapes: Phone {phone_feat_t.shape}, IMU {imu_feat_t.shape}")
    print(f"   Output shape: {combined_t.shape}")
    print(f"   Attention scores: {attention_scores.shape}")
    print(f"   ✅ TemporalDeviceAttention test passed!")
    
    # Test CrossSpeciesAttention
    print("\n3. Testing CrossSpeciesAttention...")
    species_attention = CrossSpeciesAttention(hidden_dim=C)
    features = torch.randn(B, C, T)
    
    adapted = species_attention(features, source_species=0, target_species=1)
    print(f"   Input shape: {features.shape}")
    print(f"   Adapted shape: {adapted.shape}")
    print(f"   ✅ CrossSpeciesAttention test passed!")
    
    # Test HierarchicalDeviceAttention
    print("\n4. Testing HierarchicalDeviceAttention...")
    hierarchical = HierarchicalDeviceAttention(hidden_dim=C)
    
    fused, scale_maps = hierarchical(phone_feat_t, imu_feat_t)
    print(f"   Input shapes: Phone {phone_feat_t.shape}, IMU {imu_feat_t.shape}")
    print(f"   Fused output shape: {fused.shape}")
    print(f"   Number of scale maps: {len(scale_maps)}")
    print(f"   ✅ HierarchicalDeviceAttention test passed!")
    
    print("\n✅ All device attention modules tested successfully!")