"""
Main Conv2d-VQ-HDP-HSMM model combining all components.
Designed for behavioral analysis on edge devices like Hailo-8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vector_quantization import VectorQuantization
from .hdp_clustering import HDPClustering
from .hsmm import HSMM


class TCNBlock(nn.Module):
    """
    Simplified Temporal Convolutional Network block using only Conv2d operations.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        
        # Use same padding to maintain dimensions
        padding = (kernel_size - 1) // 2
        
        # Use Conv2d with kernel (kernel_size, 1) for temporal convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=(kernel_size, 1), 
                              padding=(padding, 0))
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=(kernel_size, 1), 
                              padding=(padding, 0))
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        """
        Forward pass through TCN block.
        
        Args:
            x: Input tensor (B, C, H, W) where H is temporal dimension
            
        Returns:
            Output tensor (B, out_channels, H, W)
        """
        residual = self.residual(x)
        
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        
        return torch.relu(out + residual)


class Conv2dVQHDPHSMM(nn.Module):
    """
    Conv2d-VQ-HDP-HSMM model for behavioral analysis.
    
    Input shape: (B, 9, 2, 100) - Batch, Channels, Features, Time
    - B: Batch size
    - 9: Dual-device IMU channels (3x3 for accel/gyro/mag per device)
    - 2: Number of IMU devices
    - 100: Time steps
    
    Architecture:
    1. TCN Encoder: Conv2d-based temporal feature extraction
    2. Vector Quantization: Discrete latent space with 512 codes
    3. HDP Clustering: Hierarchical clustering with max 20 clusters
    4. HSMM: Temporal dynamics modeling
    5. TCN Decoder: Reconstruction
    """
    
    def __init__(self, 
                 input_channels=9,
                 input_features=2, 
                 sequence_length=100,
                 latent_dim=64,
                 num_embeddings=512,
                 max_clusters=20,
                 num_states=10,
                 max_duration=10):
        super(Conv2dVQHDPHSMM, self).__init__()
        
        self.input_channels = input_channels
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # TCN Encoder - using Conv2d for temporal modeling
        # Input: (B, 9, 2, 100) -> Output: (B, latent_dim, 2, T')
        self.encoder = nn.ModuleList([
            # Initial projection
            nn.Conv2d(input_channels, 32, kernel_size=1),
            TCNBlock(32, 64, kernel_size=3, dilation=1),
            TCNBlock(64, 128, kernel_size=3, dilation=2),
            TCNBlock(128, 256, kernel_size=3, dilation=4),
            # Final projection to latent space
            nn.Conv2d(256, latent_dim, kernel_size=1)
        ])
        
        # Vector Quantization
        self.vq_layer = VectorQuantization(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim
        )
        
        # HDP Clustering
        self.hdp_layer = HDPClustering(
            input_dim=latent_dim,
            max_clusters=max_clusters
        )
        
        # HSMM for temporal modeling
        self.hsmm_layer = HSMM(
            num_states=num_states,
            max_duration=max_duration,
            feature_dim=latent_dim
        )
        
        # TCN Decoder - reconstruct from quantized features
        self.decoder = nn.ModuleList([
            nn.Conv2d(latent_dim, 256, kernel_size=1),
            TCNBlock(256, 128, kernel_size=3, dilation=4),
            TCNBlock(128, 64, kernel_size=3, dilation=2),
            TCNBlock(64, 32, kernel_size=3, dilation=1),
            # Final projection back to input space
            nn.Conv2d(32, input_channels, kernel_size=1)
        ])
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x):
        """
        Encode input through TCN encoder.
        
        Args:
            x: Input tensor (B, 9, 2, 100)
            
        Returns:
            Encoded features (B, latent_dim, 2, T')
        """
        # Pass through encoder layers
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                x = torch.relu(layer(x))
            else:
                x = layer(x)
        
        return x
    
    def decode(self, x):
        """
        Decode quantized features back to input space.
        
        Args:
            x: Quantized features (B, latent_dim, 2, T')
            
        Returns:
            Reconstructed input (B, 9, 2, T')
        """
        # Pass through decoder layers
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                if i < len(self.decoder) - 1:  # Don't apply ReLU to final output
                    x = torch.relu(x)
            else:
                x = layer(x)
        
        return x
    
    def forward(self, x, return_intermediate=False):
        """
        Forward pass through the complete model.
        
        Args:
            x: Input tensor (B, 9, 2, 100)
            return_intermediate: Whether to return intermediate results
            
        Returns:
            outputs: Dictionary containing:
                - reconstructed: Reconstructed input
                - vq_loss: Vector quantization loss
                - hdp_loss: HDP clustering loss
                - hsmm_likelihood: HSMM log likelihood
                - perplexity: VQ codebook perplexity
                - cluster_assignments: HDP cluster assignments
                - state_probs: HSMM state probabilities
        """
        B, C, F, T = x.shape
        
        # Encode input
        encoded = self.encode(x)  # (B, latent_dim, F, T')
        
        # Vector Quantization
        quantized, vq_loss, perplexity, encodings = self.vq_layer(encoded)
        
        # Create temporal sequence for HSMM (use first feature dimension as context)
        context_features = quantized.mean(dim=2)  # (B, latent_dim, T')
        context_features = context_features.unsqueeze(-1)  # (B, latent_dim, T', 1) for HSMM
        
        # Prepare sequence for HSMM - need (B, latent_dim, T, H, W) format
        T_prime = quantized.shape[3]
        hsmm_sequence = quantized.permute(0, 1, 3, 2).unsqueeze(-1)  # (B, latent_dim, T', F, 1)
        hsmm_context = context_features[:, :, 0:1, :]  # (B, latent_dim, 1, 1) - take first time step as context
        
        # HSMM temporal modeling
        state_probs, hsmm_likelihood, duration_probs = self.hsmm_layer(
            hsmm_context, hsmm_sequence
        )
        
        # HDP Clustering on quantized features
        cluster_assignments, cluster_centers, hdp_loss = self.hdp_layer(quantized)
        
        # Decode back to input space
        reconstructed = self.decode(quantized)
        
        # Prepare outputs
        outputs = {
            'reconstructed': reconstructed,
            'vq_loss': vq_loss,
            'hdp_loss': hdp_loss,
            'hsmm_likelihood': hsmm_likelihood.mean(),  # Average over spatial dimensions
            'perplexity': perplexity,
            'cluster_assignments': cluster_assignments,
            'state_probs': state_probs,
            'duration_probs': duration_probs
        }
        
        if return_intermediate:
            outputs.update({
                'encoded': encoded,
                'quantized': quantized,
                'encodings': encodings,
                'cluster_centers': cluster_centers
            })
        
        return outputs
    
    def compute_total_loss(self, outputs, reconstruction_weight=1.0, vq_weight=1.0, 
                          hdp_weight=0.1, hsmm_weight=0.1):
        """
        Compute total training loss.
        
        Args:
            outputs: Model outputs dictionary
            reconstruction_weight: Weight for reconstruction loss
            vq_weight: Weight for VQ loss
            hdp_weight: Weight for HDP loss
            hsmm_weight: Weight for HSMM loss
            
        Returns:
            total_loss: Combined training loss
            loss_components: Dictionary of individual loss components
        """
        # Reconstruction loss (assuming we have original input)
        reconstruction_loss = outputs.get('reconstruction_loss', 0.0)
        
        loss_components = {
            'reconstruction': reconstruction_loss * reconstruction_weight,
            'vq': outputs['vq_loss'] * vq_weight,
            'hdp': outputs['hdp_loss'] * hdp_weight,
            'hsmm': -outputs['hsmm_likelihood'] * hsmm_weight  # Negative log likelihood
        }
        
        total_loss = sum(loss_components.values())
        
        return total_loss, loss_components
    
    def get_behavioral_analysis(self, outputs):
        """
        Extract behavioral analysis metrics from model outputs.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            analysis: Dictionary with behavioral metrics
        """
        with torch.no_grad():
            # VQ codebook usage
            vq_stats = self.vq_layer.get_codebook_usage()
            
            # HDP cluster statistics
            hdp_stats = self.hdp_layer.get_cluster_statistics(outputs['cluster_assignments'])
            
            # HSMM state analysis
            state_probs = outputs['state_probs']  # (B, num_states, T, H, W)
            dominant_states = torch.argmax(state_probs, dim=1)  # (B, T, H, W)
            
            # State transition analysis
            state_transitions = []
            for b in range(dominant_states.shape[0]):
                transitions = dominant_states[b, 1:] != dominant_states[b, :-1]
                transition_rate = transitions.float().mean().item()
                state_transitions.append(transition_rate)
            
            analysis = {
                'codebook_utilization': vq_stats['utilization'],
                'active_codes': vq_stats['active_codes'],
                'perplexity': outputs['perplexity'].item(),
                'active_clusters': hdp_stats['active_clusters'],
                'cluster_entropy': hdp_stats['entropy'],
                'avg_transition_rate': sum(state_transitions) / len(state_transitions),
                'dominant_states': dominant_states.cpu().numpy(),
                'cluster_assignments': outputs['cluster_assignments'].cpu().numpy()
            }
            
            return analysis
    
    def extract_features(self, x):
        """
        Extract features for downstream analysis.
        
        Args:
            x: Input tensor (B, 9, 2, 100)
            
        Returns:
            features: Dictionary with extracted features
        """
        with torch.no_grad():
            outputs = self.forward(x, return_intermediate=True)
            
            features = {
                'encoded': outputs['encoded'].cpu().numpy(),
                'quantized': outputs['quantized'].cpu().numpy(),
                'vq_codes': outputs['encodings'].argmax(dim=1).cpu().numpy(),
                'cluster_assignments': outputs['cluster_assignments'].cpu().numpy(),
                'state_sequence': outputs['state_probs'].argmax(dim=1).cpu().numpy()
            }
            
            return features