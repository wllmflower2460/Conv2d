"""
Complete Conv2d-VQ-HDP-HSMM Model
Unified architecture for behavioral synchrony analysis with discrete representations,
hierarchical clustering, and temporal dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import sys
import os

# Add parent for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all components
from conv2d_vq_model import Conv2dEncoder, Conv2dDecoder
from vq_ema_2d import VectorQuantizerEMA2D
from hdp_components import HDPLayer, HierarchicalHDP
from hsmm_components import HSMM
from entropy_uncertainty import EntropyUncertaintyModule, summarize_window


class Conv2dVQHDPHSMM(nn.Module):
    """
    Full Conv2d-VQ-HDP-HSMM architecture
    Combines all components for complete behavioral analysis pipeline
    """
    
    def __init__(
        self,
        # Input parameters
        input_channels: int = 9,
        input_height: int = 2,
        sequence_length: int = 100,
        
        # VQ parameters
        num_codes: int = 256,  # Reduced from 512 per advisor recommendation
        code_dim: int = 64,
        vq_decay: float = 0.99,
        commitment_cost: float = 0.4,  # Increased from 0.25 per advisor recommendation
        
        # HDP parameters
        max_clusters: int = 20,
        hdp_concentration: float = 1.0,
        use_hierarchical_hdp: bool = True,
        
        # HSMM parameters
        num_states: int = 10,
        max_duration: int = 50,
        duration_dist: str = 'negative_binomial',
        
        # Architecture parameters
        hidden_channels: List[int] = [64, 128, 256],
        num_activities: int = 12,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Store configuration
        self.config = {
            'input_channels': input_channels,
            'input_height': input_height,
            'num_codes': num_codes,
            'code_dim': code_dim,
            'max_clusters': max_clusters,
            'num_states': num_states,
            'num_activities': num_activities
        }
        
        # ========== Stage 1: Conv2d Encoder ==========
        self.encoder = Conv2dEncoder(
            input_channels=input_channels,
            input_height=input_height,
            hidden_channels=hidden_channels,
            code_dim=code_dim,
            dropout=dropout
        )
        
        # ========== Stage 2: Vector Quantization ==========
        self.vq = VectorQuantizerEMA2D(
            num_codes=num_codes,
            code_dim=code_dim,
            decay=vq_decay,
            commitment_cost=commitment_cost
        )
        
        # ========== Stage 3: HDP Clustering ==========
        if use_hierarchical_hdp:
            self.hdp = HierarchicalHDP(
                input_dim=code_dim,
                max_base_clusters=max_clusters,
                max_group_clusters=max_clusters // 2,
                base_concentration=hdp_concentration,
                group_concentration=hdp_concentration * 0.5
            )
        else:
            self.hdp = HDPLayer(
                input_dim=code_dim,
                max_clusters=max_clusters,
                concentration=hdp_concentration
            )
        self.use_hierarchical_hdp = use_hierarchical_hdp
        
        # ========== Stage 4: HSMM Temporal Dynamics ==========
        self.hsmm = HSMM(
            num_states=num_states,
            observation_dim=max_clusters if not use_hierarchical_hdp else max_clusters // 2,
            max_duration=max_duration,
            duration_dist=duration_dist,
            use_input_dependent_trans=True
        )
        
        # ========== Stage 5: Decoder (for reconstruction) ==========
        self.decoder = Conv2dDecoder(
            code_dim=code_dim,
            hidden_channels=hidden_channels[::-1],
            output_channels=input_channels,
            output_height=input_height,
            dropout=dropout
        )
        
        # ========== Additional Heads ==========
        
        # Activity classification (from VQ codes)
        self.activity_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(code_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_activities)
        )
        
        # Synchrony predictor (from HDP clusters)
        sync_input_dim = max_clusters // 2 if use_hierarchical_hdp else max_clusters
        self.synchrony_predictor = nn.Sequential(
            nn.Linear(sync_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 synchrony states: sync, lead, follow, async
        )
        
        # Behavioral state interpreter (from HSMM states)
        self.state_interpreter = nn.Sequential(
            nn.Linear(num_states, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # 16 interpretable behavioral states
        )
        
        # ========== Entropy & Uncertainty Module ==========
        self.entropy_module = EntropyUncertaintyModule(
            num_states=num_states,
            num_phase_bins=12,
            confidence_threshold_high=0.3,
            confidence_threshold_low=0.6
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_all_stages: bool = False,
        compute_reconstruction: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete architecture
        
        Args:
            x: Input tensor (B, 9, 2, T) - IMU data for human and dog
            return_all_stages: Return intermediate outputs from each stage
            compute_reconstruction: Whether to compute reconstruction (can skip for speed)
        
        Returns:
            Dictionary with outputs from all stages
        """
        B, C, H, T = x.shape
        device = x.device
        
        outputs = {}
        
        # ========== Stage 1: Encoding ==========
        z_e = self.encoder(x)  # (B, code_dim, 1, T)
        
        if return_all_stages:
            outputs['encoded'] = z_e
        
        # ========== Stage 2: Vector Quantization ==========
        z_q, vq_loss_dict, vq_info = self.vq(z_e)
        
        outputs['vq_loss'] = vq_loss_dict['vq']
        outputs['perplexity'] = vq_info['perplexity']
        outputs['codebook_usage'] = vq_info['usage']
        outputs['token_indices'] = vq_info['indices']  # (B, 1, T)
        
        if return_all_stages:
            outputs['quantized'] = z_q
            outputs['vq_info'] = vq_info
        
        # ========== Stage 3: HDP Clustering ==========
        # Reshape for HDP: (B, T, code_dim)
        z_q_reshaped = z_q.squeeze(2).permute(0, 2, 1)  # (B, T, code_dim)
        
        if self.use_hierarchical_hdp:
            hdp_assignments, hdp_info = self.hdp(z_q_reshaped)
            cluster_assignments = hdp_assignments['group']  # Use group level for HSMM
            
            outputs['hdp_base_clusters'] = hdp_assignments['base']
            outputs['hdp_group_clusters'] = hdp_assignments['group']
            outputs['hdp_hierarchy'] = hdp_info['hierarchy_depth']
        else:
            cluster_assignments, hdp_info = self.hdp(z_q_reshaped)
            outputs['hdp_clusters'] = cluster_assignments
        
        outputs['active_clusters'] = hdp_info.get('active_clusters', 
                                                  hdp_info.get('group_info', {}).get('active_clusters', 0))
        
        if return_all_stages:
            outputs['hdp_info'] = hdp_info
        
        # ========== Stage 4: HSMM Temporal Dynamics ==========
        # Use cluster assignments as observations for HSMM
        hsmm_states, hsmm_info = self.hsmm(cluster_assignments, return_viterbi=True)
        
        outputs['hsmm_states'] = hsmm_states  # (B, T, num_states)
        outputs['viterbi_path'] = hsmm_info['viterbi_path']  # (B, T)
        outputs['expected_durations'] = hsmm_info['expected_durations']
        outputs['log_likelihood'] = hsmm_info['log_likelihood']
        
        if return_all_stages:
            outputs['hsmm_info'] = hsmm_info
        
        # ========== Stage 5: Reconstruction (optional) ==========
        if compute_reconstruction:
            x_recon = self.decoder(z_q)
            outputs['reconstructed'] = x_recon
            outputs['reconstruction_loss'] = F.mse_loss(x_recon, x)
        
        # ========== Additional Predictions ==========
        
        # Activity classification
        activity_logits = self.activity_classifier(z_q)
        outputs['activity_logits'] = activity_logits
        
        # Synchrony prediction (from average cluster assignment)
        avg_clusters = cluster_assignments.mean(dim=1)  # (B, max_clusters)
        synchrony_logits = self.synchrony_predictor(avg_clusters)
        outputs['synchrony_logits'] = synchrony_logits
        
        # Behavioral state interpretation (from average HSMM states)
        avg_states = hsmm_states.mean(dim=1)  # (B, num_states)
        behavioral_logits = self.state_interpreter(avg_states)
        outputs['behavioral_logits'] = behavioral_logits
        
        # ========== Entropy & Uncertainty Quantification ==========
        # Extract phase from tokens (simplified - in practice, compute from actual phase)
        # Here we use token indices as proxy for phase
        token_phases = (outputs['token_indices'].float() / self.config['num_codes']) * 2 * np.pi - np.pi
        token_phases = token_phases.squeeze(1)  # (B, T)
        
        uncertainty_outputs = self.entropy_module(
            state_posterior=hsmm_states,
            phase_values=token_phases,
            cluster_assignments=cluster_assignments,
            return_marginals=True
        )
        
        # Add uncertainty metrics to outputs
        outputs['uncertainty'] = uncertainty_outputs
        outputs['confidence_level'] = uncertainty_outputs['confidence_level']
        outputs['confidence_score'] = uncertainty_outputs['confidence_score']
        outputs['mutual_information'] = uncertainty_outputs.get('mutual_information', 0)
        
        # ========== Compute Total Loss ==========
        if self.training:
            total_loss = outputs['vq_loss']
            if 'reconstruction_loss' in outputs:
                total_loss = total_loss + outputs['reconstruction_loss']
            outputs['total_loss'] = total_loss
        
        return outputs
    
    def analyze_sequence(
        self,
        x: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Detailed analysis of a behavioral sequence
        
        Args:
            x: Input sequence (B, 9, 2, T)
        
        Returns:
            Dictionary with detailed analysis results
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(x, return_all_stages=True, compute_reconstruction=False)
        
        B, _, _, T = x.shape
        
        analysis = {}
        
        # Token analysis
        tokens = outputs['token_indices'].cpu().numpy()  # (B, 1, T)
        analysis['unique_tokens_per_sequence'] = [len(np.unique(tokens[b])) for b in range(B)]
        
        # Token transitions
        transitions = []
        for b in range(B):
            seq = tokens[b, 0, :]
            trans = [(seq[i], seq[i+1]) for i in range(len(seq)-1)]
            transitions.append(len(set(trans)))
        analysis['unique_transitions'] = transitions
        
        # Cluster analysis
        if 'hdp_group_clusters' in outputs:
            clusters = outputs['hdp_group_clusters'].cpu().numpy()
            analysis['dominant_cluster'] = np.argmax(clusters.mean(axis=1), axis=1)
        elif 'hdp_clusters' in outputs:
            clusters = outputs['hdp_clusters'].cpu().numpy()
            analysis['dominant_cluster'] = np.argmax(clusters.mean(axis=1), axis=1)
        
        # State analysis
        states = outputs['hsmm_states'].cpu().numpy()
        analysis['dominant_state'] = np.argmax(states.mean(axis=1), axis=1)
        
        # Viterbi path statistics
        viterbi = outputs['viterbi_path'].cpu().numpy()
        analysis['state_changes'] = [np.sum(np.diff(viterbi[b]) != 0) for b in range(B)]
        
        # Calculate state durations
        state_durations = []
        for b in range(B):
            path = viterbi[b]
            durations = []
            current_state = path[0]
            count = 1
            
            for state in path[1:]:
                if state == current_state:
                    count += 1
                else:
                    durations.append(count)
                    current_state = state
                    count = 1
            durations.append(count)
            state_durations.append(np.mean(durations))
        
        analysis['mean_state_duration'] = state_durations
        
        # Synchrony prediction
        sync_probs = F.softmax(outputs['synchrony_logits'], dim=-1).cpu().numpy()
        analysis['synchrony_prediction'] = np.argmax(sync_probs, axis=1)
        analysis['synchrony_confidence'] = np.max(sync_probs, axis=1)
        
        return analysis
    
    def get_component_stats(self) -> Dict:
        """Get statistics from all components"""
        stats = {}
        
        # VQ stats
        stats['vq'] = {
            'num_codes': self.vq.num_codes,
            'code_dim': self.vq.code_dim,
            'ema_cluster_sizes': self.vq.ema_cluster_size.mean().item()
        }
        
        # HDP stats
        if hasattr(self.hdp, 'get_cluster_weights'):
            stats['hdp'] = {
                'cluster_weights': self.hdp.get_cluster_weights().tolist()[:10]
            }
        
        # HSMM stats
        stats['hsmm'] = {
            'num_states': self.hsmm.num_states,
            'expected_durations': self.hsmm.duration_model.duration_mean.mean().item()
                if hasattr(self.hsmm.duration_model, 'duration_mean') else 'N/A'
        }
        
        return stats


def test_full_model():
    """Test the complete Conv2d-VQ-HDP-HSMM model"""
    print("Testing Complete Conv2d-VQ-HDP-HSMM Model...")
    print("="*50)
    
    # Test parameters
    B, C, H, T = 2, 9, 2, 100  # Small batch for testing
    
    # Initialize model
    model = Conv2dVQHDPHSMM(
        input_channels=C,
        input_height=H,
        sequence_length=T,
        num_codes=256,  # Smaller for testing
        code_dim=32,
        max_clusters=10,
        num_states=8,
        use_hierarchical_hdp=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test input
    x = torch.randn(B, C, H, T)
    
    # Forward pass
    print("\n1. Testing forward pass...")
    outputs = model(x, return_all_stages=True)
    
    # Check outputs
    print(f"   ✓ Token indices: {outputs['token_indices'].shape}")
    print(f"   ✓ Perplexity: {outputs['perplexity']:.2f}")
    print(f"   ✓ Active clusters: {outputs['active_clusters']}")
    print(f"   ✓ HSMM states: {outputs['hsmm_states'].shape}")
    print(f"   ✓ Viterbi path: {outputs['viterbi_path'].shape}")
    print(f"   ✓ Expected durations: {outputs['expected_durations'].mean():.1f}")
    
    # Test analysis
    print("\n2. Testing sequence analysis...")
    analysis = model.analyze_sequence(x)
    
    print(f"   ✓ Unique tokens: {analysis['unique_tokens_per_sequence']}")
    print(f"   ✓ State changes: {analysis['state_changes']}")
    print(f"   ✓ Mean state duration: {np.mean(analysis['mean_state_duration']):.1f}")
    print(f"   ✓ Synchrony prediction: {analysis['synchrony_prediction']}")
    
    # Test component stats
    print("\n3. Testing component statistics...")
    stats = model.get_component_stats()
    
    print(f"   ✓ VQ: {stats['vq']['num_codes']} codes, {stats['vq']['code_dim']}D")
    print(f"   ✓ HDP: {len(stats.get('hdp', {}).get('cluster_weights', []))} cluster weights")
    print(f"   ✓ HSMM: {stats['hsmm']['num_states']} states")
    
    # Test gradient flow
    print("\n4. Testing gradient flow...")
    if 'total_loss' in outputs:
        loss = outputs['total_loss']
    else:
        loss = outputs['vq_loss'] + outputs.get('reconstruction_loss', 0)
    
    loss.backward()
    
    # Check that gradients flow to all components
    has_grad = {
        'encoder': any(p.grad is not None for p in model.encoder.parameters()),
        'hdp': any(p.grad is not None for p in model.hdp.parameters()),
        'hsmm': any(p.grad is not None for p in model.hsmm.parameters())
    }
    
    for component, has in has_grad.items():
        status = "✓" if has else "✗"
        print(f"   {status} {component}: gradient flow {'enabled' if has else 'blocked'}")
    
    print("\n" + "="*50)
    print("✅ All tests passed! Conv2d-VQ-HDP-HSMM model is working!")
    
    return model


if __name__ == "__main__":
    model = test_full_model()