"""
Hierarchical Dirichlet Process (HDP) Components for Conv2d-VQ-HDP-HSMM
Implements non-parametric Bayesian clustering for automatic behavioral discovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List


class StickBreaking(nn.Module):
    """
    Stick-breaking construction for Dirichlet Process
    Generates weights from Beta distributions
    """
    
    def __init__(self, max_clusters: int = 20, alpha: float = 1.0):
        super().__init__()
        self.max_clusters = max_clusters
        self.alpha = alpha
        
        # Learnable concentration parameter
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha)))
        
        # Beta distribution parameters for stick-breaking
        self.beta_a = nn.Parameter(torch.ones(max_clusters - 1))
        self.beta_b = nn.Parameter(torch.ones(max_clusters - 1) * alpha)
        
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate stick-breaking weights
        Returns: weights of shape (batch_size, max_clusters)
        """
        alpha = torch.exp(self.log_alpha)
        
        # Sample from Beta distributions using reparameterization
        if self.training:
            # Sample v_k ~ Beta(1, alpha)
            v = torch.distributions.Beta(
                self.beta_a.expand(batch_size, -1),
                self.beta_b.expand(batch_size, -1) * alpha
            ).rsample()
        else:
            # Use mean for inference
            v = self.beta_a / (self.beta_a + self.beta_b * alpha)
            v = v.unsqueeze(0).expand(batch_size, -1)
        
        # Compute stick-breaking weights
        weights = torch.zeros(batch_size, self.max_clusters, device=v.device)
        remaining = torch.ones(batch_size, device=v.device)
        
        for k in range(self.max_clusters - 1):
            weights[:, k] = v[:, k] * remaining
            remaining = remaining * (1 - v[:, k])
        
        weights[:, -1] = remaining
        
        return weights
    
    def get_expected_weights(self) -> torch.Tensor:
        """Get expected weights without sampling"""
        alpha = torch.exp(self.log_alpha)
        v = self.beta_a / (self.beta_a + self.beta_b * alpha)
        
        weights = torch.zeros(self.max_clusters)
        remaining = 1.0
        
        for k in range(self.max_clusters - 1):
            weights[k] = v[k] * remaining
            remaining = remaining * (1 - v[k])
        
        weights[-1] = remaining
        
        return weights


class HDPLayer(nn.Module):
    """
    Hierarchical Dirichlet Process layer for behavioral clustering
    Discovers natural groupings in VQ token sequences
    """
    
    def __init__(
        self,
        input_dim: int,
        max_clusters: int = 20,
        concentration: float = 1.0,
        temperature: float = 1.0,
        use_gumbel: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.max_clusters = max_clusters
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        
        # Stick-breaking for cluster weights
        self.stick_breaking = StickBreaking(max_clusters, concentration)
        
        # Cluster centers (behavioral prototypes)
        self.cluster_centers = nn.Parameter(
            torch.randn(max_clusters, input_dim) * 0.1
        )
        
        # Optional: learnable cluster covariances (diagonal)
        self.cluster_log_vars = nn.Parameter(
            torch.zeros(max_clusters, input_dim)
        )
        
        # Temperature annealing
        self.register_buffer('current_temperature', torch.tensor(temperature))
        
    def forward(
        self, 
        z: torch.Tensor,
        return_distances: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            z: Input features (B, D) or (B, T, D)
            return_distances: Whether to return distances to all clusters
        
        Returns:
            cluster_assignments: Soft cluster assignments (B, K) or (B, T, K)
            info: Dictionary with clustering information
        """
        original_shape = z.shape
        
        # Flatten if temporal dimension present
        if z.dim() == 3:
            B, T, D = z.shape
            z_flat = z.reshape(B * T, D)
        else:
            B, D = z.shape
            T = 1
            z_flat = z
        
        # Get cluster weights from stick-breaking
        cluster_weights = self.stick_breaking(B)
        if T > 1:
            # Repeat for each time step
            cluster_weights = cluster_weights.unsqueeze(1).expand(B, T, -1).reshape(B*T, -1)
        
        # Compute distances to cluster centers
        # Using Mahalanobis distance with learned variances
        cluster_vars = torch.exp(self.cluster_log_vars)
        
        # Expand for broadcasting
        z_expanded = z_flat.unsqueeze(1)  # (B*T, 1, D)
        centers_expanded = self.cluster_centers.unsqueeze(0)  # (1, K, D)
        vars_expanded = cluster_vars.unsqueeze(0)  # (1, K, D)
        
        # Mahalanobis distance
        diff = z_expanded - centers_expanded  # (B*T, K, D)
        distances = torch.sum(diff ** 2 / (vars_expanded + 1e-6), dim=-1)  # (B*T, K)
        
        # Convert distances to log probabilities
        log_probs = -0.5 * distances - 0.5 * torch.sum(self.cluster_log_vars, dim=-1)
        
        # Add cluster weight priors
        log_probs = log_probs + torch.log(cluster_weights + 1e-10)
        
        # Apply temperature and get assignments
        if self.use_gumbel and self.training:
            # Gumbel-Softmax for differentiable sampling
            assignments = F.gumbel_softmax(
                log_probs / self.current_temperature,
                tau=self.current_temperature,
                hard=False,
                dim=-1
            )
        else:
            # Standard softmax
            assignments = F.softmax(log_probs / self.current_temperature, dim=-1)
        
        # Reshape back if needed
        if T > 1:
            assignments = assignments.reshape(B, T, self.max_clusters)
            distances = distances.reshape(B, T, self.max_clusters)
        
        # Calculate clustering metrics
        info = self._compute_metrics(assignments, distances, cluster_weights[:B])
        
        if return_distances:
            info['distances'] = distances
        
        return assignments, info
    
    def _compute_metrics(self, assignments, distances, weights):
        """Compute clustering quality metrics"""
        # Entropy of assignments (lower = more certain)
        entropy = -torch.sum(assignments * torch.log(assignments + 1e-10), dim=-1).mean()
        
        # Number of active clusters (with >1% assignment)
        avg_assignment = assignments.mean(dim=tuple(range(assignments.dim()-1)))
        active_clusters = (avg_assignment > 0.01).sum()
        
        # Cluster balance (how evenly distributed)
        balance = 1.0 - torch.std(avg_assignment) / (torch.mean(avg_assignment) + 1e-10)
        
        # Expected number of clusters (from weights)
        cumsum = torch.cumsum(weights.mean(dim=0), dim=0)
        expected_clusters = torch.searchsorted(cumsum, 0.95) + 1
        
        return {
            'entropy': entropy,
            'active_clusters': active_clusters,
            'balance': balance,
            'expected_clusters': expected_clusters,
            'avg_assignment': avg_assignment
        }
    
    def anneal_temperature(self, step: int, total_steps: int, min_temp: float = 0.1):
        """Anneal temperature during training"""
        progress = step / total_steps
        new_temp = self.temperature * (1 - progress) + min_temp * progress
        self.current_temperature.fill_(new_temp)
        return new_temp
    
    def get_cluster_centers(self) -> torch.Tensor:
        """Get current cluster centers"""
        return self.cluster_centers.detach()
    
    def get_cluster_weights(self) -> torch.Tensor:
        """Get expected cluster weights"""
        return self.stick_breaking.get_expected_weights()


class HierarchicalHDP(nn.Module):
    """
    Two-level HDP for modeling hierarchy in behavioral patterns
    Base level: Individual behaviors
    Group level: Behavioral categories
    """
    
    def __init__(
        self,
        input_dim: int,
        max_base_clusters: int = 20,
        max_group_clusters: int = 10,
        base_concentration: float = 1.0,
        group_concentration: float = 0.5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Base level HDP (fine-grained behaviors)
        self.base_hdp = HDPLayer(
            input_dim=input_dim,
            max_clusters=max_base_clusters,
            concentration=base_concentration
        )
        
        # Group level HDP (behavioral categories)
        self.group_hdp = HDPLayer(
            input_dim=max_base_clusters,  # Input is base cluster assignments
            max_clusters=max_group_clusters,
            concentration=group_concentration
        )
        
        # Optional: direct path for residual connection
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, z: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Hierarchical clustering of behavioral features
        
        Args:
            z: Input features (B, D) or (B, T, D)
        
        Returns:
            assignments: Dict with 'base' and 'group' level assignments
            info: Hierarchical clustering information
        """
        # Base level clustering
        base_assignments, base_info = self.base_hdp(z)
        
        # Group level clustering (cluster the cluster assignments)
        group_assignments, group_info = self.group_hdp(base_assignments)
        
        # Combine assignments hierarchically
        if base_assignments.dim() == 3:  # (B, T, K_base)
            B, T, K_base = base_assignments.shape
            K_group = group_assignments.shape[-1]
            
            # Compute joint assignments: P(base, group)
            base_expanded = base_assignments.unsqueeze(-1)  # (B, T, K_base, 1)
            group_expanded = group_assignments.unsqueeze(-2)  # (B, T, 1, K_group)
            joint_assignments = base_expanded * group_expanded  # (B, T, K_base, K_group)
        else:
            joint_assignments = base_assignments.unsqueeze(-1) * group_assignments.unsqueeze(-2)
        
        assignments = {
            'base': base_assignments,
            'group': group_assignments,
            'joint': joint_assignments
        }
        
        info = {
            'base_info': base_info,
            'group_info': group_info,
            'hierarchy_depth': self._compute_hierarchy_depth(base_assignments, group_assignments)
        }
        
        return assignments, info
    
    def _compute_hierarchy_depth(self, base_assignments, group_assignments):
        """Measure the depth/quality of hierarchy"""
        # Entropy reduction from base to group level
        base_entropy = -torch.sum(base_assignments * torch.log(base_assignments + 1e-10), dim=-1)
        group_entropy = -torch.sum(group_assignments * torch.log(group_assignments + 1e-10), dim=-1)
        
        entropy_reduction = (base_entropy - group_entropy).mean()
        
        return {
            'entropy_reduction': entropy_reduction,
            'base_entropy': base_entropy.mean(),
            'group_entropy': group_entropy.mean()
        }


def test_hdp_components():
    """Test HDP components"""
    print("Testing HDP Components...")
    
    # Test dimensions
    B, T, D = 4, 100, 64  # Batch, Time, Features
    
    # Test StickBreaking
    sb = StickBreaking(max_clusters=20, alpha=1.0)
    weights = sb(batch_size=B)
    assert weights.shape == (B, 20)
    assert torch.allclose(weights.sum(dim=1), torch.ones(B), atol=1e-5)
    print(f"✓ StickBreaking: weights sum to 1")
    
    # Test HDPLayer
    hdp = HDPLayer(input_dim=D, max_clusters=20)
    
    # Test with 2D input
    z_2d = torch.randn(B, D)
    assignments_2d, info_2d = hdp(z_2d)
    assert assignments_2d.shape == (B, 20)
    print(f"✓ HDP 2D: active clusters={info_2d['active_clusters']}, entropy={info_2d['entropy']:.2f}")
    
    # Test with 3D input (temporal)
    z_3d = torch.randn(B, T, D)
    assignments_3d, info_3d = hdp(z_3d)
    assert assignments_3d.shape == (B, T, 20)
    print(f"✓ HDP 3D: shape={assignments_3d.shape}, active={info_3d['active_clusters']}")
    
    # Test HierarchicalHDP
    h_hdp = HierarchicalHDP(
        input_dim=D,
        max_base_clusters=20,
        max_group_clusters=10
    )
    
    assignments, h_info = h_hdp(z_3d)
    assert assignments['base'].shape == (B, T, 20)
    assert assignments['group'].shape == (B, T, 10)
    assert assignments['joint'].shape == (B, T, 20, 10)
    
    print(f"✓ Hierarchical HDP: base_clusters={h_info['base_info']['active_clusters']}, "
          f"group_clusters={h_info['group_info']['active_clusters']}")
    print(f"  Entropy reduction: {h_info['hierarchy_depth']['entropy_reduction']:.3f}")
    
    # Test gradient flow
    loss = assignments_3d.mean()
    loss.backward()
    print("✓ Gradient flow verified")
    
    # Test temperature annealing
    initial_temp = hdp.current_temperature.item()
    new_temp = hdp.anneal_temperature(step=50, total_steps=100, min_temp=0.1)
    assert new_temp < initial_temp
    print(f"✓ Temperature annealing: {initial_temp:.2f} → {new_temp:.2f}")
    
    print("\nAll HDP tests passed!")
    return True


if __name__ == "__main__":
    test_hdp_components()