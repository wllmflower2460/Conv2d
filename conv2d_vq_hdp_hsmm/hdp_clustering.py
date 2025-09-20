"""
Hierarchical Dirichlet Process (HDP) clustering implementation.
Uses Conv2d operations for stick-breaking process and cluster assignments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HDPClustering(nn.Module):
    """
    HDP clustering with stick-breaking construction.
    
    Args:
        input_dim: Input feature dimension
        max_clusters: Maximum number of clusters (20)
        alpha: Concentration parameter for cluster-level DP
        gamma: Concentration parameter for global-level DP
    """
    
    def __init__(self, input_dim, max_clusters=20, alpha=1.0, gamma=1.0):
        super(HDPClustering, self).__init__()
        
        self.input_dim = input_dim
        self.max_clusters = max_clusters
        self.alpha = alpha
        self.gamma = gamma
        
        # Stick-breaking weights using Conv2d (1x1 convolutions)
        self.stick_weights = nn.Conv2d(input_dim, max_clusters, kernel_size=1)
        
        # Cluster centers using Conv2d
        self.cluster_centers = nn.Conv2d(input_dim, max_clusters * input_dim, kernel_size=1)
        
        # Concentration parameters
        self.register_buffer('alpha_param', torch.tensor(alpha))
        self.register_buffer('gamma_param', torch.tensor(gamma))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize HDP parameters."""
        # Initialize stick weights to encourage sparsity
        nn.init.normal_(self.stick_weights.weight, 0, 0.1)
        nn.init.zeros_(self.stick_weights.bias)
        
        # Initialize cluster centers
        nn.init.xavier_uniform_(self.cluster_centers.weight)
        nn.init.zeros_(self.cluster_centers.bias)
    
    def _stick_breaking(self, logits):
        """
        Stick-breaking process to compute cluster probabilities.
        
        Args:
            logits: Raw stick-breaking logits (B, max_clusters, H, W)
            
        Returns:
            probs: Cluster probabilities (B, max_clusters, H, W)
        """
        # Convert logits to stick proportions using sigmoid
        sticks = torch.sigmoid(logits)  # (B, max_clusters, H, W)
        
        # Stick-breaking construction
        probs = []
        remaining = torch.ones_like(sticks[:, 0:1])  # (B, 1, H, W)
        
        for k in range(self.max_clusters):
            if k == self.max_clusters - 1:
                # Last stick gets all remaining probability
                prob_k = remaining
            else:
                prob_k = sticks[:, k:k+1] * remaining
                remaining = remaining * (1 - sticks[:, k:k+1])
            
            probs.append(prob_k)
        
        probs = torch.cat(probs, dim=1)  # (B, max_clusters, H, W)
        return probs
    
    def forward(self, features):
        """
        Forward pass through HDP clustering.
        
        Args:
            features: Input features (B, input_dim, H, W)
            
        Returns:
            cluster_assignments: Soft cluster assignments (B, max_clusters, H, W)
            cluster_centers: Cluster center representations (B, max_clusters, input_dim, H, W)
            kl_loss: KL divergence loss for regularization
        """
        B, C, H, W = features.shape
        
        # Compute stick-breaking logits
        stick_logits = self.stick_weights(features)  # (B, max_clusters, H, W)
        
        # Apply stick-breaking process
        cluster_probs = self._stick_breaking(stick_logits)  # (B, max_clusters, H, W)
        
        # Compute cluster centers
        center_features = self.cluster_centers(features)  # (B, max_clusters * input_dim, H, W)
        center_features = center_features.view(B, self.max_clusters, self.input_dim, H, W)
        
        # Compute assignment probabilities using distance to centers
        # Expand features for broadcasting
        features_expanded = features.unsqueeze(1)  # (B, 1, input_dim, H, W)
        
        # Compute distances (negative log-likelihood)
        distances = torch.sum((features_expanded - center_features) ** 2, dim=2)  # (B, max_clusters, H, W)
        
        # Convert distances to probabilities
        assignment_logits = -distances / (2.0 * 1.0)  # Assume unit variance
        assignment_probs = F.softmax(assignment_logits, dim=1)
        
        # Combine with stick-breaking probabilities
        cluster_assignments = cluster_probs * assignment_probs
        
        # Normalize to ensure probabilities sum to 1
        cluster_assignments = cluster_assignments / (cluster_assignments.sum(dim=1, keepdim=True) + 1e-10)
        
        # Compute KL divergence loss for regularization
        kl_loss = self._compute_kl_loss(cluster_probs, assignment_probs)
        
        return cluster_assignments, center_features, kl_loss
    
    def _compute_kl_loss(self, stick_probs, assignment_probs):
        """
        Compute KL divergence loss for HDP regularization.
        
        Args:
            stick_probs: Stick-breaking probabilities
            assignment_probs: Assignment probabilities
            
        Returns:
            kl_loss: KL divergence loss
        """
        # Prior probabilities (uniform for simplicity)
        uniform_prior = torch.ones_like(stick_probs) / self.max_clusters
        
        # KL divergence between stick probabilities and uniform prior
        kl_stick = F.kl_div(torch.log(stick_probs + 1e-10), uniform_prior, reduction='batchmean')
        
        # KL divergence between assignment probabilities and uniform prior
        kl_assign = F.kl_div(torch.log(assignment_probs + 1e-10), uniform_prior, reduction='batchmean')
        
        return kl_stick + kl_assign
    
    def get_cluster_statistics(self, cluster_assignments):
        """
        Get statistics about cluster usage.
        
        Args:
            cluster_assignments: Cluster assignment probabilities
            
        Returns:
            stats: Dictionary with cluster statistics
        """
        with torch.no_grad():
            # Average cluster probabilities across batch and spatial dimensions
            avg_probs = cluster_assignments.mean(dim=[0, 2, 3])  # (max_clusters,)
            
            # Number of active clusters (probability > threshold)
            active_clusters = (avg_probs > 0.01).sum().item()
            
            # Entropy of cluster distribution
            entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10)).item()
            
            return {
                'active_clusters': active_clusters,
                'entropy': entropy,
                'cluster_probs': avg_probs.cpu().numpy(),
                'max_prob': avg_probs.max().item(),
                'min_prob': avg_probs.min().item()
            }