"""
Hidden Semi-Markov Model (HSMM) implementation for temporal dynamics.
Uses Conv2d operations for forward-backward algorithm and duration modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HSMM(nn.Module):
    """
    Hidden Semi-Markov Model for temporal sequence modeling.
    
    Args:
        num_states: Number of hidden states
        max_duration: Maximum state duration
        feature_dim: Feature dimension
    """
    
    def __init__(self, num_states, max_duration=10, feature_dim=64):
        super(HSMM, self).__init__()
        
        self.num_states = num_states
        self.max_duration = max_duration
        self.feature_dim = feature_dim
        
        # Transition probabilities using Conv2d (1x1)
        self.transition_net = nn.Conv2d(feature_dim, num_states * num_states, kernel_size=1)
        
        # Duration probabilities using Conv2d
        self.duration_net = nn.Conv2d(feature_dim, num_states * max_duration, kernel_size=1)
        
        # Emission probabilities using Conv2d
        self.emission_net = nn.Conv2d(feature_dim, num_states * feature_dim, kernel_size=1)
        
        # Initial state probabilities
        self.initial_state = nn.Parameter(torch.randn(num_states))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize HSMM parameters."""
        # Initialize transition matrix parameters
        with torch.no_grad():
            # Create identity-like transition matrix
            transition_weight = self.transition_net.weight.data  # (num_states * num_states, feature_dim, 1, 1)
            transition_weight.zero_()
            # Set diagonal elements to encourage staying in same state
            for i in range(self.num_states):
                idx = i * self.num_states + i  # diagonal index
                if idx < transition_weight.shape[0]:
                    transition_weight[idx, :, 0, 0] = 1.0
        
        nn.init.zeros_(self.transition_net.bias)
        
        # Initialize duration parameters
        nn.init.normal_(self.duration_net.weight, 0, 0.1)
        nn.init.zeros_(self.duration_net.bias)
        
        # Initialize emission parameters
        nn.init.xavier_uniform_(self.emission_net.weight)
        nn.init.zeros_(self.emission_net.bias)
        
        # Initialize initial state to uniform
        nn.init.constant_(self.initial_state, 1.0 / self.num_states)
    
    def _compute_duration_probs(self, features):
        """
        Compute duration probabilities for each state.
        
        Args:
            features: Input features (B, feature_dim, H, W)
            
        Returns:
            duration_probs: Duration probabilities (B, num_states, max_duration, H, W)
        """
        # Handle different input shapes
        if len(features.shape) == 2:  # (B, feature_dim)
            B, C = features.shape
            H, W = 1, 1
            features = features.view(B, C, H, W)
        elif len(features.shape) == 3:  # (B, feature_dim, T)
            B, C, T = features.shape
            H, W = 1, 1
            features = features.view(B, C, H, W)
        else:
            B, C, H, W = features.shape
        
        # Compute duration logits
        duration_logits = self.duration_net(features)  # (B, num_states * max_duration, H, W)
        duration_logits = duration_logits.view(B, self.num_states, self.max_duration, H, W)
        
        # Apply softmax over duration dimension
        duration_probs = F.softmax(duration_logits, dim=2)
        
        return duration_probs
    
    def _compute_transition_probs(self, features):
        """
        Compute state transition probabilities.
        
        Args:
            features: Input features (B, feature_dim, H, W)
            
        Returns:
            transition_probs: Transition probabilities (B, num_states, num_states, H, W)
        """
        # Handle different input shapes
        if len(features.shape) == 2:  # (B, feature_dim)
            B, C = features.shape
            H, W = 1, 1
            features = features.view(B, C, H, W)
        elif len(features.shape) == 3:  # (B, feature_dim, T)
            B, C, T = features.shape
            H, W = 1, 1
            features = features.view(B, C, H, W)
        else:
            B, C, H, W = features.shape
        
        # Compute transition logits
        transition_logits = self.transition_net(features)  # (B, num_states * num_states, H, W)
        transition_logits = transition_logits.view(B, self.num_states, self.num_states, H, W)
        
        # Apply softmax over next state dimension
        transition_probs = F.softmax(transition_logits, dim=2)
        
        return transition_probs
    
    def _compute_emission_probs(self, features, observations):
        """
        Compute emission probabilities.
        
        Args:
            features: Input features (B, feature_dim, H, W)
            observations: Observed features (B, feature_dim, T, H, W)
            
        Returns:
            emission_probs: Emission probabilities (B, num_states, T, H, W)
        """
        # Handle different input shapes for features
        if len(features.shape) == 2:  # (B, feature_dim)
            B, C = features.shape
            H, W = 1, 1
            features = features.view(B, C, H, W)
        elif len(features.shape) == 3:  # (B, feature_dim, T)
            B, C, T_feat = features.shape
            H, W = 1, 1
            features = features.view(B, C, H, W)
        else:
            B, C, H, W = features.shape
        
        T = observations.shape[2]
        
        # Compute emission parameters
        emission_params = self.emission_net(features)  # (B, num_states * feature_dim, H, W)
        emission_params = emission_params.view(B, self.num_states, self.feature_dim, H, W)
        
        # Compute emission probabilities for each time step
        emission_probs = []
        
        for t in range(T):
            obs_t = observations[:, :, t]  # (B, feature_dim, H, W)
            if len(obs_t.shape) == 3:  # Missing spatial dimensions
                obs_t = obs_t.unsqueeze(-1)  # (B, feature_dim, H, 1)
            if len(obs_t.shape) == 2:  # Missing both spatial dimensions
                obs_t = obs_t.unsqueeze(-1).unsqueeze(-1)  # (B, feature_dim, 1, 1)
            
            obs_t = obs_t.unsqueeze(1)  # (B, 1, feature_dim, H, W)
            
            # Compute log probabilities (assuming Gaussian emissions)
            diff = obs_t - emission_params  # (B, num_states, feature_dim, H, W)
            log_prob = -0.5 * torch.sum(diff ** 2, dim=2)  # (B, num_states, H, W)
            
            emission_probs.append(log_prob)
        
        emission_probs = torch.stack(emission_probs, dim=2)  # (B, num_states, T, H, W)
        
        return emission_probs
    
    def forward_algorithm(self, features, observations):
        """
        Forward algorithm for HSMM.
        
        Args:
            features: Context features (B, feature_dim, H, W)
            observations: Sequence observations (B, feature_dim, T, H, W)
            
        Returns:
            forward_probs: Forward probabilities
            log_likelihood: Log likelihood of the sequence
        """
        # Handle input shapes
        if len(observations.shape) == 4:  # (B, C, T, Feat) -> add spatial dim
            B, C, T, Feat = observations.shape
            observations = observations.unsqueeze(-1)  # (B, C, T, Feat, 1)
        elif len(observations.shape) == 3:  # (B, C, T) -> add spatial dims
            B, C, T = observations.shape
            observations = observations.unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
        
        B, C, T, H, W = observations.shape
        
        # Compute model parameters
        duration_probs = self._compute_duration_probs(features)
        transition_probs = self._compute_transition_probs(features)
        emission_probs = self._compute_emission_probs(features, observations)
        
        # Initialize forward variables
        # alpha[t, s, d] = probability of being in state s for d steps ending at time t
        alpha = torch.zeros(B, T, self.num_states, self.max_duration, H, W, device=features.device)
        
        # Initial probabilities
        initial_probs = torch.softmax(self.initial_state, dim=0).view(1, self.num_states, 1, 1, 1)
        
        # Simplified forward pass (avoiding complex duration modeling for now)
        for t in range(T):
            for s in range(self.num_states):
                if t == 0:
                    # Initial state
                    alpha[:, t, s, 0] = (initial_probs[:, s] * 
                                         emission_probs[:, s, t] * 
                                         duration_probs[:, s, 0])
                else:
                    # Simple transition (not full HSMM for now)
                    prev_state_sum = torch.zeros_like(alpha[:, t, s, 0])
                    for prev_s in range(self.num_states):
                        prev_state_sum += alpha[:, t-1, prev_s, 0] * transition_probs[:, prev_s, s]
                    
                    alpha[:, t, s, 0] = prev_state_sum * emission_probs[:, s, t]
        
        # Compute total probability
        log_likelihood = torch.logsumexp(alpha[:, -1, :, 0].view(B, -1, H, W), dim=1)
        
        return alpha, log_likelihood
    
    def backward_algorithm(self, features, observations, alpha):
        """
        Simplified backward algorithm for HSMM.
        """
        if len(observations.shape) == 4:  # (B, C, T, Feat) -> add spatial dim
            B, C, T, Feat = observations.shape
            observations = observations.unsqueeze(-1)  # (B, C, T, Feat, 1)
        elif len(observations.shape) == 3:  # (B, C, T) -> add spatial dims
            B, C, T = observations.shape
            observations = observations.unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
        
        B, C, T, H, W = observations.shape
        
        # Initialize backward variables (simplified)
        beta = torch.ones_like(alpha)
        
        # Simple gamma calculation
        gamma = alpha * beta
        gamma = gamma / (gamma.sum(dim=[2, 3], keepdim=True) + 1e-10)
        
        return beta, gamma
    
    def forward(self, features, observations):
        """
        Forward pass through HSMM.
        
        Args:
            features: Context features (B, feature_dim, H, W)
            observations: Sequence observations (B, feature_dim, T, H, W)
            
        Returns:
            state_probs: State probability sequence (B, num_states, T, H, W)
            log_likelihood: Log likelihood of the sequence
            duration_probs: Duration probabilities
        """
        # Run forward-backward algorithm
        alpha, log_likelihood = self.forward_algorithm(features, observations)
        beta, gamma = self.backward_algorithm(features, observations, alpha)
        
        # Marginalize over durations to get state probabilities
        state_probs = gamma.sum(dim=3)  # (B, T, num_states, H, W)
        state_probs = state_probs.transpose(1, 2)  # (B, num_states, T, H, W)
        
        # Get duration probabilities
        duration_probs = self._compute_duration_probs(features)
        
        return state_probs, log_likelihood, duration_probs
    
    def viterbi_decode(self, features, observations):
        """
        Viterbi decoding to find most likely state sequence.
        
        Args:
            features: Context features (B, feature_dim, H, W)
            observations: Sequence observations (B, feature_dim, T, H, W)
            
        Returns:
            best_path: Most likely state sequence (B, T, H, W)
        """
        B, C, T, H, W = observations.shape
        
        # Compute model parameters
        duration_probs = self._compute_duration_probs(features)
        transition_probs = self._compute_transition_probs(features)
        emission_probs = self._compute_emission_probs(features, observations)
        
        # Viterbi tables
        viterbi = torch.full((B, T, self.num_states, self.max_duration, H, W), 
                           -float('inf'), device=features.device)
        path = torch.zeros((B, T, self.num_states, self.max_duration, H, W), 
                          dtype=torch.long, device=features.device)
        
        # Initialize
        initial_probs = F.log_softmax(self.initial_state, dim=0).view(1, self.num_states, 1, 1, 1)
        for s in range(self.num_states):
            viterbi[:, 0, s, 0] = (initial_probs[:, s] + 
                                  emission_probs[:, s, 0] + 
                                  torch.log(duration_probs[:, s, 0] + 1e-10))
        
        # Forward pass
        for t in range(1, T):
            for s in range(self.num_states):
                for d in range(self.max_duration):
                    if d == 0:
                        # New state entry
                        for prev_s in range(self.num_states):
                            for prev_d in range(self.max_duration):
                                score = (viterbi[:, t-1, prev_s, prev_d] + 
                                       torch.log(transition_probs[:, prev_s, s] + 1e-10) +
                                       emission_probs[:, s, t] + 
                                       torch.log(duration_probs[:, s, d] + 1e-10))
                                
                                mask = score > viterbi[:, t, s, d]
                                viterbi[:, t, s, d] = torch.where(mask, score, viterbi[:, t, s, d])
                                path[:, t, s, d] = torch.where(mask, prev_s, path[:, t, s, d])
                    else:
                        # Continue in same state
                        if d < self.max_duration:
                            score = viterbi[:, t-1, s, d-1] + emission_probs[:, s, t]
                            viterbi[:, t, s, d] = score
                            path[:, t, s, d] = s
        
        # Backtrack to find best path
        best_path = torch.zeros((B, T, H, W), dtype=torch.long, device=features.device)
        
        # Find best final state
        final_scores = viterbi[:, -1].view(B, -1, H, W)
        best_final = torch.argmax(final_scores, dim=1)
        
        # Convert to state and duration indices
        best_final_state = best_final // self.max_duration
        best_final_duration = best_final % self.max_duration
        
        best_path[:, -1] = best_final_state
        
        # Backtrack
        for t in range(T - 2, -1, -1):
            for b in range(B):
                for h in range(H):
                    for w in range(W):
                        curr_state = int(best_path[b, t+1, h, w])
                        curr_duration = int(best_final_duration[b, h, w]) if t == T - 2 else 0
                        best_path[b, t, h, w] = path[b, t+1, curr_state, curr_duration, h, w]
        
        return best_path