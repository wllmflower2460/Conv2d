"""
Hidden Semi-Markov Model (HSMM) Components for Conv2d-VQ-HDP-HSMM
Models temporal dynamics with explicit duration distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List


class DurationModel(nn.Module):
    """
    Duration distribution model for HSMM states
    Supports multiple parametric distributions
    """
    
    def __init__(
        self,
        num_states: int,
        max_duration: int = 100,
        distribution: str = 'negative_binomial',
        min_duration: int = 1
    ):
        super().__init__()
        
        self.num_states = num_states
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.distribution = distribution
        
        if distribution == 'negative_binomial':
            # Parameters: r (number of failures) and p (success probability)
            self.log_r = nn.Parameter(torch.ones(num_states))  # log(r) for stability
            self.logit_p = nn.Parameter(torch.zeros(num_states))  # logit(p) for [0,1] constraint
            
        elif distribution == 'poisson':
            # Parameter: lambda (rate)
            self.log_lambda = nn.Parameter(torch.ones(num_states))
            
        elif distribution == 'gaussian':
            # Parameters: mean and log(std)
            self.duration_mean = nn.Parameter(torch.ones(num_states) * 10)
            self.duration_log_std = nn.Parameter(torch.zeros(num_states))
            
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def forward(self, state_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get duration probabilities for given states
        
        Args:
            state_idx: State indices (B,) or (B, T)
        
        Returns:
            duration_probs: Probabilities for each duration (B, max_duration)
            expected_duration: Expected duration for each state (B,)
        """
        batch_shape = state_idx.shape
        state_flat = state_idx.flatten()
        
        if self.distribution == 'negative_binomial':
            r = torch.exp(self.log_r[state_flat])
            p = torch.sigmoid(self.logit_p[state_flat])
            
            # Compute PMF for durations 1 to max_duration
            durations = torch.arange(1, self.max_duration + 1, device=r.device).float()
            durations = durations.unsqueeze(0).expand(len(state_flat), -1)
            
            # Negative binomial PMF: P(X=k) = C(k+r-1, k) * p^r * (1-p)^k
            # Using log-space for numerical stability
            log_pmf = (
                torch.lgamma(durations + r.unsqueeze(1)) -
                torch.lgamma(durations + 1) -
                torch.lgamma(r.unsqueeze(1)) +
                r.unsqueeze(1) * torch.log(p.unsqueeze(1) + 1e-10) +
                durations * torch.log(1 - p.unsqueeze(1) + 1e-10)
            )
            duration_probs = torch.exp(log_pmf)
            # Normalize to ensure proper probability distribution
            duration_probs = duration_probs / duration_probs.sum(dim=1, keepdim=True)
            
            # Expected duration: r(1-p)/p
            expected_duration = r * (1 - p) / p
            
        elif self.distribution == 'poisson':
            lambda_param = torch.exp(self.log_lambda[state_flat])
            
            durations = torch.arange(1, self.max_duration + 1, device=lambda_param.device).float()
            durations = durations.unsqueeze(0).expand(len(state_flat), -1)
            
            # Poisson PMF: P(X=k) = λ^k * e^(-λ) / k!
            log_pmf = (
                durations * torch.log(lambda_param.unsqueeze(1) + 1e-10) -
                lambda_param.unsqueeze(1) -
                torch.lgamma(durations + 1)
            )
            duration_probs = torch.exp(log_pmf)
            # Normalize to ensure proper probability distribution
            duration_probs = duration_probs / duration_probs.sum(dim=1, keepdim=True)
            expected_duration = lambda_param
            
        elif self.distribution == 'gaussian':
            mean = self.duration_mean[state_flat]
            std = torch.exp(self.duration_log_std[state_flat])
            
            durations = torch.arange(1, self.max_duration + 1, device=mean.device).float()
            durations = durations.unsqueeze(0).expand(len(state_flat), -1)
            
            # Discretized Gaussian
            z = (durations - mean.unsqueeze(1)) / std.unsqueeze(1)
            duration_probs = torch.exp(-0.5 * z**2) / (std.unsqueeze(1) * np.sqrt(2 * np.pi))
            
            # Normalize
            duration_probs = duration_probs / duration_probs.sum(dim=1, keepdim=True)
            expected_duration = mean
        
        # Apply minimum duration constraint
        if self.min_duration > 1:
            duration_probs[:, :self.min_duration-1] = 0
            duration_probs = duration_probs / duration_probs.sum(dim=1, keepdim=True)
        
        # Reshape to original batch shape
        duration_probs = duration_probs.reshape(*batch_shape, self.max_duration)
        expected_duration = expected_duration.reshape(*batch_shape)
        
        return duration_probs, expected_duration


class HSMMTransitions(nn.Module):
    """
    State transition model for HSMM
    Includes self-transition prohibition (handled by duration model)
    """
    
    def __init__(
        self,
        num_states: int,
        input_dim: Optional[int] = None,
        use_input_dependent: bool = False
    ):
        super().__init__()
        
        self.num_states = num_states
        self.use_input_dependent = use_input_dependent
        
        if use_input_dependent and input_dim is not None:
            # Input-dependent transitions
            self.transition_net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_states * num_states)
            )
        else:
            # Fixed transition matrix
            self.log_trans_matrix = nn.Parameter(
                torch.randn(num_states, num_states) * 0.1
            )
            
            # Mask for self-transitions (will be handled by duration)
            self.register_buffer(
                'self_trans_mask',
                1.0 - torch.eye(num_states)
            )
    
    def forward(self, input_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get transition probabilities
        
        Args:
            input_features: Optional input for input-dependent transitions (B, D)
        
        Returns:
            trans_probs: Transition probabilities (B, S, S) or (S, S)
        """
        if self.use_input_dependent and input_features is not None:
            B = input_features.shape[0]
            logits = self.transition_net(input_features)
            logits = logits.reshape(B, self.num_states, self.num_states)
            
            # Create self-transition mask
            mask = 1.0 - torch.eye(self.num_states, device=logits.device)
            mask = mask.unsqueeze(0).expand(B, -1, -1)
            logits = logits * mask - 1e10 * (1 - mask)
            
            trans_probs = F.softmax(logits, dim=-1)
            
        else:
            # Apply mask to prevent self-transitions
            masked_logits = self.log_trans_matrix * self.self_trans_mask - 1e10 * (1 - self.self_trans_mask)
            trans_probs = F.softmax(masked_logits, dim=-1)
        
        return trans_probs


class HSMM(nn.Module):
    """
    Complete Hidden Semi-Markov Model
    Combines duration modeling with state transitions
    """
    
    def __init__(
        self,
        num_states: int = 10,
        observation_dim: int = 64,
        max_duration: int = 100,
        duration_dist: str = 'negative_binomial',
        use_input_dependent_trans: bool = False
    ):
        super().__init__()
        
        self.num_states = num_states
        self.observation_dim = observation_dim
        self.max_duration = max_duration
        
        # Duration model
        self.duration_model = DurationModel(
            num_states=num_states,
            max_duration=max_duration,
            distribution=duration_dist
        )
        
        # Transition model
        self.transition_model = HSMMTransitions(
            num_states=num_states,
            input_dim=observation_dim if use_input_dependent_trans else None,
            use_input_dependent=use_input_dependent_trans
        )
        
        # Emission model (observation probabilities)
        self.emission_net = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_states)
        )
        
        # Initial state distribution
        self.log_init_probs = nn.Parameter(torch.zeros(num_states))
        
        # State tracking
        self.register_buffer('current_state', torch.zeros(1, dtype=torch.long))
        self.register_buffer('duration_remaining', torch.zeros(1, dtype=torch.long))
        
    def forward(
        self,
        observations: torch.Tensor,
        return_viterbi: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process sequence through HSMM
        
        Args:
            observations: Input observations (B, T, D)
            return_viterbi: Whether to run Viterbi for most likely path
        
        Returns:
            state_probs: State probabilities at each time (B, T, S)
            info: Dictionary with HSMM information
        """
        B, T, D = observations.shape
        device = observations.device
        
        # Get emission probabilities
        emission_logits = self.emission_net(observations)  # (B, T, S)
        emission_probs = F.softmax(emission_logits, dim=-1)
        
        # Get transition probabilities
        trans_probs = self.transition_model(observations.reshape(B*T, D))
        if trans_probs.dim() == 2:
            trans_probs = trans_probs.unsqueeze(0).expand(B, -1, -1)
        else:
            trans_probs = trans_probs.reshape(B, T, self.num_states, self.num_states)
            trans_probs = trans_probs.mean(dim=1)  # Average over time
        
        # Forward-backward algorithm with duration
        alpha, beta, state_probs = self._forward_backward(
            emission_probs, trans_probs
        )
        
        # Viterbi decoding for most likely path
        if return_viterbi:
            viterbi_path = self._viterbi(emission_probs, trans_probs)
        else:
            viterbi_path = None
        
        # Compute expected durations
        state_marginals = state_probs.mean(dim=1)  # (B, S)
        _, expected_durations = self.duration_model(
            torch.arange(self.num_states, device=device)
        )
        
        info = {
            'emission_probs': emission_probs,
            'trans_probs': trans_probs,
            'expected_durations': expected_durations,
            'state_marginals': state_marginals,
            'viterbi_path': viterbi_path,
            'log_likelihood': self._compute_log_likelihood(alpha)
        }
        
        return state_probs, info
    
    def _forward_backward(
        self,
        emission_probs: torch.Tensor,
        trans_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward-backward algorithm with duration modeling
        Simplified version - full implementation would include duration probabilities
        """
        B, T, S = emission_probs.shape
        device = emission_probs.device
        
        # Initialize forward variables
        log_alpha = torch.zeros(B, T, S, device=device)
        log_alpha[:, 0] = torch.log_softmax(self.log_init_probs, dim=0) + torch.log(emission_probs[:, 0])
        
        # Forward pass
        for t in range(1, T):
            # Standard forward recursion (simplified - full HSMM would include duration)
            log_trans = torch.log(trans_probs + 1e-10)
            log_alpha[:, t] = torch.logsumexp(
                log_alpha[:, t-1].unsqueeze(-1) + log_trans,
                dim=1
            ) + torch.log(emission_probs[:, t] + 1e-10)
        
        # Initialize backward variables
        log_beta = torch.zeros(B, T, S, device=device)
        log_beta[:, -1] = 0  # Log(1) = 0
        
        # Backward pass
        for t in range(T-2, -1, -1):
            log_trans = torch.log(trans_probs + 1e-10)
            log_beta[:, t] = torch.logsumexp(
                log_trans + torch.log(emission_probs[:, t+1] + 1e-10).unsqueeze(1) + 
                log_beta[:, t+1].unsqueeze(1),
                dim=2
            )
        
        # Compute posterior probabilities
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=-1, keepdim=True)
        gamma = torch.exp(log_gamma)
        
        return torch.exp(log_alpha), torch.exp(log_beta), gamma
    
    def _viterbi(
        self,
        emission_probs: torch.Tensor,
        trans_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Viterbi algorithm for most likely state sequence
        Simplified version without duration
        """
        B, T, S = emission_probs.shape
        device = emission_probs.device
        
        # Initialize
        log_delta = torch.zeros(B, T, S, device=device)
        psi = torch.zeros(B, T, S, dtype=torch.long, device=device)
        
        log_delta[:, 0] = torch.log_softmax(self.log_init_probs, dim=0) + torch.log(emission_probs[:, 0] + 1e-10)
        
        # Forward pass
        for t in range(1, T):
            log_trans = torch.log(trans_probs + 1e-10)
            
            # Find max and argmax
            values = log_delta[:, t-1].unsqueeze(-1) + log_trans
            log_delta[:, t], psi[:, t] = torch.max(values, dim=1)
            log_delta[:, t] = log_delta[:, t] + torch.log(emission_probs[:, t] + 1e-10)
        
        # Backward pass - trace back best path
        path = torch.zeros(B, T, dtype=torch.long, device=device)
        path[:, -1] = torch.argmax(log_delta[:, -1], dim=1)
        
        for t in range(T-2, -1, -1):
            path[:, t] = torch.gather(psi[:, t+1], 1, path[:, t+1].unsqueeze(1)).squeeze(1)
        
        return path
    
    def _compute_log_likelihood(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute log likelihood from forward variables"""
        return torch.log(alpha[:, -1].sum(dim=-1) + 1e-10).mean()
    
    def sample_sequence(self, length: int, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a sequence from the HSMM
        
        Returns:
            states: Sampled state sequence (B, T)
            durations: Duration for each state change (B, num_changes)
        """
        device = self.log_init_probs.device
        states = torch.zeros(batch_size, length, dtype=torch.long, device=device)
        durations_list = []
        
        # Sample initial states
        init_probs = F.softmax(self.log_init_probs, dim=0)
        current_states = torch.multinomial(init_probs.expand(batch_size, -1), 1).squeeze(1)
        
        # Sample initial durations
        duration_probs, _ = self.duration_model(current_states)
        current_durations = torch.multinomial(duration_probs, 1).squeeze(1) + 1
        
        t = 0
        while t < length:
            # Fill in current state for its duration
            duration_to_use = torch.minimum(current_durations, torch.tensor(length - t))
            
            for b in range(batch_size):
                states[b, t:t+duration_to_use[b]] = current_states[b]
            
            durations_list.append(duration_to_use.clone())
            t += duration_to_use.max().item()
            
            if t < length:
                # Sample next states
                trans_probs = self.transition_model()
                if trans_probs.dim() == 2:
                    next_state_probs = trans_probs[current_states]
                else:
                    next_state_probs = trans_probs[torch.arange(batch_size), current_states]
                
                current_states = torch.multinomial(next_state_probs, 1).squeeze(1)
                
                # Sample new durations
                duration_probs, _ = self.duration_model(current_states)
                current_durations = torch.multinomial(duration_probs, 1).squeeze(1) + 1
        
        durations = torch.stack(durations_list, dim=1) if durations_list else torch.zeros(batch_size, 0)
        
        return states[:, :length], durations


def test_hsmm_components():
    """Test HSMM components"""
    print("Testing HSMM Components...")
    
    B, T, D, S = 4, 100, 64, 10  # Batch, Time, Dim, States
    
    # Test DurationModel
    print("\n1. Testing DurationModel...")
    for dist in ['negative_binomial', 'poisson', 'gaussian']:
        dur_model = DurationModel(num_states=S, distribution=dist)
        state_idx = torch.randint(0, S, (B,))
        dur_probs, expected_dur = dur_model(state_idx)
        
        assert dur_probs.shape == (B, 100)
        assert expected_dur.shape == (B,)
        assert torch.allclose(dur_probs.sum(dim=1), torch.ones(B), atol=1e-5)
        
        print(f"  ✓ {dist}: expected duration = {expected_dur.mean():.1f}")
    
    # Test HSMMTransitions
    print("\n2. Testing HSMMTransitions...")
    
    # Fixed transitions
    trans_fixed = HSMMTransitions(num_states=S, use_input_dependent=False)
    trans_probs_fixed = trans_fixed()
    assert trans_probs_fixed.shape == (S, S)
    assert torch.allclose(trans_probs_fixed.sum(dim=1), torch.ones(S), atol=1e-5)
    # Check self-transitions are near zero
    assert torch.diag(trans_probs_fixed).max() < 1e-9
    print(f"  ✓ Fixed transitions: shape={trans_probs_fixed.shape}")
    
    # Input-dependent transitions
    trans_input = HSMMTransitions(num_states=S, input_dim=D, use_input_dependent=True)
    input_features = torch.randn(B, D)
    trans_probs_input = trans_input(input_features)
    assert trans_probs_input.shape == (B, S, S)
    print(f"  ✓ Input-dependent transitions: shape={trans_probs_input.shape}")
    
    # Test complete HSMM
    print("\n3. Testing complete HSMM...")
    hsmm = HSMM(
        num_states=S,
        observation_dim=D,
        max_duration=50,
        duration_dist='negative_binomial'
    )
    
    observations = torch.randn(B, T, D)
    state_probs, info = hsmm(observations, return_viterbi=True)
    
    assert state_probs.shape == (B, T, S)
    assert torch.allclose(state_probs.sum(dim=-1), torch.ones(B, T), atol=1e-5)
    assert info['viterbi_path'].shape == (B, T)
    
    print(f"  ✓ Forward-backward: state_probs shape={state_probs.shape}")
    print(f"  ✓ Viterbi path: shape={info['viterbi_path'].shape}")
    print(f"  ✓ Log likelihood: {info['log_likelihood']:.3f}")
    print(f"  ✓ Expected durations: {info['expected_durations'].mean():.1f}")
    
    # Test sequence sampling
    print("\n4. Testing sequence sampling...")
    sampled_states, sampled_durations = hsmm.sample_sequence(length=200, batch_size=2)
    assert sampled_states.shape == (2, 200)
    print(f"  ✓ Sampled sequence: shape={sampled_states.shape}")
    print(f"  ✓ Number of state changes: {(sampled_durations > 0).sum(dim=1).tolist()}")
    
    # Test gradient flow
    loss = state_probs.mean() + info['log_likelihood']
    loss.backward()
    print("\n✓ Gradient flow verified")
    
    print("\nAll HSMM tests passed!")
    return True


if __name__ == "__main__":
    test_hsmm_components()