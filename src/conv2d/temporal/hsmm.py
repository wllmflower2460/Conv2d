"""Hidden Semi-Markov Model policy for temporal smoothing.

Implements HSMM-based smoothing with explicit duration modeling
for realistic behavioral persistence.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from conv2d.temporal.interface import TemporalConfig, TemporalPolicy

logger = logging.getLogger(__name__)


class HSMMPolicy(TemporalPolicy):
    """HSMM-based temporal smoothing with duration modeling.
    
    Models motif durations explicitly using:
    - Geometric or Poisson duration distributions
    - Forward-backward algorithm for smoothing
    - Viterbi decoding for MAP sequence
    """
    
    def __init__(self, config: TemporalConfig):
        """Initialize HSMM policy.
        
        Args:
            config: Temporal configuration
        """
        super().__init__(config)
        
        # HSMM parameters
        self.duration_type = config.extra_params.get("duration_type", "geometric")
        self.mean_duration = config.extra_params.get("mean_duration", 10)
        self.max_duration = config.extra_params.get("max_duration", 100)
        self.use_viterbi = config.extra_params.get("use_viterbi", True)
        
        # Initialize duration distributions
        self.duration_probs_: Optional[NDArray[np.float32]] = None
        self.transition_matrix_: Optional[NDArray[np.float32]] = None
        
        logger.info(
            f"HSMMPolicy: duration={self.duration_type}, "
            f"mean={self.mean_duration}, max={self.max_duration}, "
            f"viterbi={self.use_viterbi}"
        )
    
    def smooth_sequence(
        self, 
        motifs: NDArray[np.int32],
        confidences: Optional[NDArray[np.float32]] = None,
    ) -> NDArray[np.int32]:
        """Apply HSMM-based smoothing.
        
        Args:
            motifs: Raw motif predictions (B, T) or (T,)
            confidences: Optional emission probabilities (B, T, K)
            
        Returns:
            Smoothed motif sequence
        """
        # Handle batch dimension
        squeeze_output = False
        if motifs.ndim == 1:
            motifs = motifs[np.newaxis, :]
            squeeze_output = True
            if confidences is not None:
                if confidences.ndim == 2:
                    confidences = confidences[np.newaxis, :]
        
        B, T = motifs.shape
        
        # Determine number of states
        K = int(np.max(motifs) + 1)
        
        # Initialize model parameters if needed
        if self.duration_probs_ is None:
            self._initialize_model(K)
        
        smoothed = np.zeros_like(motifs)
        
        for b in range(B):
            # Get sequence
            seq = motifs[b]
            
            # Convert to emission probabilities if needed
            if confidences is not None and confidences.ndim == 3:
                emissions = confidences[b]  # (T, K)
            else:
                # Create one-hot emissions from predictions
                emissions = np.zeros((T, K), dtype=np.float32)
                for t in range(T):
                    emissions[t, seq[t]] = 1.0
                
                # Add small noise for numerical stability
                emissions = emissions * 0.9 + 0.1 / K
            
            # Apply HSMM smoothing
            if self.use_viterbi:
                smoothed[b] = self._viterbi_decode(emissions)
            else:
                smoothed[b] = self._forward_backward_decode(emissions)
            
            # Enforce minimum dwell as post-processing
            smoothed[b] = self._enforce_min_dwell(smoothed[b])
        
        if squeeze_output:
            smoothed = smoothed.squeeze(0)
        
        return smoothed
    
    def _initialize_model(self, K: int) -> None:
        """Initialize HSMM parameters.
        
        Args:
            K: Number of motif states
        """
        # Initialize duration distributions
        self.duration_probs_ = self._create_duration_distribution(K)
        
        # Initialize transition matrix (uniform with self-loop penalty)
        self.transition_matrix_ = np.ones((K, K), dtype=np.float32) / K
        np.fill_diagonal(self.transition_matrix_, 0.1 / K)  # Discourage immediate return
        
        # Normalize rows
        self.transition_matrix_ = (
            self.transition_matrix_ / 
            self.transition_matrix_.sum(axis=1, keepdims=True)
        )
        
        logger.debug(f"Initialized HSMM with {K} states")
    
    def _create_duration_distribution(
        self, 
        K: int
    ) -> NDArray[np.float32]:
        """Create duration probability distributions.
        
        Args:
            K: Number of states
            
        Returns:
            Duration probabilities (K, max_duration)
        """
        probs = np.zeros((K, self.max_duration), dtype=np.float32)
        
        for k in range(K):
            if self.duration_type == "geometric":
                # Geometric distribution
                p = 1.0 / self.mean_duration
                for d in range(self.max_duration):
                    probs[k, d] = p * (1 - p) ** d
                    
            elif self.duration_type == "poisson":
                # Poisson distribution
                from scipy.stats import poisson
                probs[k, :] = poisson.pmf(
                    np.arange(self.max_duration), 
                    mu=self.mean_duration
                )
                
            elif self.duration_type == "gaussian":
                # Truncated Gaussian
                from scipy.stats import norm
                x = np.arange(self.max_duration)
                probs[k, :] = norm.pdf(
                    x, 
                    loc=self.mean_duration,
                    scale=self.mean_duration / 3
                )
            
            # Enforce minimum duration
            if self.config.min_dwell > 1:
                probs[k, :self.config.min_dwell-1] = 0
            
            # Normalize
            probs[k] = probs[k] / probs[k].sum()
        
        return probs
    
    def _viterbi_decode(
        self, 
        emissions: NDArray[np.float32]
    ) -> NDArray[np.int32]:
        """Viterbi decoding for MAP sequence.
        
        Args:
            emissions: Emission probabilities (T, K)
            
        Returns:
            Most likely state sequence
        """
        T, K = emissions.shape
        
        # Initialize Viterbi tables
        delta = np.zeros((T, K), dtype=np.float32)
        psi = np.zeros((T, K), dtype=np.int32)
        
        # Initialize with uniform prior
        delta[0] = np.log(emissions[0] + 1e-10) + np.log(1.0 / K)
        
        # Forward pass
        for t in range(1, T):
            for j in range(K):
                # Consider all previous states
                scores = delta[t-1] + np.log(self.transition_matrix_[:, j] + 1e-10)
                psi[t, j] = np.argmax(scores)
                delta[t, j] = np.max(scores) + np.log(emissions[t, j] + 1e-10)
        
        # Backward pass - traceback
        path = np.zeros(T, dtype=np.int32)
        path[T-1] = np.argmax(delta[T-1])
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path
    
    def _forward_backward_decode(
        self, 
        emissions: NDArray[np.float32]
    ) -> NDArray[np.int32]:
        """Forward-backward algorithm for smoothing.
        
        Args:
            emissions: Emission probabilities (T, K)
            
        Returns:
            Smoothed state sequence
        """
        T, K = emissions.shape
        
        # Forward pass
        alpha = np.zeros((T, K), dtype=np.float32)
        alpha[0] = emissions[0] / K  # Uniform prior
        
        for t in range(1, T):
            for j in range(K):
                alpha[t, j] = emissions[t, j] * np.sum(
                    alpha[t-1] * self.transition_matrix_[:, j]
                )
            # Normalize to prevent underflow
            alpha[t] = alpha[t] / (alpha[t].sum() + 1e-10)
        
        # Backward pass
        beta = np.zeros((T, K), dtype=np.float32)
        beta[T-1] = 1.0
        
        for t in range(T-2, -1, -1):
            for i in range(K):
                beta[t, i] = np.sum(
                    self.transition_matrix_[i] * emissions[t+1] * beta[t+1]
                )
            beta[t] = beta[t] / (beta[t].sum() + 1e-10)
        
        # Compute posterior
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        # Decode as MAP estimate
        path = np.argmax(gamma, axis=1).astype(np.int32)
        
        return path
    
    def fit_parameters(
        self,
        sequences: NDArray[np.int32],
        n_iter: int = 10,
    ) -> None:
        """Fit HSMM parameters from observed sequences.
        
        Args:
            sequences: Observed motif sequences (N, T)
            n_iter: Number of EM iterations
        """
        N, T = sequences.shape
        K = int(np.max(sequences) + 1)
        
        # Initialize if needed
        if self.transition_matrix_ is None:
            self._initialize_model(K)
        
        logger.info(f"Fitting HSMM parameters from {N} sequences")
        
        for iteration in range(n_iter):
            # E-step: collect statistics
            trans_counts = np.zeros((K, K), dtype=np.float32)
            duration_counts = np.zeros((K, self.max_duration), dtype=np.float32)
            
            for n in range(N):
                seq = sequences[n]
                
                # Count transitions
                for t in range(1, T):
                    if seq[t] != seq[t-1]:
                        trans_counts[seq[t-1], seq[t]] += 1
                
                # Count durations
                dwells = self._extract_durations(seq)
                for state, duration in dwells:
                    if duration < self.max_duration:
                        duration_counts[state, duration] += 1
            
            # M-step: update parameters
            # Update transition matrix
            for i in range(K):
                if trans_counts[i].sum() > 0:
                    self.transition_matrix_[i] = trans_counts[i] / trans_counts[i].sum()
            
            # Update duration distributions
            for k in range(K):
                if duration_counts[k].sum() > 0:
                    self.duration_probs_[k] = duration_counts[k] / duration_counts[k].sum()
        
        logger.info("HSMM parameter fitting complete")
    
    def _extract_durations(
        self, 
        sequence: NDArray[np.int32]
    ) -> list[Tuple[int, int]]:
        """Extract state durations from sequence.
        
        Args:
            sequence: Motif sequence
            
        Returns:
            List of (state, duration) tuples
        """
        if len(sequence) == 0:
            return []
        
        durations = []
        current_state = sequence[0]
        current_duration = 1
        
        for t in range(1, len(sequence)):
            if sequence[t] == current_state:
                current_duration += 1
            else:
                durations.append((current_state, current_duration))
                current_state = sequence[t]
                current_duration = 1
        
        # Add final segment
        durations.append((current_state, current_duration))
        
        return durations