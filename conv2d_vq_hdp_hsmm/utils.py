"""
Utility functions for monitoring, visualization, and analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class PerplexityMonitor:
    """Monitor VQ codebook perplexity over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.perplexity_history = []
        self.utilization_history = []
        self.active_codes_history = []
    
    def update(self, perplexity: float, utilization: float, active_codes: int):
        """Update monitoring statistics."""
        self.perplexity_history.append(perplexity)
        self.utilization_history.append(utilization)
        self.active_codes_history.append(active_codes)
        
        # Keep only recent history
        if len(self.perplexity_history) > self.window_size:
            self.perplexity_history.pop(0)
            self.utilization_history.pop(0)
            self.active_codes_history.pop(0)
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        if not self.perplexity_history:
            return {}
        
        return {
            'avg_perplexity': np.mean(self.perplexity_history),
            'std_perplexity': np.std(self.perplexity_history),
            'avg_utilization': np.mean(self.utilization_history),
            'avg_active_codes': np.mean(self.active_codes_history),
            'current_perplexity': self.perplexity_history[-1],
            'trend_perplexity': self._compute_trend(self.perplexity_history)
        }
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction."""
        if len(values) < 2:
            return 'stable'
        
        recent = np.mean(values[-10:]) if len(values) >= 10 else values[-1]
        older = np.mean(values[:-10]) if len(values) >= 20 else values[0]
        
        if recent > older * 1.05:
            return 'increasing'
        elif recent < older * 0.95:
            return 'decreasing'
        else:
            return 'stable'


class ClusterMonitor:
    """Monitor HDP cluster assignments and evolution."""
    
    def __init__(self, max_clusters: int = 20):
        self.max_clusters = max_clusters
        self.cluster_history = []
        self.entropy_history = []
    
    def update(self, cluster_assignments: np.ndarray):
        """Update cluster monitoring."""
        # Average over spatial dimensions
        avg_assignments = cluster_assignments.mean(axis=(0, 2, 3))
        
        # Compute active clusters
        active_clusters = (avg_assignments > 0.01).sum()
        
        # Compute entropy
        entropy = -np.sum(avg_assignments * np.log(avg_assignments + 1e-10))
        
        self.cluster_history.append({
            'assignments': avg_assignments,
            'active_clusters': active_clusters,
            'entropy': entropy
        })
        
        self.entropy_history.append(entropy)
    
    def get_stats(self) -> Dict:
        """Get current cluster statistics."""
        if not self.cluster_history:
            return {}
        
        recent = self.cluster_history[-1]
        
        return {
            'active_clusters': recent['active_clusters'],
            'entropy': recent['entropy'],
            'avg_entropy': np.mean(self.entropy_history),
            'cluster_distribution': recent['assignments'],
            'dominant_cluster': np.argmax(recent['assignments']),
            'cluster_balance': self._compute_balance(recent['assignments'])
        }
    
    def _compute_balance(self, assignments: np.ndarray) -> float:
        """Compute cluster balance (0=imbalanced, 1=perfectly balanced)."""
        active_mask = assignments > 0.01
        if active_mask.sum() <= 1:
            return 0.0
        
        active_assignments = assignments[active_mask]
        uniform_prob = 1.0 / len(active_assignments)
        
        # KL divergence from uniform distribution
        kl_div = np.sum(active_assignments * np.log(active_assignments / uniform_prob + 1e-10))
        
        # Convert to balance score (lower KL = higher balance)
        max_kl = np.log(len(active_assignments))
        balance = 1.0 - (kl_div / max_kl)
        
        return max(0.0, balance)


class StateTransitionAnalyzer:
    """Analyze HSMM state transitions and patterns."""
    
    def __init__(self, num_states: int):
        self.num_states = num_states
        self.transition_counts = np.zeros((num_states, num_states))
        self.state_durations = {i: [] for i in range(num_states)}
        self.sequence_history = []
    
    def update(self, state_sequence: np.ndarray):
        """Update with new state sequence."""
        # state_sequence shape: (B, T, H, W)
        # Average over spatial dimensions and batch
        avg_sequence = state_sequence.mean(axis=(0, 2, 3)).astype(int)
        
        self.sequence_history.append(avg_sequence)
        
        # Count transitions
        for t in range(len(avg_sequence) - 1):
            current_state = avg_sequence[t]
            next_state = avg_sequence[t + 1]
            self.transition_counts[current_state, next_state] += 1
        
        # Compute state durations
        current_state = avg_sequence[0]
        duration = 1
        
        for t in range(1, len(avg_sequence)):
            if avg_sequence[t] == current_state:
                duration += 1
            else:
                self.state_durations[current_state].append(duration)
                current_state = avg_sequence[t]
                duration = 1
        
        # Add final duration
        self.state_durations[current_state].append(duration)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get normalized transition matrix."""
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        return np.divide(self.transition_counts, row_sums, 
                        out=np.zeros_like(self.transition_counts), 
                        where=row_sums!=0)
    
    def get_duration_stats(self) -> Dict:
        """Get state duration statistics."""
        stats = {}
        for state in range(self.num_states):
            durations = self.state_durations[state]
            if durations:
                stats[f'state_{state}'] = {
                    'mean_duration': np.mean(durations),
                    'std_duration': np.std(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'visits': len(durations)
                }
            else:
                stats[f'state_{state}'] = {
                    'mean_duration': 0.0,
                    'std_duration': 0.0,
                    'min_duration': 0,
                    'max_duration': 0,
                    'visits': 0
                }
        
        return stats
    
    def detect_patterns(self) -> Dict:
        """Detect common behavioral patterns."""
        if len(self.sequence_history) < 2:
            return {}
        
        # Concatenate all sequences
        full_sequence = np.concatenate(self.sequence_history)
        
        # Find common subsequences of length 3
        pattern_counts = {}
        for i in range(len(full_sequence) - 2):
            pattern = tuple(full_sequence[i:i+3])
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Get most common patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        top_patterns = sorted_patterns[:5]  # Top 5 patterns
        
        return {
            'common_patterns': top_patterns,
            'total_patterns': len(pattern_counts),
            'pattern_diversity': len(pattern_counts) / max(1, len(full_sequence) - 2)
        }


class EdgeDeploymentAnalyzer:
    """Analyze model for edge deployment compatibility."""
    
    @staticmethod
    def analyze_model(model: torch.nn.Module) -> Dict:
        """Analyze model for edge deployment."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Memory estimation (rough)
        param_memory = total_params * 4  # 4 bytes per float32
        
        # Count layer types
        layer_counts = {}
        for name, module in model.named_modules():
            layer_type = type(module).__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        # Check for unsupported operations
        unsupported_ops = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.GroupNorm)):
                unsupported_ops.append(f"{name}: {type(module).__name__}")
            elif hasattr(module, 'forward') and 'softmax' in str(module.forward).lower():
                unsupported_ops.append(f"{name}: contains softmax")
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_memory_mb': param_memory / (1024 * 1024),
            'layer_counts': layer_counts,
            'unsupported_operations': unsupported_ops,
            'edge_ready': len(unsupported_ops) == 0,
            'complexity_score': total_params / 1e6  # Parameters in millions
        }
    
    @staticmethod
    def benchmark_inference(model: torch.nn.Module, input_shape: Tuple[int, ...], 
                          device: str = 'cpu', num_runs: int = 100) -> Dict:
        """Benchmark model inference speed."""
        model.eval()
        model = model.to(device)
        
        # Warmup
        dummy_input = torch.randn(input_shape).to(device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        if device == 'cuda':
            torch.cuda.synchronize()
        
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            'mean_inference_time_ms': np.mean(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000,
            'min_inference_time_ms': np.min(times) * 1000,
            'max_inference_time_ms': np.max(times) * 1000,
            'throughput_fps': 1.0 / np.mean(times),
            'device': device
        }


def create_monitoring_suite(model: torch.nn.Module) -> Dict:
    """Create a complete monitoring suite for the model."""
    
    # Extract model parameters
    vq_layer = None
    hdp_layer = None
    hsmm_layer = None
    
    for name, module in model.named_modules():
        if hasattr(module, 'num_embeddings'):  # VQ layer
            vq_layer = module
        elif hasattr(module, 'max_clusters'):  # HDP layer
            hdp_layer = module
        elif hasattr(module, 'num_states'):  # HSMM layer
            hsmm_layer = module
    
    monitors = {
        'perplexity': PerplexityMonitor(),
        'clusters': ClusterMonitor(hdp_layer.max_clusters if hdp_layer else 20),
        'transitions': StateTransitionAnalyzer(hsmm_layer.num_states if hsmm_layer else 10),
        'edge_analysis': EdgeDeploymentAnalyzer.analyze_model(model)
    }
    
    return monitors


def update_monitors(monitors: Dict, model_outputs: Dict):
    """Update all monitors with new model outputs."""
    
    # Update perplexity monitor
    if 'perplexity' in model_outputs:
        # Get VQ stats from model if available
        vq_stats = {'utilization': 0.8, 'active_codes': 400}  # Placeholder
        monitors['perplexity'].update(
            model_outputs['perplexity'].item(),
            vq_stats['utilization'],
            vq_stats['active_codes']
        )
    
    # Update cluster monitor
    if 'cluster_assignments' in model_outputs:
        cluster_assignments = model_outputs['cluster_assignments'].detach().cpu().numpy()
        monitors['clusters'].update(cluster_assignments)
    
    # Update transition monitor
    if 'state_probs' in model_outputs:
        state_sequence = model_outputs['state_probs'].argmax(dim=1).detach().cpu().numpy()
        monitors['transitions'].update(state_sequence)


def get_monitoring_report(monitors: Dict) -> Dict:
    """Generate comprehensive monitoring report."""
    
    report = {
        'timestamp': torch.cuda.current_device() if torch.cuda.is_available() else 'cpu',
        'perplexity_stats': monitors['perplexity'].get_stats(),
        'cluster_stats': monitors['clusters'].get_stats(),
        'transition_matrix': monitors['transitions'].get_transition_matrix().tolist(),
        'duration_stats': monitors['transitions'].get_duration_stats(),
        'behavioral_patterns': monitors['transitions'].detect_patterns(),
        'edge_analysis': monitors['edge_analysis']
    }
    
    return report