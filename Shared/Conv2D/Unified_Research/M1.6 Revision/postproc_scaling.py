"""
Hailo Post-Quantization Scaling Alignment
Ensures consistency between training FSQ and deployment quantization
"""

import numpy as np
import json
import os
from pathlib import Path


class FSQPostProcessor:
    """
    Post-processor for Hailo deployment with proper feature scaling.
    
    FIXES: Train-serve skew by maintaining consistent standardization
    between training and inference.
    """
    
    def __init__(self, levels, mean=None, std=None, feature_range=(-1, 1)):
        """
        Initialize FSQ post-processor with scaling parameters.
        
        Args:
            levels: List of quantization levels per dimension
            mean: Per-dimension mean from training (None = zeros)
            std: Per-dimension std from training (None = ones)
            feature_range: Expected range after standardization
        """
        self.levels = np.array(levels, dtype=np.int32)
        self.dim = len(levels)
        
        # Initialize or load statistics
        self.mean = np.zeros(self.dim) if mean is None else np.asarray(mean)
        self.std = np.ones(self.dim) if std is None else np.asarray(std)
        
        # Ensure shapes match
        assert len(self.mean) == self.dim, f"Mean dim mismatch: {len(self.mean)} vs {self.dim}"
        assert len(self.std) == self.dim, f"Std dim mismatch: {len(self.std)} vs {self.dim}"
        
        self.feature_range = feature_range
        
        # Precompute quantization boundaries
        self._setup_quantization_boundaries()
    
    def _setup_quantization_boundaries(self):
        """Precompute uniform quantization boundaries for each dimension."""
        self.boundaries = []
        
        for L in self.levels:
            # Uniform steps in feature_range
            min_val, max_val = self.feature_range
            edges = np.linspace(min_val, max_val, L + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            self.boundaries.append({
                'edges': edges,
                'centers': centers,
                'L': L
            })
    
    def standardize(self, x):
        """
        Apply training statistics for standardization.
        
        Args:
            x: Input features (batch_size, dim)
            
        Returns:
            x_std: Standardized features
        """
        return (x - self.mean) / (self.std + 1e-8)
    
    def destandardize(self, x_std):
        """
        Inverse standardization for reconstruction.
        
        Args:
            x_std: Standardized features
            
        Returns:
            x: Original scale features
        """
        return x_std * self.std + self.mean
    
    def quantize(self, x, return_indices=True):
        """
        Quantize features using FSQ with proper scaling.
        
        Args:
            x: Input features (batch_size, dim) in original scale
            return_indices: If True, return indices; if False, return quantized values
            
        Returns:
            Quantized features or indices
        """
        # Step 1: Standardize using training statistics
        x_std = self.standardize(x)
        
        # Step 2: Clip to expected range
        x_clipped = np.clip(x_std, self.feature_range[0], self.feature_range[1])
        
        # Step 3: Quantize each dimension
        batch_size = x.shape[0]
        quantized = np.zeros_like(x_clipped)
        indices = np.zeros((batch_size, self.dim), dtype=np.int32)
        
        for d in range(self.dim):
            # Find nearest quantization level
            edges = self.boundaries[d]['edges']
            centers = self.boundaries[d]['centers']
            
            # Digitize to find bin indices
            idx = np.digitize(x_clipped[:, d], edges[1:-1])
            idx = np.clip(idx, 0, len(centers) - 1)
            
            indices[:, d] = idx
            quantized[:, d] = centers[idx]
        
        if return_indices:
            return indices
        else:
            # Return in original scale
            return self.destandardize(quantized)
    
    def save_config(self, filepath):
        """
        Save scaling configuration for deployment.
        
        Args:
            filepath: Path to save JSON config
        """
        config = {
            'levels': self.levels.tolist(),
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'feature_range': list(self.feature_range),
            'dim': self.dim
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved FSQ config to {filepath}")
    
    @classmethod
    def load_config(cls, filepath):
        """
        Load scaling configuration from file.
        
        Args:
            filepath: Path to JSON config
            
        Returns:
            FSQPostProcessor instance
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return cls(
            levels=config['levels'],
            mean=config['mean'],
            std=config['std'],
            feature_range=tuple(config['feature_range'])
        )


class TrainingStatsExporter:
    """
    Export training statistics for deployment alignment.
    
    CRITICAL: Must be called during training to capture feature statistics.
    """
    
    def __init__(self):
        self.stats_collected = False
        self.running_mean = None
        self.running_var = None
        self.count = 0
    
    def update(self, features):
        """
        Update running statistics with batch of features.
        
        Args:
            features: (batch_size, dim) tensor/array
        """
        features = np.asarray(features)
        batch_size = features.shape[0]
        
        if self.running_mean is None:
            self.running_mean = np.zeros(features.shape[1])
            self.running_var = np.zeros(features.shape[1])
        
        # Welford's online algorithm for stable mean/variance
        batch_mean = np.mean(features, axis=0)
        batch_var = np.var(features, axis=0)
        
        if self.count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            delta = batch_mean - self.running_mean
            total_count = self.count + batch_size
            
            self.running_mean += delta * batch_size / total_count
            
            # Update variance
            m_a = self.running_var * self.count
            m_b = batch_var * batch_size
            M2 = m_a + m_b + delta**2 * self.count * batch_size / total_count
            self.running_var = M2 / total_count
        
        self.count += batch_size
        self.stats_collected = True
    
    def get_stats(self):
        """
        Get final statistics for export.
        
        Returns:
            Dict with mean and std
        """
        if not self.stats_collected:
            raise ValueError("No statistics collected yet")
        
        return {
            'mean': self.running_mean,
            'std': np.sqrt(self.running_var + 1e-8),
            'count': self.count
        }
    
    def export_for_deployment(self, model_dir, levels):
        """
        Export complete configuration for deployment.
        
        Args:
            model_dir: Directory to save config alongside model
            levels: FSQ levels configuration
            
        Returns:
            Path to saved config
        """
        stats = self.get_stats()
        
        # Create post-processor with collected stats
        post_processor = FSQPostProcessor(
            levels=levels,
            mean=stats['mean'],
            std=stats['std']
        )
        
        # Save config
        config_path = Path(model_dir) / 'fsq_config.json'
        post_processor.save_config(config_path)
        
        # Also save raw stats for debugging
        stats_path = Path(model_dir) / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump({
                'mean': stats['mean'].tolist(),
                'std': stats['std'].tolist(),
                'sample_count': stats['count']
            }, f, indent=2)
        
        print(f"Exported deployment config to {config_path}")
        print(f"Exported training stats to {stats_path}")
        
        return config_path


# Example integration with training pipeline
def example_training_integration():
    """
    Example showing how to integrate with training loop.
    """
    print("Example: Training Integration")
    print("-" * 40)
    
    # Initialize stats collector during training
    stats_exporter = TrainingStatsExporter()
    
    # Simulate training batches
    np.random.seed(42)
    for epoch in range(3):
        for batch in range(10):
            # Simulated encoder features
            features = np.random.randn(32, 8) * 2 + 0.5  # Non-standard dist
            stats_exporter.update(features)
    
    # After training, export configuration
    levels = [4, 4, 3, 3, 2, 2, 2, 2]  # Example FSQ levels
    model_dir = "./model_export"
    os.makedirs(model_dir, exist_ok=True)
    
    config_path = stats_exporter.export_for_deployment(model_dir, levels)
    
    # Load in deployment
    print("\nDeployment Loading:")
    post_processor = FSQPostProcessor.load_config(config_path)
    
    # Test quantization
    test_features = np.random.randn(5, 8) * 2 + 0.5
    indices = post_processor.quantize(test_features, return_indices=True)
    quantized = post_processor.quantize(test_features, return_indices=False)
    
    print(f"Input shape: {test_features.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Quantized shape: {quantized.shape}")
    print(f"Reconstruction error: {np.mean((test_features - quantized)**2):.6f}")


if __name__ == "__main__":
    # Run example
    example_training_integration()
    
    # Test edge cases
    print("\n" + "="*50)
    print("Edge Case Tests:")
    print("="*50)
    
    # Test with extreme values
    levels = [8, 8, 8, 8]
    processor = FSQPostProcessor(
        levels=levels,
        mean=np.array([0, 1, -1, 0.5]),
        std=np.array([1, 2, 0.5, 1.5])
    )
    
    # Test data with outliers
    test_data = np.array([
        [0, 1, -1, 0.5],      # Normal
        [10, -10, 5, -5],     # Outliers
        [-3, 3, -2, 2],       # Moderate outliers
    ])
    
    print("Test data:")
    print(test_data)
    
    indices = processor.quantize(test_data, return_indices=True)
    print(f"\nQuantized indices:")
    print(indices)
    
    reconstructed = processor.quantize(test_data, return_indices=False)
    print(f"\nReconstructed values:")
    print(reconstructed)
