#!/usr/bin/env python3
"""Shape and dtype tests - guarantees float32 and canonical shapes across each stage."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.kmeans import KMeansClusterer
from conv2d.clustering.gmm import GMMClusterer
from conv2d.temporal.median import MedianHysteresisPolicy


class TestDtypeShapes:
    """Test dtype and shape contracts - critical for edge deployment safety."""
    
    def test_fsq_input_validation_strict(self):
        """FSQ must reject invalid input shapes and dtypes."""
        
        # Valid input first
        x_valid = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        result = encode_fsq(x_valid)  # Should work
        
        # Test invalid shapes
        invalid_shapes = [
            (16, 8, 2, 100),   # Wrong channel count
            (16, 9, 3, 100),   # Wrong sensor count  
            (16, 9, 2, 50),    # Wrong timesteps
            (16, 9, 2),        # Missing dimension
            (9, 2, 100),       # Missing batch dimension
        ]
        
        for shape in invalid_shapes:
            with pytest.raises((ValueError, RuntimeError)), \
                 pytest.warns(None) as warnings:
                x_invalid = torch.randn(*shape, dtype=torch.float32)
                encode_fsq(x_invalid)
        
        # Test invalid dtypes
        invalid_dtypes = [
            torch.float64,
            torch.float16, 
            torch.int32,
            torch.int64,
        ]
        
        for dtype in invalid_dtypes:
            with pytest.raises((TypeError, ValueError, RuntimeError)):
                x_invalid = torch.randn(16, 9, 2, 100, dtype=dtype)
                encode_fsq(x_invalid)
    
    def test_fsq_output_shape_dtype_guarantees(self):
        """FSQ outputs must have guaranteed shapes and dtypes."""
        
        batch_sizes = [1, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 9, 2, 100, dtype=torch.float32)
            result = encode_fsq(x, levels=[8, 6, 5], embedding_dim=64)
            
            # Codes: (B, embedding_dim) int32
            assert result.codes.shape == (batch_size, 64), \
                f"Wrong codes shape: {result.codes.shape}, expected ({batch_size}, 64)"
            assert result.codes.dtype == torch.int32, \
                f"Wrong codes dtype: {result.codes.dtype}, expected int32"
            
            # Features: (B, feature_dim) float32
            assert result.features.shape[0] == batch_size, \
                f"Wrong features batch size: {result.features.shape[0]}, expected {batch_size}"
            assert result.features.dtype == torch.float32, \
                f"Wrong features dtype: {result.features.dtype}, expected float32"
            
            # Embeddings: (B, embedding_dim) float32
            assert result.embeddings.shape == (batch_size, 64), \
                f"Wrong embeddings shape: {result.embeddings.shape}, expected ({batch_size}, 64)"
            assert result.embeddings.dtype == torch.float32, \
                f"Wrong embeddings dtype: {result.embeddings.dtype}, expected float32"
            
            # Perplexity: scalar float
            assert isinstance(result.perplexity, float), \
                f"Wrong perplexity type: {type(result.perplexity)}, expected float"
            assert result.perplexity > 0, \
                f"Invalid perplexity: {result.perplexity}, must be positive"
    
    def test_clustering_input_validation(self):
        """Clustering must validate input features."""
        
        # Valid features first
        features_valid = np.random.randn(1000, 256).astype(np.float32)
        
        clusterer = KMeansClusterer(random_state=42)
        labels = clusterer.fit_predict(features_valid, k=4)  # Should work
        
        # Test invalid dtypes
        invalid_dtypes = [np.float64, np.int32, np.int64, np.float16]
        
        for dtype in invalid_dtypes:
            features_invalid = np.random.randn(100, 256).astype(dtype)
            
            # Should either convert or raise error
            try:
                labels = clusterer.fit_predict(features_invalid, k=4)
                # If it works, check output dtype
                assert labels.dtype == np.int32, \
                    f"Labels not int32 for input dtype {dtype}: {labels.dtype}"
            except (TypeError, ValueError):
                # Acceptable to reject invalid input
                pass
        
        # Test invalid shapes
        invalid_shapes = [
            (100,),         # 1D
            (100, 256, 10), # 3D
            (0, 256),       # Empty
        ]
        
        for shape in invalid_shapes:
            if 0 not in shape:  # Skip empty arrays for now
                features_invalid = np.random.randn(*shape).astype(np.float32)
                
                with pytest.raises((ValueError, IndexError)):
                    clusterer.fit_predict(features_invalid, k=4)
    
    def test_clustering_output_guarantees(self):
        """Clustering outputs must have guaranteed properties."""
        
        sample_counts = [100, 500, 1000, 2000]
        feature_dims = [64, 128, 256]
        
        for n_samples in sample_counts:
            for feature_dim in feature_dims:
                features = np.random.randn(n_samples, feature_dim).astype(np.float32)
                
                for k in [3, 4, 5, 8]:
                    clusterer = KMeansClusterer(random_state=42)
                    labels = clusterer.fit_predict(features, k=k)
                    
                    # Shape: (n_samples,)
                    assert labels.shape == (n_samples,), \
                        f"Wrong labels shape: {labels.shape}, expected ({n_samples},)"
                    
                    # Dtype: int32
                    assert labels.dtype == np.int32, \
                        f"Wrong labels dtype: {labels.dtype}, expected int32"
                    
                    # Value range: [0, k-1]  
                    assert labels.min() >= 0, \
                        f"Labels contain negative values: {labels.min()}"
                    assert labels.max() < k, \
                        f"Labels exceed k-1: {labels.max()} >= {k}"
                    
                    # All labels should be integers
                    assert np.all(labels == labels.astype(int)), \
                        "Labels contain non-integer values"
    
    def test_temporal_shape_preservation_strict(self):
        """Temporal smoothing must preserve shapes exactly."""
        
        shapes = [
            (1, 10),
            (5, 50),
            (10, 100),
            (32, 200),
        ]
        
        for batch_size, timesteps in shapes:
            labels = np.random.randint(0, 4, size=(batch_size, timesteps), dtype=np.int32)
            
            policy = MedianHysteresisPolicy(min_dwell=3, window_size=5)
            smoothed = policy.smooth(labels)
            
            # Shape must be identical
            assert smoothed.shape == labels.shape, \
                f"Shape changed: {labels.shape} → {smoothed.shape}"
            
            # Dtype must be preserved
            assert smoothed.dtype == labels.dtype, \
                f"Dtype changed: {labels.dtype} → {smoothed.dtype}"
            
            # Values must be in original range
            original_values = set(labels.flatten())
            smoothed_values = set(smoothed.flatten())
            assert smoothed_values <= original_values, \
                f"New values introduced: {smoothed_values - original_values}"
    
    def test_end_to_end_pipeline_shapes(self):
        """Complete pipeline must maintain shape contracts."""
        
        batch_sizes = [1, 4, 16]
        
        for batch_size in batch_sizes:
            # Start with canonical IMU input
            x = torch.randn(batch_size, 9, 2, 100, dtype=torch.float32)
            
            # FSQ encoding
            result = encode_fsq(x, levels=[8, 6, 5])
            
            # Validate FSQ output shapes
            assert result.features.dtype == torch.float32
            assert result.codes.dtype == torch.int32
            assert result.codes.shape[0] == batch_size
            
            # Clustering  
            features_np = result.features.detach().numpy()
            assert features_np.dtype == np.float32, \
                f"Features not float32 after conversion: {features_np.dtype}"
            
            clusterer = GMMClusterer(random_state=42)
            labels = clusterer.fit_predict(features_np, k=4)
            
            # Validate clustering output
            assert labels.dtype == np.int32
            assert labels.shape == (batch_size,)
            
            # Temporal smoothing (reshape for temporal processing)
            if batch_size > 1:
                # Simulate temporal sequence by reshaping
                seq_length = batch_size  
                labels_temporal = labels.reshape(1, seq_length)
                
                policy = MedianHysteresisPolicy(min_dwell=2)
                smoothed = policy.smooth(labels_temporal)
                
                # Validate temporal output
                assert smoothed.dtype == np.int32
                assert smoothed.shape == (1, seq_length)
            
            print(f"✓ End-to-end pipeline validated for batch_size={batch_size}")
    
    def test_dtype_consistency_across_devices(self):
        """Dtypes should be consistent across CPU/GPU (if available)."""
        
        x_cpu = torch.randn(8, 9, 2, 100, dtype=torch.float32)
        result_cpu = encode_fsq(x_cpu)
        
        # Check CPU results
        assert result_cpu.codes.dtype == torch.int32
        assert result_cpu.features.dtype == torch.float32
        assert result_cpu.embeddings.dtype == torch.float32
        
        if torch.cuda.is_available():
            x_gpu = x_cpu.cuda()
            result_gpu = encode_fsq(x_gpu)
            
            # Check GPU results have same dtypes
            assert result_gpu.codes.dtype == torch.int32
            assert result_gpu.features.dtype == torch.float32
            assert result_gpu.embeddings.dtype == torch.float32
            
            # Results should be numerically similar
            assert torch.allclose(result_cpu.codes.float(), result_gpu.codes.cpu().float())
            assert torch.allclose(result_cpu.features, result_gpu.features.cpu())
    
    def test_memory_layout_consistency(self):
        """Memory layouts should be consistent and efficient."""
        
        x = torch.randn(16, 9, 2, 100, dtype=torch.float32)
        result = encode_fsq(x)
        
        # Check memory is contiguous (important for deployment)
        assert result.codes.is_contiguous(), "Codes tensor not contiguous"
        assert result.features.is_contiguous(), "Features tensor not contiguous" 
        assert result.embeddings.is_contiguous(), "Embeddings tensor not contiguous"
        
        # Check no unexpected strides
        expected_codes_stride = (result.codes.shape[1], 1)
        assert result.codes.stride() == expected_codes_stride, \
            f"Unexpected codes stride: {result.codes.stride()}"
    
    def test_numerical_stability_edge_cases(self):
        """Edge case inputs should not break dtype/shape contracts."""
        
        edge_cases = [
            torch.zeros(4, 9, 2, 100, dtype=torch.float32),                    # All zeros
            torch.ones(4, 9, 2, 100, dtype=torch.float32),                     # All ones
            torch.full((4, 9, 2, 100), float('inf'), dtype=torch.float32),     # Inf values  
            torch.full((4, 9, 2, 100), float('-inf'), dtype=torch.float32),    # -Inf values
            torch.full((4, 9, 2, 100), 1e10, dtype=torch.float32),            # Very large
            torch.full((4, 9, 2, 100), 1e-10, dtype=torch.float32),           # Very small
        ]
        
        for i, x in enumerate(edge_cases):
            try:
                result = encode_fsq(x)
                
                # Even with edge cases, output contracts must hold
                assert result.codes.dtype == torch.int32, \
                    f"Edge case {i}: codes not int32"
                assert result.features.dtype == torch.float32, \
                    f"Edge case {i}: features not float32"
                assert result.codes.shape[0] == x.shape[0], \
                    f"Edge case {i}: batch size mismatch"
                
                # Check for NaN/Inf in outputs
                assert torch.isfinite(result.features).all(), \
                    f"Edge case {i}: non-finite features"
                assert torch.isfinite(result.embeddings).all(), \
                    f"Edge case {i}: non-finite embeddings"
                
            except (ValueError, RuntimeError) as e:
                # Acceptable to reject problematic inputs
                print(f"Edge case {i} properly rejected: {e}")
    
    def test_batch_independence(self):
        """Different batch sizes should not affect per-sample shapes/dtypes."""
        
        # Create identical samples
        single_sample = torch.randn(1, 9, 2, 100, dtype=torch.float32)
        
        # Process individually  
        result_single = encode_fsq(single_sample)
        
        # Process in batch
        batch_samples = single_sample.repeat(8, 1, 1, 1)
        result_batch = encode_fsq(batch_samples)
        
        # Per-sample shapes should be identical
        assert result_single.codes.shape[1:] == result_batch.codes.shape[1:], \
            "Per-sample code shape depends on batch size"
        assert result_single.features.shape[1:] == result_batch.features.shape[1:], \
            "Per-sample feature shape depends on batch size"
        
        # Dtypes should be identical
        assert result_single.codes.dtype == result_batch.codes.dtype
        assert result_single.features.dtype == result_batch.features.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])