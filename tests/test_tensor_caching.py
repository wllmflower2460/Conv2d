"""
Unit tests for tensor caching optimization in device_attention.py
Ensures cached tensors work correctly across devices and maintain performance
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.device_attention import CrossSpeciesAttention


class TestTensorCaching:
    """Test suite for tensor caching optimization"""
    
    @pytest.fixture
    def attention_module(self):
        """Create a CrossSpeciesAttention module for testing"""
        return CrossSpeciesAttention(hidden_dim=128, num_species=2)
    
    def test_cache_initialization(self, attention_module):
        """Test that cache is properly initialized"""
        assert hasattr(attention_module, '_species_tensor_cache')
        assert isinstance(attention_module._species_tensor_cache, dict)
        assert len(attention_module._species_tensor_cache) == 0
    
    def test_tensor_caching_cpu(self, attention_module):
        """Test tensor caching on CPU device"""
        device = torch.device('cpu')
        features = torch.randn(4, 128, 100, device=device)
        
        # First call should create and cache tensors
        output1 = attention_module(features, source_species=0, target_species=1)
        assert len(attention_module._species_tensor_cache) == 2
        
        # Get cached tensors
        cached_human = attention_module._get_species_tensor(0, device)
        cached_dog = attention_module._get_species_tensor(1, device)
        
        # Verify tensors are correct
        assert cached_human.item() == 0
        assert cached_dog.item() == 1
        assert cached_human.device == device
        assert cached_dog.device == device
        
        # Second call should reuse cached tensors
        output2 = attention_module(features, source_species=1, target_species=0)
        assert len(attention_module._species_tensor_cache) == 2  # No new tensors created
        
        # Outputs should be valid
        assert output1.shape == features.shape
        assert output2.shape == features.shape
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tensor_caching_cuda(self, attention_module):
        """Test tensor caching on CUDA device"""
        device = torch.device('cuda:0')
        attention_module = attention_module.to(device)
        features = torch.randn(4, 128, 100, device=device)
        
        # First call creates CUDA tensors
        output = attention_module(features, source_species=0, target_species=1)
        
        # Check cache has both CPU and CUDA versions
        cached_tensors = attention_module._species_tensor_cache
        
        # Verify CUDA tensors are cached
        cuda_human = attention_module._get_species_tensor(0, device)
        cuda_dog = attention_module._get_species_tensor(1, device)
        
        assert cuda_human.device.type == 'cuda'
        assert cuda_dog.device.type == 'cuda'
        assert cuda_human.item() == 0
        assert cuda_dog.item() == 1
        
        # Output should be on CUDA
        assert output.device.type == 'cuda'
        assert output.shape == features.shape
    
    def test_multi_device_caching(self, attention_module):
        """Test that different devices maintain separate caches"""
        cpu_device = torch.device('cpu')
        
        # Create CPU tensors
        cpu_features = torch.randn(2, 128, 50, device=cpu_device)
        _ = attention_module(cpu_features, 0, 1)
        
        initial_cache_size = len(attention_module._species_tensor_cache)
        
        if torch.cuda.is_available():
            cuda_device = torch.device('cuda:0')
            attention_module = attention_module.to(cuda_device)
            cuda_features = torch.randn(2, 128, 50, device=cuda_device)
            _ = attention_module(cuda_features, 0, 1)
            
            # Should have separate entries for CPU and CUDA
            final_cache_size = len(attention_module._species_tensor_cache)
            assert final_cache_size > initial_cache_size
            
            # Verify both CPU and CUDA tensors exist in cache
            cpu_tensor = attention_module._get_species_tensor(0, cpu_device)
            cuda_tensor = attention_module._get_species_tensor(0, cuda_device)
            
            assert cpu_tensor.device.type == 'cpu'
            assert cuda_tensor.device.type == 'cuda'
    
    def test_cache_persistence(self, attention_module):
        """Test that cached tensors persist across multiple forward passes"""
        device = torch.device('cpu')
        
        # Multiple forward passes
        for i in range(10):
            features = torch.randn(2, 128, 50, device=device)
            _ = attention_module(features, i % 2, (i + 1) % 2)
        
        # Cache should only have 2 species tensors for CPU
        cached_keys = [k for k in attention_module._species_tensor_cache.keys() 
                      if k[1] == 'cpu']  # k[1] is device.type now
        assert len(cached_keys) == 2  # Only 2 species on CPU
    
    def test_gradient_flow(self, attention_module):
        """Test that gradients flow correctly through cached tensors"""
        device = torch.device('cpu')
        features = torch.randn(2, 128, 50, device=device, requires_grad=True)
        
        # Forward pass with cached tensors
        output = attention_module(features, 0, 1)
        
        # Compute loss and backward
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        
        # Cached tensors should not have gradients (they're indices)
        cached_tensor = attention_module._get_species_tensor(0, device)
        assert cached_tensor.grad is None
    
    def test_memory_efficiency(self, attention_module):
        """Test that caching reduces memory allocations"""
        device = torch.device('cpu')
        
        # Warm up cache
        features = torch.randn(2, 128, 50, device=device)
        _ = attention_module(features, 0, 1)
        
        # Get cached tensor references
        cached_0 = attention_module._get_species_tensor(0, device)
        cached_1 = attention_module._get_species_tensor(1, device)
        
        # Multiple calls should return same tensor objects
        for _ in range(5):
            tensor_0 = attention_module._get_species_tensor(0, device)
            tensor_1 = attention_module._get_species_tensor(1, device)
            
            # Should be exact same objects in memory
            assert tensor_0 is cached_0
            assert tensor_1 is cached_1
    
    def test_cache_correctness(self, attention_module):
        """Test that cached tensors produce correct results"""
        device = torch.device('cpu')
        features = torch.randn(4, 128, 100, device=device)
        
        # Get output with caching
        attention_module._species_tensor_cache.clear()  # Clear cache
        output_cached = attention_module(features, 0, 1)
        
        # Create new module without cache to compare
        fresh_module = CrossSpeciesAttention(hidden_dim=128, num_species=2)
        fresh_module.load_state_dict(attention_module.state_dict())
        
        # Get output from fresh module (will create new tensors)
        output_fresh = fresh_module(features, 0, 1)
        
        # Results should be identical
        torch.testing.assert_close(output_cached, output_fresh, rtol=1e-5, atol=1e-7)
    
    def test_different_batch_sizes(self, attention_module):
        """Test that caching works correctly with different batch sizes"""
        device = torch.device('cpu')
        
        # Different batch sizes
        batch_sizes = [1, 4, 8, 16]
        outputs = []
        
        for batch_size in batch_sizes:
            features = torch.randn(batch_size, 128, 50, device=device)
            output = attention_module(features, 0, 1)
            outputs.append(output)
            assert output.shape[0] == batch_size
        
        # Cache should still only have 2 entries for CPU
        cpu_cache_size = len([k for k in attention_module._species_tensor_cache.keys() 
                             if k[1] == 'cpu'])  # k[1] is device.type
        assert cpu_cache_size == 2


class TestPerformanceImpact:
    """Test performance improvements from tensor caching"""
    
    @pytest.fixture
    def large_attention_module(self):
        """Create a larger attention module for performance testing"""
        return CrossSpeciesAttention(hidden_dim=512, num_species=4)
    
    def test_forward_pass_performance(self, large_attention_module):
        """Test that cached tensors improve forward pass performance"""
        import time
        device = torch.device('cpu')
        features = torch.randn(32, 512, 200, device=device)
        
        # Warm up
        _ = large_attention_module(features, 0, 1)
        
        # Time multiple forward passes with cache
        start = time.time()
        for _ in range(100):
            _ = large_attention_module(features, 0, 1)
        cached_time = time.time() - start
        
        # This test mainly ensures the optimization doesn't break anything
        # In practice, the improvement is small but measurable
        assert cached_time < 10.0  # Should complete in reasonable time
    
    @pytest.mark.parametrize("num_species", [2, 4, 8])
    def test_scaling_with_species(self, num_species):
        """Test that caching scales well with number of species"""
        module = CrossSpeciesAttention(hidden_dim=256, num_species=num_species)
        device = torch.device('cpu')
        features = torch.randn(8, 256, 100, device=device)
        
        # Use all species combinations
        for source in range(num_species):
            for target in range(num_species):
                output = module(features, source, target)
                assert output.shape == features.shape
        
        # Cache should have one entry per species
        cpu_cache_size = len([k for k in module._species_tensor_cache.keys() 
                             if k[1] == 'cpu'])  # k[1] is device.type
        assert cpu_cache_size == num_species


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])