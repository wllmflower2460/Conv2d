#!/usr/bin/env python3
"""
Integration test suite for Conv2d-FSQ pipeline.
Tests the complete pipeline from IMU input to behavioral analysis output.
Addresses D1 review requirement for comprehensive integration testing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch
import numpy as np
from pathlib import Path
import json
import tempfile
from typing import Dict, Tuple

# Import models and components
from models.conv2d_fsq_optimized import Conv2dFSQOptimized
from models.transfer_entropy_real import BehavioralSynchronyMetrics
from preprocessing.enhanced_pipeline import EnhancedMovementDataset


class TestFSQPipeline:
    """Integration tests for FSQ behavioral analysis pipeline."""
    
    @pytest.fixture
    def sample_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sample IMU data for testing."""
        batch_size = 32
        n_channels = 9  # 3-axis acc, gyro, mag
        spatial_dims = 2
        timesteps = 100
        n_classes = 12  # PAMAP2 activities
        
        # Synthetic IMU data
        X = torch.randn(batch_size, n_channels, spatial_dims, timesteps)
        y = torch.randint(0, n_classes, (batch_size,))
        
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data) -> Conv2dFSQOptimized:
        """Create and train a simple FSQ model."""
        X, y = sample_data
        
        model = Conv2dFSQOptimized(
            input_channels=9,
            hidden_dim=128,
            num_classes=12,
            fsq_levels=[4, 4, 4],  # 64 codes
            dropout=0.2
        )
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            logits, _ = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        return model
    
    def test_pipeline_end_to_end(self, trained_model, sample_data):
        """Test complete pipeline from input to output."""
        X, y = sample_data
        
        # Forward pass
        with torch.no_grad():
            logits, codes = trained_model(X)
        
        # Validate outputs
        assert logits.shape == (32, 12), "Incorrect logit shape"
        assert codes.shape == (32,), "Incorrect code shape"
        assert codes.max() < 64, "Code exceeds maximum value"
        
        # Check predictions
        predictions = logits.argmax(dim=1)
        assert predictions.shape == y.shape, "Prediction shape mismatch"
    
    def test_codebook_utilization(self, trained_model):
        """Test that codebook utilization meets requirements."""
        # Generate more data to test utilization
        for _ in range(20):
            X = torch.randn(32, 9, 2, 100)
            trained_model.train()
            _, codes = trained_model(X)
        
        # Check statistics
        stats = trained_model.get_codebook_stats()
        
        assert stats['total_codes'] == 64, "Incorrect total codes"
        assert stats['usage_ratio'] > 0.5, f"Low usage: {stats['usage_ratio']:.2%}"
        assert stats['perplexity'] > 10, f"Low perplexity: {stats['perplexity']:.2f}"
        
        print(f"Codebook utilization: {stats['usage_ratio']:.2%}")
        print(f"Perplexity: {stats['perplexity']:.2f}")
    
    def test_temporal_consistency(self, trained_model):
        """Test temporal consistency of behavioral codes."""
        # Create temporally correlated data
        batch_size = 10
        X1 = torch.randn(batch_size, 9, 2, 100)
        X2 = X1 + 0.1 * torch.randn_like(X1)  # Similar to X1
        X3 = torch.randn(batch_size, 9, 2, 100)  # Different
        
        with torch.no_grad():
            _, codes1 = trained_model(X1)
            _, codes2 = trained_model(X2)
            _, codes3 = trained_model(X3)
        
        # Similar inputs should produce similar codes
        similarity_12 = (codes1 == codes2).float().mean()
        similarity_13 = (codes1 == codes3).float().mean()
        
        assert similarity_12 > similarity_13, "Temporal consistency violation"
        print(f"Similar input agreement: {similarity_12:.2%}")
        print(f"Different input agreement: {similarity_13:.2%}")
    
    def test_synchrony_metrics(self):
        """Test behavioral synchrony metric calculation."""
        # Create coupled time series
        n = 500
        t = np.linspace(0, 10, n)
        
        # X drives Y with delay
        x = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(n)
        y = np.roll(x, 10) + 0.2 * np.random.randn(n)
        
        # Calculate metrics
        sync_calc = BehavioralSynchronyMetrics()
        metrics = sync_calc.calculate_all_metrics(x, y)
        
        # Validate metrics
        assert 'te_x_to_y' in metrics, "Missing TE(X→Y)"
        assert 'te_y_to_x' in metrics, "Missing TE(Y→X)"
        assert 'mutual_information' in metrics, "Missing MI"
        assert 'phase_locking_value' in metrics, "Missing PLV"
        
        # X should drive Y (positive net TE)
        assert metrics['net_te'] > 0, f"Incorrect causality: {metrics['net_te']:.4f}"
        
        print(f"TE(X→Y): {metrics['te_x_to_y']:.4f}")
        print(f"Net TE: {metrics['net_te']:.4f}")
    
    def test_batch_processing(self, trained_model):
        """Test batch processing capabilities."""
        batch_sizes = [1, 16, 32, 64]
        
        for bs in batch_sizes:
            X = torch.randn(bs, 9, 2, 100)
            
            with torch.no_grad():
                logits, codes = trained_model(X)
            
            assert logits.shape[0] == bs, f"Batch size {bs} failed"
            assert codes.shape[0] == bs, f"Batch size {bs} code shape incorrect"
    
    def test_gradient_flow(self, sample_data):
        """Test gradient flow through the pipeline."""
        X, y = sample_data
        
        model = Conv2dFSQOptimized(
            input_channels=9,
            hidden_dim=128,
            num_classes=12,
            fsq_levels=[4, 4, 4]
        )
        
        # Check gradient flow
        model.train()
        logits, codes = model(X)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        
        # All parameters should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    
    def test_export_onnx(self, trained_model, tmp_path):
        """Test ONNX export for edge deployment."""
        # Sample input
        dummy_input = torch.randn(1, 9, 2, 100)
        
        # Export path
        onnx_path = tmp_path / "model.onnx"
        
        # Export model
        torch.onnx.export(
            trained_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['logits', 'codes'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'codes': {0: 'batch_size'}
            }
        )
        
        assert onnx_path.exists(), "ONNX export failed"
        assert onnx_path.stat().st_size > 0, "ONNX file is empty"
        
        print(f"ONNX model size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    def test_statistical_corrections(self):
        """Test Bonferroni correction implementation."""
        # Simulate p-values from multiple tests
        p_values = [0.01, 0.03, 0.001, 0.05, 0.02, 0.004]
        alpha = 0.05
        n_tests = len(p_values)
        
        # Apply Bonferroni correction
        corrected_alpha = alpha / n_tests  # 0.0083
        
        significant = [p < corrected_alpha for p in p_values]
        
        # Expected: indices 0, 2, 5 are significant
        expected = [True, False, True, False, False, True]
        
        assert significant == expected, "Bonferroni correction error"
        print(f"Corrected alpha: {corrected_alpha:.4f}")
        print(f"Significant tests: {sum(significant)}/{n_tests}")
    
    def test_performance_targets(self, trained_model):
        """Test that performance targets are met."""
        # Measure inference time
        import time
        
        X = torch.randn(1, 9, 2, 100)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = trained_model(X)
        
        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                _ = trained_model(X)
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        
        print(f"Inference latency - P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")
        
        # Check targets (relaxed for CPU)
        assert p50 < 100, f"P50 latency {p50:.2f}ms exceeds target"
        assert p95 < 200, f"P95 latency {p95:.2f}ms exceeds target"


class TestDataValidation:
    """Test data validation and quality assurance."""
    
    def test_nan_handling(self):
        """Test handling of NaN values in input."""
        model = Conv2dFSQOptimized()
        
        # Create data with NaNs
        X = torch.randn(10, 9, 2, 100)
        X[0, 0, 0, :10] = float('nan')
        
        # Model should handle gracefully
        with pytest.raises(Exception) as exc_info:
            # We expect the model to detect and reject NaN inputs
            _ = model(X)
        
        # Or it should handle them
        # This depends on implementation
    
    def test_data_normalization(self):
        """Test data normalization requirements."""
        # Create unnormalized data
        X = torch.randn(32, 9, 2, 100) * 100 + 50  # Large values
        
        # Normalize
        X_norm = (X - X.mean(dim=(2, 3), keepdim=True)) / (X.std(dim=(2, 3), keepdim=True) + 1e-8)
        
        # Check normalization
        assert torch.abs(X_norm.mean()) < 0.1, "Poor normalization"
        assert torch.abs(X_norm.std() - 1.0) < 0.1, "Poor normalization"


@pytest.mark.integration
class TestFullPipeline:
    """Full pipeline integration tests."""
    
    def test_complete_workflow(self):
        """Test complete workflow from data loading to results."""
        print("\n" + "="*60)
        print("COMPLETE WORKFLOW TEST")
        print("="*60)
        
        # 1. Create model
        model = Conv2dFSQOptimized(
            input_channels=9,
            hidden_dim=128,
            num_classes=12,
            fsq_levels=[4, 4, 4]
        )
        
        # 2. Generate training data
        X_train = torch.randn(100, 9, 2, 100)
        y_train = torch.randint(0, 12, (100,))
        
        # 3. Train model
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            logits, codes = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
        
        # 4. Evaluate
        model.eval()
        X_test = torch.randn(50, 9, 2, 100)
        y_test = torch.randint(0, 12, (50,))
        
        with torch.no_grad():
            logits, codes = model(X_test)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == y_test).float().mean()
        
        # 5. Check metrics
        stats = model.get_codebook_stats()
        
        print(f"Test Accuracy: {accuracy:.2%}")
        print(f"Codebook Usage: {stats['usage_ratio']:.2%}")
        print(f"Unique Codes: {stats['used_codes']}/{stats['total_codes']}")
        
        # 6. Calculate synchrony
        sync_calc = BehavioralSynchronyMetrics()
        x = np.random.randn(500)
        y = np.random.randn(500)
        metrics = sync_calc.calculate_all_metrics(x, y)
        
        print(f"Synchrony Index: {metrics['synchrony_index']:.4f}")
        
        assert accuracy > 0.0, "Model not learning"
        assert stats['usage_ratio'] > 0.3, "Poor codebook usage"
        
        print("\n✓ Complete workflow test passed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])