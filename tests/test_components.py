"""
Basic tests for Conv2d-VQ-HDP-HSMM components.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from conv2d_vq_hdp_hsmm import VectorQuantization, HDPClustering, HSMM, Conv2dVQHDPHSMM


def test_vector_quantization():
    """Test Vector Quantization layer."""
    print("Testing Vector Quantization...")
    
    # Initialize VQ layer
    vq = VectorQuantization(num_embeddings=16, embedding_dim=8)
    
    # Test input
    batch_size, channels, height, width = 2, 8, 4, 10
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    quantized, vq_loss, perplexity, encodings = vq(x)
    
    # Check shapes
    assert quantized.shape == x.shape, f"Quantized shape {quantized.shape} != input shape {x.shape}"
    assert encodings.shape == (batch_size, 16, height, width), f"Encodings shape {encodings.shape} incorrect"
    assert isinstance(vq_loss.item(), float), "VQ loss should be a scalar"
    assert isinstance(perplexity.item(), float), "Perplexity should be a scalar"
    
    # Check codebook usage
    stats = vq.get_codebook_usage()
    assert 'cluster_size' in stats
    assert 'active_codes' in stats
    assert 'utilization' in stats
    
    print("✓ Vector Quantization test passed")


def test_hdp_clustering():
    """Test HDP Clustering layer."""
    print("Testing HDP Clustering...")
    
    # Initialize HDP layer
    hdp = HDPClustering(input_dim=8, max_clusters=5)
    
    # Test input
    batch_size, channels, height, width = 2, 8, 4, 10
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    cluster_assignments, cluster_centers, kl_loss = hdp(x)
    
    # Check shapes
    assert cluster_assignments.shape == (batch_size, 5, height, width), f"Cluster assignments shape {cluster_assignments.shape} incorrect"
    assert cluster_centers.shape == (batch_size, 5, 8, height, width), f"Cluster centers shape {cluster_centers.shape} incorrect"
    assert isinstance(kl_loss.item(), float), "KL loss should be a scalar"
    
    # Check probabilities sum to 1
    prob_sums = cluster_assignments.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Cluster probabilities should sum to 1"
    
    # Check cluster statistics
    stats = hdp.get_cluster_statistics(cluster_assignments)
    assert 'active_clusters' in stats
    assert 'entropy' in stats
    
    print("✓ HDP Clustering test passed")


def test_hsmm():
    """Test HSMM layer."""
    print("Testing HSMM...")
    
    # Initialize HSMM layer
    hsmm = HSMM(num_states=4, max_duration=3, feature_dim=8)
    
    # Test input
    batch_size, feature_dim, height, width = 2, 8, 1, 1
    time_steps = 5
    
    features = torch.randn(batch_size, feature_dim, height, width)
    observations = torch.randn(batch_size, feature_dim, time_steps, height, width)
    
    # Forward pass
    state_probs, log_likelihood, duration_probs = hsmm(features, observations)
    
    # Check shapes
    assert state_probs.shape == (batch_size, 4, time_steps, height, width), f"State probs shape {state_probs.shape} incorrect"
    assert log_likelihood.shape == (batch_size, height, width), f"Log likelihood shape {log_likelihood.shape} incorrect"
    assert duration_probs.shape == (batch_size, 4, 3, height, width), f"Duration probs shape {duration_probs.shape} incorrect"
    
    # Check probabilities are valid
    assert torch.all(state_probs >= 0), "State probabilities should be non-negative"
    assert torch.all(duration_probs >= 0), "Duration probabilities should be non-negative"
    
    print("✓ HSMM test passed")


def test_full_model():
    """Test the complete Conv2d-VQ-HDP-HSMM model."""
    print("Testing full model...")
    
    # Initialize model with small parameters for testing
    model = Conv2dVQHDPHSMM(
        input_channels=9,
        input_features=2,
        sequence_length=20,  # Smaller for testing
        latent_dim=16,       # Smaller for testing
        num_embeddings=32,   # Smaller for testing
        max_clusters=5,      # Smaller for testing
        num_states=3,        # Smaller for testing
        max_duration=3
    )
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 9, 2, 20)
    
    # Forward pass
    outputs = model(x)
    
    # Check outputs
    assert 'reconstructed' in outputs
    assert 'vq_loss' in outputs
    assert 'hdp_loss' in outputs
    assert 'hsmm_likelihood' in outputs
    assert 'perplexity' in outputs
    assert 'cluster_assignments' in outputs
    assert 'state_probs' in outputs
    
    # Check shapes
    reconstructed = outputs['reconstructed']
    assert len(reconstructed.shape) == 4, f"Reconstructed should be 4D, got {reconstructed.shape}"
    assert reconstructed.shape[0] == batch_size, f"Batch size mismatch: {reconstructed.shape[0]} != {batch_size}"
    assert reconstructed.shape[1] == 9, f"Channel mismatch: {reconstructed.shape[1]} != 9"
    
    # Test behavioral analysis
    analysis = model.get_behavioral_analysis(outputs)
    assert 'codebook_utilization' in analysis
    assert 'perplexity' in analysis
    assert 'active_clusters' in analysis
    
    # Test feature extraction
    features = model.extract_features(x)
    assert 'encoded' in features
    assert 'quantized' in features
    assert 'vq_codes' in features
    
    # Test loss computation
    outputs['reconstruction_loss'] = torch.nn.MSELoss()(reconstructed, x)
    total_loss, loss_components = model.compute_total_loss(outputs)
    
    assert isinstance(total_loss.item(), float)
    assert 'reconstruction' in loss_components
    assert 'vq' in loss_components
    assert 'hdp' in loss_components
    assert 'hsmm' in loss_components
    
    print("✓ Full model test passed")


def test_edge_compatibility():
    """Test edge deployment compatibility."""
    print("Testing edge compatibility...")
    
    model = Conv2dVQHDPHSMM()
    
    # Check for prohibited operations
    prohibited_found = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            prohibited_found.append(f"Conv1d found: {name}")
        elif isinstance(module, torch.nn.GroupNorm):
            prohibited_found.append(f"GroupNorm found: {name}")
        elif hasattr(module, 'forward'):
            # Check for softmax in forward method (simplified check)
            forward_code = str(module.forward)
            if 'F.softmax' in forward_code or 'torch.softmax' in forward_code:
                prohibited_found.append(f"Softmax found: {name}")
    
    if prohibited_found:
        print("WARNING: Found potentially problematic operations:")
        for op in prohibited_found:
            print(f"  - {op}")
    else:
        print("✓ No prohibited operations found")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    # Test inference time (rough)
    model.eval()
    x = torch.randn(1, 9, 2, 100)
    
    import time
    with torch.no_grad():
        start_time = time.time()
        _ = model(x)
        inference_time = time.time() - start_time
    
    print(f"✓ Single inference time: {inference_time*1000:.2f} ms")
    
    print("✓ Edge compatibility test completed")


def run_all_tests():
    """Run all tests."""
    print("Running Conv2d-VQ-HDP-HSMM Component Tests")
    print("=" * 50)
    
    try:
        test_vector_quantization()
        test_hdp_clustering()
        test_hsmm()
        test_full_model()
        test_edge_compatibility()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()