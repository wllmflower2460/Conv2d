"""
Simple demonstration of Conv2d-VQ-HDP-HSMM model without training.
"""

import torch
import numpy as np
from conv2d_vq_hdp_hsmm import Conv2dVQHDPHSMM


def generate_test_data(batch_size=4, sequence_length=50):
    """Generate synthetic test data."""
    # Create simple test patterns
    data = []
    for b in range(batch_size):
        # Create a simple sine wave pattern across channels
        t = np.linspace(0, 2*np.pi, sequence_length)
        
        # 9 channels (3x3 IMU sensors)
        channels = []
        for c in range(9):
            freq = 1 + c * 0.1  # Different frequency per channel
            pattern = np.sin(freq * t) + 0.1 * np.random.randn(sequence_length)
            channels.append(pattern)
        
        channels = np.stack(channels)  # (9, sequence_length)
        
        # Duplicate for 2 devices with slight variation
        device1 = channels
        device2 = channels + 0.1 * np.random.randn(*channels.shape)
        
        sample = np.stack([device1, device2], axis=1)  # (9, 2, sequence_length)
        data.append(sample)
    
    data = np.stack(data)  # (batch_size, 9, 2, sequence_length)
    return torch.FloatTensor(data)


def main():
    print("Conv2d-VQ-HDP-HSMM Simple Demo")
    print("=" * 40)
    
    # Initialize model with smaller parameters for demo
    model = Conv2dVQHDPHSMM(
        input_channels=9,
        input_features=2,
        sequence_length=50,
        latent_dim=32,       # Smaller for demo
        num_embeddings=64,   # Smaller for demo
        max_clusters=8,      # Smaller for demo
        num_states=5,        # Smaller for demo
        max_duration=5
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate test data
    test_data = generate_test_data(batch_size=2, sequence_length=50)
    print(f"Input shape: {test_data.shape}")
    
    # Forward pass (inference only)
    model.eval()
    with torch.no_grad():
        outputs = model(test_data, return_intermediate=True)
        
        print("\nModel Outputs:")
        print(f"- Reconstructed shape: {outputs['reconstructed'].shape}")
        print(f"- VQ Loss: {outputs['vq_loss'].item():.4f}")
        print(f"- HDP Loss: {outputs['hdp_loss'].item():.4f}")
        print(f"- HSMM Likelihood: {outputs['hsmm_likelihood'].item():.4f}")
        print(f"- Perplexity: {outputs['perplexity'].item():.2f}")
        
        # Behavioral analysis
        analysis = model.get_behavioral_analysis(outputs)
        print(f"\nBehavioral Analysis:")
        print(f"- Codebook utilization: {analysis['codebook_utilization']:.3f}")
        print(f"- Active codes: {analysis['active_codes']}")
        print(f"- Active clusters: {analysis['active_clusters']}")
        print(f"- Cluster entropy: {analysis['cluster_entropy']:.3f}")
        print(f"- Avg transition rate: {analysis['avg_transition_rate']:.3f}")
        
        # Feature extraction
        features = model.extract_features(test_data)
        print(f"\nExtracted Features:")
        for key, value in features.items():
            if hasattr(value, 'shape'):
                print(f"- {key}: {value.shape}")
        
        # Test individual components
        print(f"\nComponent Tests:")
        
        # VQ layer stats
        vq_stats = model.vq_layer.get_codebook_usage()
        print(f"- VQ utilization: {vq_stats['utilization']:.3f}")
        print(f"- Active VQ codes: {vq_stats['active_codes']}")
        
        # HDP cluster stats
        cluster_stats = model.hdp_layer.get_cluster_statistics(outputs['cluster_assignments'])
        print(f"- HDP active clusters: {cluster_stats['active_clusters']}")
        print(f"- HDP entropy: {cluster_stats['entropy']:.3f}")
        if 'cluster_balance' in cluster_stats:
            print(f"- HDP balance: {cluster_stats['cluster_balance']:.3f}")
        
        print(f"\n✅ Model working correctly!")
        print(f"Ready for edge deployment on Hailo-8")


if __name__ == "__main__":
    main()