"""
Example usage of Conv2d-VQ-HDP-HSMM model for behavioral analysis.
Demonstrates model instantiation, training, and inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from conv2d_vq_hdp_hsmm import Conv2dVQHDPHSMM
from conv2d_vq_hdp_hsmm.utils import create_monitoring_suite, update_monitors, get_monitoring_report


def generate_synthetic_imu_data(batch_size=8, num_samples=100):
    """
    Generate synthetic IMU data for testing.
    
    Returns:
        data: Synthetic IMU data (B, 9, 2, 100)
        labels: Optional behavioral labels for evaluation
    """
    # Simulate dual-device IMU data
    # 9 channels: 3 accelerometer + 3 gyroscope + 3 magnetometer per device
    # 2 devices (e.g., wrist and ankle)
    # 100 time steps
    
    data = []
    labels = []
    
    for b in range(batch_size):
        # Generate different behavioral patterns
        behavior_type = np.random.randint(0, 3)  # 3 behavior types
        
        if behavior_type == 0:  # Walking
            # Periodic patterns for walking
            t = np.linspace(0, 4*np.pi, num_samples)
            
            # Accelerometer (walking rhythm)
            accel = np.sin(t).reshape(1, 1, -1) + 0.1 * np.random.randn(3, 1, num_samples)
            # Gyroscope (turning movements)  
            gyro = 0.5 * np.cos(2*t).reshape(1, 1, -1) + 0.05 * np.random.randn(3, 1, num_samples)
            # Magnetometer (orientation)
            mag = 0.3 * np.ones((3, 1, num_samples)) + 0.02 * np.random.randn(3, 1, num_samples)
            
        elif behavior_type == 1:  # Running
            # Higher frequency patterns for running
            t = np.linspace(0, 8*np.pi, num_samples)
            
            accel = 1.5 * np.sin(t).reshape(1, 1, -1) + 0.2 * np.random.randn(3, 1, num_samples)
            gyro = 0.8 * np.cos(3*t).reshape(1, 1, -1) + 0.1 * np.random.randn(3, 1, num_samples)
            mag = 0.3 * np.ones((3, 1, num_samples)) + 0.03 * np.random.randn(3, 1, num_samples)
            
        else:  # Stationary/sitting
            # Low amplitude random patterns
            accel = 0.1 * np.random.randn(3, 1, num_samples)
            gyro = 0.05 * np.random.randn(3, 1, num_samples)
            mag = 0.3 * np.ones((3, 1, num_samples)) + 0.01 * np.random.randn(3, 1, num_samples)
        
        # Combine for dual device (same pattern with slight variation)
        device_data = np.concatenate([accel, gyro, mag], axis=0)  # (9, 1, 100)
        
        # Duplicate for second device with small variation
        device2_data = device_data + 0.05 * np.random.randn(*device_data.shape)
        
        # Combine both devices
        sample_data = np.concatenate([device_data, device2_data], axis=1)  # (9, 2, 100)
        
        data.append(sample_data)
        labels.append(behavior_type)
    
    data = np.stack(data, axis=0)  # (B, 9, 2, 100)
    labels = np.array(labels)
    
    return torch.FloatTensor(data), torch.LongTensor(labels)


def train_model():
    """Train the Conv2d-VQ-HDP-HSMM model."""
    
    # Model configuration
    config = {
        'input_channels': 9,
        'input_features': 2,
        'sequence_length': 100,
        'latent_dim': 64,
        'num_embeddings': 512,
        'max_clusters': 20,
        'num_states': 10,
        'max_duration': 10
    }
    
    # Initialize model
    model = Conv2dVQHDPHSMM(**config)
    
    # Create monitoring suite
    monitors = create_monitoring_suite(model)
    
    print("Model initialized with {} parameters".format(
        sum(p.numel() for p in model.parameters())
    ))
    
    print("Edge deployment analysis:")
    for key, value in monitors['edge_analysis'].items():
        print(f"  {key}: {value}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    num_epochs = 50
    batch_size = 8
    
    model.train()
    
    for epoch in range(num_epochs):
        # Generate batch of synthetic data
        data, labels = generate_synthetic_imu_data(batch_size)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        
        # Compute reconstruction loss
        reconstruction_loss = nn.MSELoss()(outputs['reconstructed'], data)
        outputs['reconstruction_loss'] = reconstruction_loss
        
        # Compute total loss
        total_loss, loss_components = model.compute_total_loss(outputs)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Update monitors
        update_monitors(monitors, outputs)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Total Loss: {total_loss.item():.4f}")
            print("Loss Components:")
            for name, loss in loss_components.items():
                print(f"  {name}: {loss.item():.4f}")
            
            # Print monitoring stats
            perp_stats = monitors['perplexity'].get_stats()
            if perp_stats:
                print(f"VQ Perplexity: {perp_stats['current_perplexity']:.2f} "
                      f"(trend: {perp_stats['trend_perplexity']})")
            
            cluster_stats = monitors['clusters'].get_stats()
            if cluster_stats:
                print(f"Active Clusters: {cluster_stats['active_clusters']}/20 "
                      f"(balance: {cluster_stats['cluster_balance']:.2f})")
    
    return model, monitors


def evaluate_model(model, monitors):
    """Evaluate the trained model."""
    
    model.eval()
    
    # Generate test data
    test_data, test_labels = generate_synthetic_imu_data(batch_size=16)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(test_data, return_intermediate=True)
        
        # Get behavioral analysis
        behavioral_analysis = model.get_behavioral_analysis(outputs)
        
        # Extract features
        features = model.extract_features(test_data)
        
        print("\n=== Model Evaluation ===")
        print(f"Test batch shape: {test_data.shape}")
        print(f"Reconstruction error: {nn.MSELoss()(outputs['reconstructed'], test_data).item():.4f}")
        
        print("\n=== Behavioral Analysis ===")
        for key, value in behavioral_analysis.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
        
        print(f"\nExtracted features shapes:")
        for key, value in features.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        # Generate monitoring report
        report = get_monitoring_report(monitors)
        
        print("\n=== Monitoring Report ===")
        print(f"Edge ready: {report['edge_analysis']['edge_ready']}")
        print(f"Model complexity: {report['edge_analysis']['complexity_score']:.2f}M params")
        
        if report['perplexity_stats']:
            print(f"Average perplexity: {report['perplexity_stats']['avg_perplexity']:.2f}")
        
        if report['cluster_stats']:
            print(f"Active clusters: {report['cluster_stats']['active_clusters']}")
            print(f"Cluster entropy: {report['cluster_stats']['entropy']:.2f}")
        
        if report['behavioral_patterns']:
            print(f"Pattern diversity: {report['behavioral_patterns']['pattern_diversity']:.3f}")
    
    return outputs, behavioral_analysis, features


def demonstrate_inference():
    """Demonstrate real-time inference capabilities."""
    
    print("\n=== Inference Demonstration ===")
    
    # Create model
    model = Conv2dVQHDPHSMM()
    model.eval()
    
    # Simulate real-time data stream
    print("Simulating real-time IMU data stream...")
    
    streaming_results = []
    
    for i in range(5):  # 5 time windows
        # Generate single sample
        data, _ = generate_synthetic_imu_data(batch_size=1)
        
        with torch.no_grad():
            outputs = model(data)
            behavioral_analysis = model.get_behavioral_analysis(outputs)
            
            result = {
                'window': i,
                'perplexity': outputs['perplexity'].item(),
                'dominant_cluster': behavioral_analysis['cluster_assignments'].argmax(axis=1).mean(),
                'avg_transition_rate': behavioral_analysis['avg_transition_rate']
            }
            
            streaming_results.append(result)
            
            print(f"Window {i}: Perplexity={result['perplexity']:.2f}, "
                  f"Main Cluster={result['dominant_cluster']:.0f}, "
                  f"Transition Rate={result['avg_transition_rate']:.3f}")
    
    return streaming_results


if __name__ == "__main__":
    print("Conv2d-VQ-HDP-HSMM Behavioral Analysis Demo")
    print("=" * 50)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model
    print("\n1. Training model...")
    model, monitors = train_model()
    
    # Evaluate model
    print("\n2. Evaluating model...")
    outputs, analysis, features = evaluate_model(model, monitors)
    
    # Demonstrate inference
    print("\n3. Real-time inference demo...")
    streaming_results = demonstrate_inference()
    
    print("\n=== Demo Complete ===")
    print("Model is ready for deployment on Hailo-8 or similar edge devices.")
    print(f"Key capabilities demonstrated:")
    print("- Vector quantization with codebook learning")
    print("- Hierarchical clustering with HDP")
    print("- Temporal modeling with HSMM")
    print("- Real-time behavioral analysis")
    print("- Edge-compatible Conv2d-only architecture")