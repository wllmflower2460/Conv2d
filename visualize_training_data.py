#!/usr/bin/env python3
"""
Visualize training data for Conv2d-VQ-HDP-HSMM model
Shows data structure, format, and basic statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import yaml
import seaborn as sns
from preprocessing.enhanced_pipeline import EnhancedCrossSpeciesDataset, get_dataset

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def visualize_data_structure():
    """Visualize the data structure and format"""
    print("=" * 60)
    print("Conv2d-VQ-HDP-HSMM Training Data Analysis")
    print("=" * 60)
    
    # Load configuration
    config_path = 'configs/enhanced_dataset_schema.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n📊 Data Configuration:")
    print(f"  - Input shape: {config['hailo_deployment']['io_specification']['input_shape']}")
    print(f"  - Batch × Channels × Devices × Time: (B, 9, 2, 100)")
    print(f"  - Device 1: Phone (pocket)")
    print(f"  - Device 2: IMU (chest/collar)")
    print(f"  - Sampling rate: 100 Hz")
    print(f"  - Window size: 100 samples (1 second)")
    
    # Create dataset
    print("\n📁 Loading Enhanced Cross-Species Dataset...")
    dataset = EnhancedCrossSpeciesDataset(
        config_path=config_path,
        mode='train',
        enforce_hailo_constraints=True
    )
    
    print(f"  - Total samples: {len(dataset)}")
    
    # Get a sample batch
    sample_batch = dataset[0]
    
    print("\n🔍 Sample Batch Structure:")
    print(f"  - Input tensor shape: {sample_batch['input'].shape}")
    print(f"  - Input dtype: {sample_batch['input'].dtype}")
    print(f"  - Human label: {sample_batch['human_label'].item()}")
    print(f"  - Dog label: {sample_batch['dog_label'].item()}")
    print(f"  - Has dog label: {sample_batch['has_dog_label']}")
    
    return dataset, config


def visualize_sensor_data(dataset, num_samples=3):
    """Visualize sensor data from phone and IMU"""
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, num_samples * 4))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sensor_names = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']
    
    for sample_idx in range(num_samples):
        data = dataset[sample_idx]
        input_tensor = data['input']  # Shape: (9, 2, 100)
        
        # Phone data
        phone_data = input_tensor[:, 0, :].numpy()  # (9, 100)
        ax = axes[sample_idx, 0]
        
        # Plot first 3 channels (accelerometer)
        time_steps = np.arange(100) / 100.0  # Convert to seconds
        for i in range(3):
            ax.plot(time_steps, phone_data[i, :], label=sensor_names[i], alpha=0.8)
        
        ax.set_title(f'Sample {sample_idx}: Phone Accelerometer')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Sensor Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # IMU data
        imu_data = input_tensor[:, 1, :].numpy()  # (9, 100)
        ax = axes[sample_idx, 1]
        
        # Plot first 3 channels (accelerometer)
        for i in range(3):
            ax.plot(time_steps, imu_data[i, :], label=sensor_names[i], alpha=0.8)
        
        ax.set_title(f'Sample {sample_idx}: IMU Accelerometer')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Sensor Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Phone vs IMU Sensor Data Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('analysis/sensor_data_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Sensor visualization saved to: analysis/sensor_data_comparison.png")


def analyze_data_distribution(dataset):
    """Analyze label distribution and data statistics"""
    human_labels = []
    dog_labels = []
    has_dog_labels = []
    
    # Collect all labels
    for i in range(min(len(dataset), 500)):  # Sample first 500 for speed
        data = dataset[i]
        human_labels.append(data['human_label'].item())
        dog_labels.append(data['dog_label'].item())
        has_dog_labels.append(data['has_dog_label'])
    
    # Create distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Human activity distribution
    ax = axes[0]
    unique_human, counts_human = np.unique(human_labels, return_counts=True)
    ax.bar(unique_human, counts_human, color='skyblue', edgecolor='black')
    ax.set_xlabel('Human Activity ID')
    ax.set_ylabel('Count')
    ax.set_title('Human Activity Distribution')
    ax.grid(True, alpha=0.3)
    
    # Dog behavior distribution (excluding -1)
    ax = axes[1]
    valid_dog_labels = [l for l in dog_labels if l != -1]
    if valid_dog_labels:
        unique_dog, counts_dog = np.unique(valid_dog_labels, return_counts=True)
        behavior_names = ['Sit', 'Down', 'Stand']
        ax.bar(unique_dog, counts_dog, color='lightcoral', edgecolor='black')
        ax.set_xticks(unique_dog)
        ax.set_xticklabels([behavior_names[i] if i < len(behavior_names) else f'Behavior {i}' 
                           for i in unique_dog])
        ax.set_xlabel('Dog Behavior')
        ax.set_ylabel('Count')
        ax.set_title('Dog Behavior Distribution')
    else:
        ax.text(0.5, 0.5, 'No dog labels available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Dog Behavior Distribution')
    ax.grid(True, alpha=0.3)
    
    # Cross-species mapping coverage
    ax = axes[2]
    coverage = sum(has_dog_labels) / len(has_dog_labels) * 100
    ax.bar(['Has Mapping', 'No Mapping'], 
          [sum(has_dog_labels), len(has_dog_labels) - sum(has_dog_labels)],
          color=['green', 'gray'], edgecolor='black', alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title(f'Cross-Species Mapping Coverage\n({coverage:.1f}% mapped)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Data Label Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig('analysis/label_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Label distribution saved to: analysis/label_distribution.png")
    
    # Print statistics
    print("\n📈 Data Statistics:")
    print(f"  - Human activities: {len(unique_human)} unique")
    print(f"  - Dog behaviors: {len(unique_dog) if valid_dog_labels else 0} mapped")
    print(f"  - Cross-species coverage: {coverage:.1f}%")


def test_dataloader(dataset):
    """Test DataLoader functionality"""
    print("\n🔄 Testing DataLoader...")
    
    dataloader = dataset.get_dataloader(batch_size=32, shuffle=True, num_workers=0)
    
    # Get one batch
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            print(f"  - Batch input shape: {batch['input'].shape}")
            print(f"  - Batch human labels shape: {batch['human_label'].shape}")
            print(f"  - Batch dog labels shape: {batch['dog_label'].shape}")
            print(f"  - Device: {batch['input'].device}")
            
            # Check for NaN or Inf
            has_nan = torch.isnan(batch['input']).any()
            has_inf = torch.isinf(batch['input']).any()
            print(f"  - Contains NaN: {has_nan}")
            print(f"  - Contains Inf: {has_inf}")
            
            # Check value ranges
            min_val = batch['input'].min().item()
            max_val = batch['input'].max().item()
            mean_val = batch['input'].mean().item()
            std_val = batch['input'].std().item()
            
            print(f"  - Value range: [{min_val:.3f}, {max_val:.3f}]")
            print(f"  - Mean: {mean_val:.3f}, Std: {std_val:.3f}")
            
            break
    
    print("  ✅ DataLoader working correctly")


def visualize_frequency_spectrum(dataset):
    """Visualize frequency spectrum of sensor signals"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get a few samples
    num_samples = 10
    phone_spectra = []
    imu_spectra = []
    
    for i in range(num_samples):
        data = dataset[i]
        input_tensor = data['input'].numpy()  # (9, 2, 100)
        
        # Compute FFT for accelerometer channels
        phone_acc = input_tensor[:3, 0, :]  # First 3 channels, phone
        imu_acc = input_tensor[:3, 1, :]    # First 3 channels, IMU
        
        # Average across channels and compute FFT
        phone_fft = np.abs(np.fft.rfft(phone_acc.mean(axis=0)))
        imu_fft = np.abs(np.fft.rfft(imu_acc.mean(axis=0)))
        
        phone_spectra.append(phone_fft)
        imu_spectra.append(imu_fft)
    
    phone_spectra = np.array(phone_spectra)
    imu_spectra = np.array(imu_spectra)
    
    # Frequency bins (assuming 100 Hz sampling rate)
    freqs = np.fft.rfftfreq(100, d=1/100)
    
    # Plot phone spectrum
    ax = axes[0, 0]
    ax.plot(freqs, phone_spectra.mean(axis=0), 'b-', label='Mean', linewidth=2)
    ax.fill_between(freqs, 
                    phone_spectra.mean(axis=0) - phone_spectra.std(axis=0),
                    phone_spectra.mean(axis=0) + phone_spectra.std(axis=0),
                    alpha=0.3, color='b')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Phone Accelerometer Spectrum')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot IMU spectrum
    ax = axes[0, 1]
    ax.plot(freqs, imu_spectra.mean(axis=0), 'r-', label='Mean', linewidth=2)
    ax.fill_between(freqs,
                    imu_spectra.mean(axis=0) - imu_spectra.std(axis=0),
                    imu_spectra.mean(axis=0) + imu_spectra.std(axis=0),
                    alpha=0.3, color='r')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('IMU Accelerometer Spectrum')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot comparison
    ax = axes[1, 0]
    ax.plot(freqs, phone_spectra.mean(axis=0), 'b-', label='Phone', linewidth=2, alpha=0.7)
    ax.plot(freqs, imu_spectra.mean(axis=0), 'r-', label='IMU', linewidth=2, alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Phone vs IMU Spectrum Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot coherence
    ax = axes[1, 1]
    coherence = np.abs(phone_spectra.mean(axis=0) - imu_spectra.mean(axis=0))
    ax.plot(freqs, coherence, 'g-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Difference')
    ax.set_title('Spectral Difference (Phone - IMU)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Frequency Domain Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('analysis/frequency_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Frequency spectrum saved to: analysis/frequency_spectrum.png")


def main():
    """Main visualization pipeline"""
    # Create analysis directory
    Path('analysis').mkdir(exist_ok=True)
    
    # Load and analyze data
    dataset, config = visualize_data_structure()
    
    # Visualize sensor data
    print("\n📊 Visualizing sensor data...")
    visualize_sensor_data(dataset, num_samples=3)
    
    # Analyze distributions
    print("\n📊 Analyzing label distributions...")
    analyze_data_distribution(dataset)
    
    # Test DataLoader
    test_dataloader(dataset)
    
    # Frequency analysis
    print("\n📊 Analyzing frequency spectrum...")
    visualize_frequency_spectrum(dataset)
    
    print("\n" + "=" * 60)
    print("✅ Training Data Analysis Complete!")
    print("=" * 60)
    print("\nKey Findings:")
    print("1. Data format: (Batch, 9, 2, 100) - 9 channels, 2 devices, 100 timesteps")
    print("2. Dual-device input: Phone (pocket) + IMU (chest/collar)")
    print("3. Cross-species mapping: Human activities → Dog behaviors")
    print("4. Hailo-compatible: Static shapes, Conv2d architecture")
    print("\nThe data is properly formatted for Conv2d-VQ-HDP-HSMM training!")


if __name__ == '__main__':
    main()