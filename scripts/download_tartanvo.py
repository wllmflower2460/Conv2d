#!/usr/bin/env python3
"""
Download TartanVO/TartanAir IMU datasets for semi-synthetic validation.
Provides 1000Hz IMU with realistic noise characteristics.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

def setup_tartanair():
    """Install tartanair if not already installed."""
    try:
        import tartanair
    except ImportError:
        print("Installing tartanair package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tartanair"])
        print("TartanAir installed successfully!")

def download_environment(env_name: str, base_path: str = "/mnt/ssd/Conv2d_Datasets/semi_synthetic/tartanvo"):
    """
    Download IMU data for a specific environment.
    
    Args:
        env_name: Environment name (e.g., 'Downtown', 'OldTown')
        base_path: Base directory for dataset storage
    """
    import tartanair as ta
    
    # Create directory structure
    dataset_path = Path(base_path)
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading TartanVO {env_name} IMU data")
    print(f"Target: {dataset_path}")
    print(f"{'='*60}")
    
    # Initialize TartanAir
    ta.init(str(dataset_path))
    
    # Download IMU modality (includes acc, gyro, time)
    print(f"Downloading {env_name}...")
    ta.download(
        env=env_name,
        difficulty=['easy', 'hard'],  # Get both difficulty levels (lowercase)
        modality=['imu'],
        unzip=True
    )
    
    print(f"✅ {env_name} downloaded successfully!")
    
    # Verify downloaded files
    env_path = dataset_path / env_name
    if env_path.exists():
        imu_files = list(env_path.rglob("*.npy"))
        print(f"Found {len(imu_files)} IMU files")
        
        # Show sample structure
        if imu_files:
            print("\nSample files:")
            for f in imu_files[:5]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name}: {size_mb:.2f} MB")
    
    return dataset_path / env_name

def process_imu_data(env_path: Path):
    """
    Process downloaded IMU data into our standard format.
    
    Args:
        env_path: Path to environment directory
    """
    import numpy as np
    
    print(f"\nProcessing IMU data from {env_path.name}...")
    
    # Find all IMU sequences
    acc_files = list(env_path.rglob("acc.npy"))
    gyro_files = list(env_path.rglob("gyro.npy"))
    time_files = list(env_path.rglob("imu_time.npy"))
    
    print(f"Found {len(acc_files)} accelerometer files")
    print(f"Found {len(gyro_files)} gyroscope files")
    
    # Create processed directory
    processed_dir = env_path.parent.parent.parent / "processed" / "tartanvo" / env_path.name
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    all_labels = []
    
    for idx, (acc_file, gyro_file) in enumerate(zip(acc_files, gyro_files)):
        # Load data
        acc_data = np.load(acc_file)  # (N, 3)
        gyro_data = np.load(gyro_file)  # (N, 3)
        
        # Combine into 6-DOF IMU
        imu_data = np.concatenate([acc_data, gyro_data], axis=1)  # (N, 6)
        
        # Add magnetometer placeholder (zeros) to match 9-channel format
        mag_placeholder = np.zeros((imu_data.shape[0], 3))
        imu_9dof = np.concatenate([acc_data, gyro_data, mag_placeholder], axis=1)  # (N, 9)
        
        print(f"  Sequence {idx}: {imu_9dof.shape[0]} samples")
        
        # Window into 100-timestep segments (matching our model)
        window_size = 100
        stride = 50  # 50% overlap
        
        for start in range(0, len(imu_9dof) - window_size, stride):
            window = imu_9dof[start:start + window_size]  # (100, 9)
            
            # Reshape to (9, 2, 100) format expected by model
            # Split into 2 spatial dimensions (arbitrary for IMU)
            reshaped = np.zeros((9, 2, 100))
            reshaped[:, 0, :] = window.T  # First spatial dim gets the data
            reshaped[:, 1, :] = window.T * 0.1  # Second dim gets scaled version
            
            all_data.append(reshaped)
            
            # Assign pseudo-label based on motion characteristics
            acc_magnitude = np.linalg.norm(acc_data[start:start + window_size], axis=1).mean()
            gyro_magnitude = np.linalg.norm(gyro_data[start:start + window_size], axis=1).mean()
            
            # Simple heuristic labeling (will be refined with domain adaptation)
            if acc_magnitude < 1.0:
                label = 0  # stationary
            elif acc_magnitude < 3.0 and gyro_magnitude < 0.5:
                label = 1  # walking
            elif acc_magnitude < 5.0:
                label = 2  # trotting
            else:
                label = 3  # running
            
            all_labels.append(label)
    
    # Convert to arrays
    data_array = np.array(all_data, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)
    
    print(f"\nProcessed data shape: {data_array.shape}")
    print(f"Label distribution: {np.bincount(labels_array)}")
    
    # Save processed data
    np.save(processed_dir / "data.npy", data_array)
    np.save(processed_dir / "labels.npy", labels_array)
    
    print(f"✅ Saved to {processed_dir}")
    
    return data_array, labels_array

def main():
    import numpy as np
    parser = argparse.ArgumentParser(description="Download TartanVO IMU datasets")
    parser.add_argument("--env", type=str, default="Downtown",
                       choices=["Downtown", "OldTown", "Hospital", "Neighborhood"],
                       help="Environment to download")
    parser.add_argument("--base_path", type=str, 
                       default="/mnt/ssd/Conv2d_Datasets/semi_synthetic/tartanvo",
                       help="Base directory for datasets")
    parser.add_argument("--process", action="store_true",
                       help="Process downloaded data into standard format")
    
    args = parser.parse_args()
    
    # Check if base path exists, create if not
    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Creating base directory: {base_path}")
        base_path.mkdir(parents=True, exist_ok=True)
    
    # Setup TartanAir
    setup_tartanair()
    
    # Download environment
    env_path = download_environment(args.env, args.base_path)
    
    # Process if requested
    if args.process:
        data, labels = process_imu_data(env_path)
        print(f"\n🎉 TartanVO {args.env} ready for validation!")
        print(f"Data shape: {data.shape}")
        print(f"Unique labels: {np.unique(labels)}")

if __name__ == "__main__":
    main()