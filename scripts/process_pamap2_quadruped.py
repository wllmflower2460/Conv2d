#!/usr/bin/env python3
"""
Process PAMAP2 human activity data for quadruped behavioral analysis.

Maps human IMU sensor data (3 IMUs: chest, hand, ankle) to quadruped configuration.
PAMAP2 contains 52 columns with IMU data from chest, dominant wrist, and dominant ankle.
We'll map these to quadruped front-left, front-right, back-left, back-right legs.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm

# PAMAP2 Activity Labels
ACTIVITY_MAP = {
    0: 'other/transient',
    1: 'lying',
    2: 'sitting', 
    3: 'standing',
    4: 'walking',
    5: 'running',
    6: 'cycling',
    7: 'nordic_walking',
    9: 'watching_TV',
    10: 'computer_work',
    11: 'car_driving',
    12: 'ascending_stairs',
    13: 'descending_stairs',
    16: 'vacuum_cleaning',
    17: 'ironing',
    18: 'folding_laundry',
    19: 'house_cleaning',
    20: 'playing_soccer',
    24: 'rope_jumping'
}

# Map to simplified quadruped-relevant activities
QUADRUPED_ACTIVITY_MAP = {
    1: 'rest',        # lying
    2: 'sit',         # sitting
    3: 'stand',       # standing
    4: 'walk',        # walking
    5: 'trot',        # running -> trot
    7: 'walk',        # nordic_walking -> walk
    12: 'walk',       # ascending_stairs -> walk (with variation)
    13: 'walk',       # descending_stairs -> walk (with variation)
    20: 'gallop',     # playing_soccer -> gallop (high energy)
    24: 'trot',       # rope_jumping -> trot (rhythmic)
}

# PAMAP2 column structure (52 columns total)
# Columns 1: timestamp
# Column 2: activity label
# Column 3: heart rate
# Columns 4-20: IMU hand (17 channels)
# Columns 21-37: IMU chest (17 channels)
# Columns 38-54: IMU ankle (17 channels)

# Each IMU has 17 channels:
# 3D acceleration (3), 3D gyroscope (3), 3D magnetometer (3), 
# orientation (4), temperature (1), 3D acceleration 16g (3)

class PAMAP2QuadrupedProcessor:
    """Process PAMAP2 human data for quadruped behavioral analysis."""
    
    def __init__(self, 
                 data_dir: str = '/mnt/ssd/Conv2d_Datasets/har_adapted/pamap2/PAMAP2_Dataset',
                 output_dir: str = '/mnt/ssd/Conv2d_Datasets/quadruped_adapted',
                 window_size: int = 100,
                 stride: int = 50):
        """
        Initialize processor.
        
        Args:
            data_dir: Directory containing PAMAP2 Protocol data
            output_dir: Output directory for processed quadruped data
            window_size: Window size in samples (100Hz sampling rate)
            stride: Stride for sliding window
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = 100  # Hz
        
        # Define sensor column indices
        self.hand_cols = list(range(3, 20))    # IMU hand columns
        self.chest_cols = list(range(20, 37))  # IMU chest columns  
        self.ankle_cols = list(range(37, 54))  # IMU ankle columns
        
    def load_subject_data(self, subject_file: str) -> pd.DataFrame:
        """Load and parse PAMAP2 subject data."""
        filepath = self.data_dir / 'Protocol' / subject_file
        
        # Read data with space separator
        df = pd.read_csv(filepath, sep=' ', header=None, engine='python')
        
        # Extract key columns
        df_clean = pd.DataFrame({
            'timestamp': df.iloc[:, 0],
            'activity': df.iloc[:, 1].astype('Int64'),  # Handle NaN
            'heart_rate': df.iloc[:, 2],
        })
        
        # Add IMU data
        for i, col_idx in enumerate(self.hand_cols):
            df_clean[f'hand_{i}'] = df.iloc[:, col_idx]
        for i, col_idx in enumerate(self.chest_cols):
            df_clean[f'chest_{i}'] = df.iloc[:, col_idx]
        for i, col_idx in enumerate(self.ankle_cols):
            df_clean[f'ankle_{i}'] = df.iloc[:, col_idx]
            
        # Drop rows with NaN activity (transient periods)
        df_clean = df_clean.dropna(subset=['activity'])
        
        return df_clean
    
    def map_to_quadruped(self, df: pd.DataFrame) -> Dict:
        """
        Map human sensor data to quadruped configuration.
        
        Strategy:
        - Chest IMU → Body/spine orientation
        - Hand IMU → Front leg (alternating left/right based on gait phase)
        - Ankle IMU → Back leg (alternating left/right based on gait phase)
        
        For simplicity, we'll initially map:
        - Chest → average body reference
        - Hand → front legs (duplicated with phase shift)
        - Ankle → back legs (duplicated with phase shift)
        """
        quadruped_data = {
            'front_left': [],
            'front_right': [],
            'back_left': [],
            'back_right': [],
            'body': [],
            'activity': []
        }
        
        # Extract accelerometer and gyroscope data (first 6 channels of each IMU)
        hand_acc_gyro = df[[f'hand_{i}' for i in range(6)]].values
        chest_acc_gyro = df[[f'chest_{i}' for i in range(6)]].values
        ankle_acc_gyro = df[[f'ankle_{i}' for i in range(6)]].values
        
        # Map to quadruped configuration
        # Use phase shifts to simulate left/right alternation
        n_samples = len(df)
        
        for i in range(0, n_samples - self.window_size, self.stride):
            window_slice = slice(i, i + self.window_size)
            
            # Get activity label for window (majority vote)
            activity_window = df['activity'].iloc[window_slice].values
            activity_mode = pd.Series(activity_window).mode()
            if len(activity_mode) > 0:
                activity_label = int(activity_mode.iloc[0])
            else:
                continue
                
            # Skip if not a mapped activity
            if activity_label not in QUADRUPED_ACTIVITY_MAP:
                continue
            
            # Extract sensor data for window
            hand_data = hand_acc_gyro[window_slice]
            chest_data = chest_acc_gyro[window_slice]
            ankle_data = ankle_acc_gyro[window_slice]
            
            # Simulate quadruped gait with phase shifts
            # For walking/trotting, create diagonal phase coupling
            phase_shift = self.window_size // 4  # 90-degree phase shift
            
            # Front legs: hand sensor with phase shift
            front_left_data = hand_data
            front_right_data = np.roll(hand_data, phase_shift, axis=0)
            
            # Back legs: ankle sensor with opposite phase
            back_left_data = np.roll(ankle_data, phase_shift, axis=0)
            back_right_data = ankle_data
            
            # Body: chest sensor
            body_data = chest_data
            
            # Store processed windows
            quadruped_data['front_left'].append(front_left_data)
            quadruped_data['front_right'].append(front_right_data)
            quadruped_data['back_left'].append(back_left_data)
            quadruped_data['back_right'].append(back_right_data)
            quadruped_data['body'].append(body_data)
            quadruped_data['activity'].append(QUADRUPED_ACTIVITY_MAP[activity_label])
            
        return quadruped_data
    
    def process_all_subjects(self):
        """Process all PAMAP2 subjects and create quadruped dataset."""
        protocol_dir = self.data_dir / 'Protocol'
        subject_files = sorted(protocol_dir.glob('subject*.dat'))
        
        all_quadruped_data = {
            'front_left': [],
            'front_right': [],
            'back_left': [],
            'back_right': [],
            'body': [],
            'activity': []
        }
        
        print(f"Processing {len(subject_files)} subjects...")
        
        for subject_file in tqdm(subject_files):
            print(f"\nProcessing {subject_file.name}...")
            
            try:
                # Load subject data
                df = self.load_subject_data(subject_file.name)
                print(f"  Loaded {len(df)} samples")
                
                # Map to quadruped
                quadruped_data = self.map_to_quadruped(df)
                
                # Aggregate data
                for key in all_quadruped_data:
                    all_quadruped_data[key].extend(quadruped_data[key])
                    
                print(f"  Generated {len(quadruped_data['activity'])} windows")
                
            except Exception as e:
                print(f"  Error processing {subject_file.name}: {e}")
                continue
        
        # Convert to numpy arrays
        print("\nConverting to numpy arrays...")
        for key in ['front_left', 'front_right', 'back_left', 'back_right', 'body']:
            all_quadruped_data[key] = np.array(all_quadruped_data[key])
        
        # Create Conv2d-compatible format
        # Shape: (N, 9, 2, 100) - 9 channels (3 acc + 3 gyro + 3 mag), 2 spatial dims, 100 time
        n_samples = len(all_quadruped_data['activity'])
        
        # We'll use accelerometer (3) + gyroscope (3) = 6 channels
        # Add 3 zero channels to get to 9 channels for compatibility
        conv2d_data = np.zeros((n_samples, 9, 2, self.window_size))
        
        for i in range(n_samples):
            # Stack limb data as spatial dimensions
            # Row 0: front legs (left, right)
            # Row 1: back legs (left, right)
            
            # Front left accelerometer and gyroscope
            conv2d_data[i, 0:6, 0, :] = all_quadruped_data['front_left'][i, :, :6].T
            
            # Back left accelerometer and gyroscope  
            conv2d_data[i, 0:6, 1, :] = all_quadruped_data['back_left'][i, :, :6].T
            
            # Could also include front_right and back_right or body data
            # For now, using simplified 2-row spatial layout
        
        # Create labels
        activity_labels = all_quadruped_data['activity']
        unique_activities = list(set(activity_labels))
        activity_to_idx = {act: idx for idx, act in enumerate(unique_activities)}
        labels = np.array([activity_to_idx[act] for act in activity_labels])
        
        # Save processed data
        output_file = self.output_dir / 'pamap2_quadruped_processed.npz'
        np.savez_compressed(
            output_file,
            data=conv2d_data,
            labels=labels,
            activity_names=unique_activities,
            activity_to_idx=activity_to_idx,
            window_size=self.window_size,
            stride=self.stride,
            sampling_rate=self.sampling_rate
        )
        
        print(f"\nProcessed data saved to {output_file}")
        print(f"Data shape: {conv2d_data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Activities: {unique_activities}")
        
        # Print activity distribution
        print("\nActivity distribution:")
        for act in unique_activities:
            count = np.sum(labels == activity_to_idx[act])
            percentage = 100 * count / len(labels)
            print(f"  {act}: {count} samples ({percentage:.1f}%)")
        
        return conv2d_data, labels, activity_to_idx


def main():
    """Main processing pipeline."""
    processor = PAMAP2QuadrupedProcessor()
    
    # Process all subjects
    data, labels, activity_map = processor.process_all_subjects()
    
    # Split into train/val/test (60/20/20)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Save splits
    output_dir = Path('/mnt/ssd/Conv2d_Datasets/quadruped_adapted')
    
    np.savez_compressed(
        output_dir / 'pamap2_quadruped_train.npz',
        data=data[train_indices],
        labels=labels[train_indices]
    )
    
    np.savez_compressed(
        output_dir / 'pamap2_quadruped_val.npz',
        data=data[val_indices],
        labels=labels[val_indices]
    )
    
    np.savez_compressed(
        output_dir / 'pamap2_quadruped_test.npz',
        data=data[test_indices],
        labels=labels[test_indices]
    )
    
    print(f"\nDataset splits created:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")
    
    print("\nPAMAP2 quadruped preprocessing complete!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()