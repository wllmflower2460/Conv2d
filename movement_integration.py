#!/usr/bin/env python3
"""
Integration of movement package with 24-point dog pose data.
Demonstrates kinematic analysis for dog behavior classification.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add movement to path if needed
sys.path.append('/home/wllmflower/Development/movement')

from movement.io import load_poses
from movement.kinematics import kinematics
from movement.utils.vector import compute_norm
import xarray as xr


class DogMovementAnalyzer:
    """Analyze dog movements using 24-point skeleton data"""
    
    def __init__(self, keypoint_names=None):
        """Initialize with 24-point dog skeleton"""
        if keypoint_names is None:
            # Default CVAT 24-point structure
            self.keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "throat", "withers", "left_front_shoulder", "left_front_elbow", "left_front_paw",
                "right_front_shoulder", "right_front_elbow", "right_front_paw",
                "center", "left_hip", "left_knee", "left_back_paw",
                "right_hip", "right_knee", "right_back_paw",
                "tail_base", "tail_mid_1", "tail_mid_2", "tail_tip"
            ]
        else:
            self.keypoint_names = keypoint_names
            
        self.dataset = None
        self.kinematics = {}
        
    def load_from_stanford(self, json_path, sample_indices=None, fps=30):
        """Load Stanford Dogs data and convert to movement dataset"""
        
        # Load Stanford Dogs annotations
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        if sample_indices is None:
            # Use first 100 samples as demo
            sample_indices = range(min(100, len(annotations)))
        
        # Extract keypoints for selected samples
        positions = []
        confidences = []
        
        for idx in sample_indices:
            if idx >= len(annotations):
                continue
            ann = annotations[idx]
            joints = np.array(ann['joints'])  # Shape: (24, 3)
            
            # Extract x,y positions and visibility
            pos = joints[:, :2]  # (24, 2)
            vis = joints[:, 2]   # (24,)
            
            positions.append(pos)
            confidences.append(vis)
        
        # Convert to numpy arrays
        # Shape: (n_frames, 2, 24, 1) for single dog
        positions = np.array(positions)  # (n_frames, 24, 2)
        positions = positions.transpose(0, 2, 1)[:, :, :, np.newaxis]  # (n_frames, 2, 24, 1)
        
        confidences = np.array(confidences)[:, :, np.newaxis]  # (n_frames, 24, 1)
        
        # Create movement dataset
        self.dataset = load_poses.from_numpy(
            position_array=positions,
            confidence_array=confidences,
            individual_names=["dog"],
            keypoint_names=self.keypoint_names,
            fps=fps,
            source_software="StanfordDogs"
        )
        
        print(f"Loaded {len(positions)} frames with {len(self.keypoint_names)} keypoints")
        return self.dataset
    
    def compute_kinematics(self):
        """Compute velocity, acceleration, and speed for all keypoints"""
        
        if self.dataset is None:
            raise ValueError("No data loaded. Call load_from_stanford first.")
        
        position = self.dataset.position
        
        # Compute velocity (first derivative)
        velocity = kinematics.compute_time_derivative(position, order=1)
        self.kinematics['velocity'] = velocity
        
        # Compute acceleration (second derivative)  
        acceleration = kinematics.compute_time_derivative(position, order=2)
        self.kinematics['acceleration'] = acceleration
        
        # Compute speed (magnitude of velocity)
        speed = compute_norm(velocity)
        self.kinematics['speed'] = speed
        
        # Compute displacement
        displacement = kinematics.compute_displacement(position)
        self.kinematics['displacement'] = displacement
        
        print("Computed kinematics: velocity, acceleration, speed, displacement")
        return self.kinematics
    
    def analyze_gait(self):
        """Analyze gait patterns from paw movements"""
        
        if 'speed' not in self.kinematics:
            self.compute_kinematics()
        
        # Focus on paw keypoints
        paw_indices = [
            self.keypoint_names.index("left_front_paw"),
            self.keypoint_names.index("right_front_paw"),
            self.keypoint_names.index("left_back_paw"),
            self.keypoint_names.index("right_back_paw")
        ]
        
        paw_names = ["left_front_paw", "right_front_paw", "left_back_paw", "right_back_paw"]
        
        # Extract paw speeds
        paw_speeds = {}
        for idx, name in zip(paw_indices, paw_names):
            paw_speeds[name] = self.kinematics['speed'].sel(keypoints=self.keypoint_names[idx])
        
        # Compute gait metrics
        gait_metrics = {
            'mean_speed': {},
            'max_speed': {},
            'speed_variance': {}
        }
        
        for name, speed_data in paw_speeds.items():
            # Handle NaN values
            valid_speed = speed_data.values[~np.isnan(speed_data.values)]
            if len(valid_speed) > 0:
                gait_metrics['mean_speed'][name] = np.mean(valid_speed)
                gait_metrics['max_speed'][name] = np.max(valid_speed)
                gait_metrics['speed_variance'][name] = np.var(valid_speed)
            else:
                gait_metrics['mean_speed'][name] = 0
                gait_metrics['max_speed'][name] = 0
                gait_metrics['speed_variance'][name] = 0
        
        # Classify gait type based on speed patterns
        avg_speed = np.mean(list(gait_metrics['mean_speed'].values()))
        
        if avg_speed < 50:
            gait_type = "standing/sitting"
        elif avg_speed < 150:
            gait_type = "walking"
        elif avg_speed < 300:
            gait_type = "trotting"
        else:
            gait_type = "running"
        
        gait_metrics['gait_type'] = gait_type
        gait_metrics['average_speed'] = avg_speed
        
        return gait_metrics
    
    def extract_behavioral_features(self):
        """Extract features useful for behavior classification"""
        
        if 'velocity' not in self.kinematics:
            self.compute_kinematics()
        
        features = {}
        
        # Body center movement
        center_idx = self.keypoint_names.index("center")
        center_speed = self.kinematics['speed'].sel(keypoints=self.keypoint_names[center_idx])
        features['center_mean_speed'] = float(np.nanmean(center_speed.values))
        features['center_speed_std'] = float(np.nanstd(center_speed.values))
        
        # Head movement (nose tracking)
        nose_idx = self.keypoint_names.index("nose")
        nose_speed = self.kinematics['speed'].sel(keypoints=self.keypoint_names[nose_idx])
        features['head_mean_speed'] = float(np.nanmean(nose_speed.values))
        features['head_speed_std'] = float(np.nanstd(nose_speed.values))
        
        # Tail movement (if annotated)
        tail_base_idx = self.keypoint_names.index("tail_base")
        tail_speed = self.kinematics['speed'].sel(keypoints=self.keypoint_names[tail_base_idx])
        features['tail_mean_speed'] = float(np.nanmean(tail_speed.values))
        features['tail_speed_std'] = float(np.nanstd(tail_speed.values))
        
        # Overall body acceleration (activity level)
        accel_magnitude = compute_norm(self.kinematics['acceleration'])
        features['mean_acceleration'] = float(np.nanmean(accel_magnitude.values))
        features['max_acceleration'] = float(np.nanmax(accel_magnitude.values))
        
        # Gait features
        gait = self.analyze_gait()
        features['gait_type'] = gait['gait_type']
        features['gait_speed'] = gait['average_speed']
        
        return features


def demo_analysis():
    """Demo analysis with Stanford Dogs data"""
    
    print("Dog Movement Analysis Demo")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DogMovementAnalyzer()
    
    # Load sample data
    stanford_json = Path("datasets/stanford_dogs/StanfordExtra_v12.json")
    
    if stanford_json.exists():
        # Load first 50 frames for demo
        dataset = analyzer.load_from_stanford(stanford_json, sample_indices=range(50), fps=30)
        
        # Compute kinematics
        kinematics = analyzer.compute_kinematics()
        
        # Analyze gait
        print("\nGait Analysis:")
        print("-" * 40)
        gait_metrics = analyzer.analyze_gait()
        for key, value in gait_metrics.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Extract behavioral features
        print("\nBehavioral Features:")
        print("-" * 40)
        features = analyzer.extract_behavioral_features()
        for key, value in features.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        print("\n✅ Movement analysis complete!")
        
    else:
        print(f"Stanford Dogs data not found at {stanford_json}")
        print("Creating synthetic demo data...")
        
        # Create synthetic data for testing
        n_frames = 100
        positions = np.random.randn(n_frames, 2, 24, 1) * 100 + 200
        confidences = np.ones((n_frames, 24, 1))
        
        dataset = load_poses.from_numpy(
            position_array=positions,
            confidence_array=confidences,
            individual_names=["demo_dog"],
            keypoint_names=analyzer.keypoint_names,
            fps=30,
            source_software="Synthetic"
        )
        
        print(f"Created synthetic dataset: {dataset}")


if __name__ == "__main__":
    demo_analysis()