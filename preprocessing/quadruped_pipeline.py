import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class QuadrupedDatasetHAR:
    """Quadruped-focused preprocessing pipeline with synthetic animal datasets"""
    
    def __init__(self, window_size=100, overlap=0.5):
        self.window_size = window_size  # ~1 second at 100Hz
        self.overlap = overlap
        self.datasets = {}
        
        # Quadruped-specific label mappings
        self.label_mappings = {
            'awa_pose': self._awa_pose_labels(),
            'animal_activity': self._animal_activity_labels(), 
            'cear_quadruped': self._cear_labels()
        }
        
        # Canonical label mapping focused on quadruped behaviors
        self.canonical_labels = {
            # Static poses (core for trainer MVP)
            'sit': 0, 'down': 1, 'stand': 2, 'stay': 3, 'lying': 4,
            # Transitions (key for training detection)
            'sit_to_down': 5, 'down_to_sit': 6, 'sit_to_stand': 7, 'stand_to_sit': 8,
            'down_to_stand': 9, 'stand_to_down': 10,
            # Movement gaits (auxiliary from CEAR)
            'walking': 11, 'trotting': 12, 'running': 13, 'turning': 14,
            # Feeding/grooming behaviors
            'eating': 15, 'drinking': 16, 'grooming': 17,
            # Alert behaviors  
            'alert': 18, 'sniffing': 19, 'looking': 20
        }
    
    def _awa_pose_labels(self):
        """AwA Pose dataset - 39 keypoints for quadrupeds"""
        # Canonical poses derived from visual analysis
        return {
            'sitting': 'sit', 'lying': 'down', 'standing': 'stand',
            'walking': 'walking', 'running': 'running', 'eating': 'eating',
            'drinking': 'drinking', 'grooming': 'grooming'
        }
    
    def _animal_activity_labels(self):
        """Animal Activity dataset - behavior classification from images"""
        return {
            'sitting': 'sit', 'standing': 'stand', 'eating': 'eating',
            'lying': 'down', 'walking': 'walking', 'drinking': 'drinking',
            'playing': 'alert', 'sleeping': 'lying'
        }
    
    def _cear_labels(self):
        """CEAR quadruped robot dataset - IMU + joint encoders"""
        return {
            'stationary': 'stand', 'walking': 'walking', 'trotting': 'trotting',
            'running': 'running', 'turning': 'turning', 'sitting': 'sit',
            'lying': 'down'
        }
    
    def map_to_canonical(self, original_label, dataset_name):
        """Map dataset-specific labels to canonical quadruped taxonomy"""
        dataset_labels = self.label_mappings[dataset_name]
        descriptive = dataset_labels.get(original_label, 'unknown')
        return self.canonical_labels.get(descriptive, len(self.canonical_labels))
    
    def create_synthetic_awa_pose(self):
        """Create synthetic AwA-style pose sequences from IMU-equivalent data"""
        print("Creating synthetic AwA Pose quadruped data...")
        np.random.seed(44)
        n_samples = 8000
        
        data = []
        labels = []
        
        # Define quadruped behaviors with distinctive motion patterns
        behaviors = {
            'sit': {'pattern': [0, 0, 9.8, 0, 0, 0, 0, 0, 0], 'std': [0.3, 0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]},
            'down': {'pattern': [0, 0, 9.8, 0, 0, 0, 0, 0, 0], 'std': [0.2, 0.2, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02]},
            'stand': {'pattern': [0, 0, 9.8, 0, 0, 0, 0, 0, 0], 'std': [0.8, 0.8, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]},
            'walking': {'pattern': [0, 2, 9.5, 1, 0, 0.5, 0, 0, 0], 'std': [2, 1.5, 1, 1, 0.8, 0.8, 0.3, 0.3, 0.3]},
            'eating': {'pattern': [0, -1, 8.5, 0.5, 0, 0, 0, 0, 0], 'std': [1, 0.8, 0.5, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1]},
            'sit_to_stand': {'pattern': [0, 1, 9.2, 2, 1, 1, 0, 0, 0], 'std': [3, 2, 1.5, 2, 1.5, 1.5, 0.5, 0.5, 0.5]},
            'stand_to_sit': {'pattern': [0, -1, 9.2, -1.5, -1, -0.5, 0, 0, 0], 'std': [2.5, 2, 1.2, 1.8, 1.2, 1.2, 0.4, 0.4, 0.4]},
        }
        
        for behavior, config in behaviors.items():
            n_behavior_samples = n_samples // len(behaviors)
            
            # Generate samples with temporal correlations for realistic sequences
            behavior_data = []
            for _ in range(n_behavior_samples):
                # Add temporal smoothing for realistic motion
                sample = np.random.normal(config['pattern'], config['std'])
                # Add small temporal drift
                drift = np.random.normal(0, 0.1, 9)
                final_sample = sample + drift
                behavior_data.append(final_sample)
            
            data.append(np.array(behavior_data))
            labels.extend([self.canonical_labels.get(behavior, 0)] * n_behavior_samples)
        
        return np.vstack(data), np.array(labels)
    
    def create_synthetic_animal_activity(self):
        """Create synthetic Animal Activity dataset - image-derived behaviors"""
        print("Creating synthetic Animal Activity data...")
        np.random.seed(45)
        n_samples = 6000
        
        data = []
        labels = []
        
        # Image-derived behaviors (translated to IMU equivalents)
        activities = {
            'sitting': {'base': [0, 0, 9.8, 0, 0, 0, 0, 0, 0], 'variation': 0.3},
            'standing': {'base': [0, 0, 9.8, 0, 0, 0, 0, 0, 0], 'variation': 0.8},
            'eating': {'base': [0, -1, 8.5, 0.3, 0, 0, 0, 0, 0], 'variation': 0.6},
            'lying': {'base': [0, 0, 9.8, 0, 0, 0, 0, 0, 0], 'variation': 0.1},
            'drinking': {'base': [0, -0.5, 8.8, 0.2, 0, 0, 0, 0, 0], 'variation': 0.4},
        }
        
        for activity, config in activities.items():
            n_activity_samples = n_samples // len(activities)
            
            activity_data = np.random.normal(
                config['base'], 
                [config['variation']] * 9, 
                (n_activity_samples, 9)
            )
            
            data.append(activity_data)
            labels.extend([self.canonical_labels.get(activity, 0)] * n_activity_samples)
        
        return np.vstack(data), np.array(labels)
    
    def create_synthetic_cear(self):
        """Create synthetic CEAR quadruped robot data - IMU + joint encoders"""
        print("Creating synthetic CEAR quadruped robot data...")
        np.random.seed(46)
        n_samples = 10000
        
        data = []
        labels = []
        
        # Robot gaits with IMU signatures
        gaits = {
            'walking': {'pattern': [0, 1.5, 9.2, 1.2, 0, 0.8, 0, 0, 0], 'std': [1.5, 1, 0.8, 1, 0.8, 0.8, 0.2, 0.2, 0.2]},
            'trotting': {'pattern': [0, 2.5, 9.0, 2, 0, 1.5, 0, 0, 0], 'std': [2.5, 1.5, 1, 1.8, 1.2, 1.2, 0.4, 0.4, 0.4]},
            'running': {'pattern': [0, 4, 8.5, 3, 0, 2.5, 0, 0, 0], 'std': [3.5, 2, 1.5, 2.5, 1.8, 1.8, 0.6, 0.6, 0.6]},
            'turning': {'pattern': [2, 1, 9.2, 1, 2, 1, 1, 0, 0], 'std': [2, 1.5, 0.8, 1.2, 1.8, 1, 0.8, 0.3, 0.3]},
            'stationary': {'pattern': [0, 0, 9.8, 0, 0, 0, 0, 0, 0], 'std': [0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]},
        }
        
        for gait, config in gaits.items():
            n_gait_samples = n_samples // len(gaits)
            
            # Add periodic components for gait rhythms
            gait_data = []
            for i in range(n_gait_samples):
                base_sample = np.random.normal(config['pattern'], config['std'])
                
                # Add gait periodicity
                if gait in ['walking', 'trotting', 'running']:
                    phase = 2 * np.pi * i / 50  # ~50 sample gait cycle
                    periodic = np.sin(phase) * 0.5  # Periodic component
                    base_sample[1] += periodic  # Add to Y-axis acceleration
                    base_sample[4] += periodic * 0.3  # Add to gyro Y
                
                gait_data.append(base_sample)
            
            data.append(np.array(gait_data))
            labels.extend([self.canonical_labels.get(gait, 0)] * n_gait_samples)
        
        return np.vstack(data), np.array(labels)
    
    def create_windows(self, data, labels, dataset_name):
        """Create sliding windows with overlap"""
        windows = []
        window_labels = []
        domain_labels = []
        
        step_size = int(self.window_size * (1 - self.overlap))
        
        for i in range(0, len(data) - self.window_size + 1, step_size):
            window = data[i:i + self.window_size]
            label = labels[i + self.window_size // 2]  # Center label
            
            windows.append(window)
            window_labels.append(label)
            domain_labels.append(dataset_name)
        
        return np.array(windows), np.array(window_labels), np.array(domain_labels)
    
    def preprocess_all_quadruped(self):
        """Comprehensive quadruped preprocessing pipeline"""
        all_windows = []
        all_labels = []
        all_domains = []
        
        print("=== Loading Quadruped Datasets ===")
        
        # AwA Pose (synthetic keypoint-derived IMU)
        try:
            awa_data, awa_labels = self.create_synthetic_awa_pose()
            windows, labels, domains = self.create_windows(awa_data, awa_labels, 'awa_pose')
            all_windows.append(windows)
            all_labels.append(labels)
            all_domains.append(domains)
            print(f"AwA Pose: {len(windows)} windows")
        except Exception as e:
            print(f"Error with AwA Pose: {e}")
        
        # Animal Activity (synthetic image-derived behaviors)
        try:
            animal_data, animal_labels = self.create_synthetic_animal_activity()
            windows, labels, domains = self.create_windows(animal_data, animal_labels, 'animal_activity')
            all_windows.append(windows)
            all_labels.append(labels)
            all_domains.append(domains)
            print(f"Animal Activity: {len(windows)} windows")
        except Exception as e:
            print(f"Error with Animal Activity: {e}")
        
        # CEAR Quadruped Robot (synthetic IMU + joint encoders)
        try:
            cear_data, cear_labels = self.create_synthetic_cear()
            windows, labels, domains = self.create_windows(cear_data, cear_labels, 'cear_quadruped')
            all_windows.append(windows)
            all_labels.append(labels)
            all_domains.append(domains)
            print(f"CEAR Quadruped: {len(windows)} windows")
        except Exception as e:
            print(f"Error with CEAR: {e}")
        
        if not all_windows:
            raise ValueError("No quadruped datasets loaded successfully!")
        
        # Combine all datasets
        X_combined = np.vstack(all_windows)
        y_combined = np.hstack(all_labels)
        domains_combined = np.hstack(all_domains)
        
        print(f"\\n=== Quadruped Dataset Statistics ===")
        print(f"Total windows: {len(X_combined)}")
        print(f"Unique behaviors: {len(np.unique(y_combined))}")
        print(f"Datasets: {np.unique(domains_combined)}")
        
        # Remap labels to be contiguous
        unique_labels = np.unique(y_combined)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        y_remapped = np.array([label_mapping[label] for label in y_combined])
        
        print(f"Label remapping: {label_mapping}")
        
        # Encode domains
        domain_encoder = LabelEncoder()
        domains_encoded = domain_encoder.fit_transform(domains_combined)
        
        # Normalize features
        scaler = StandardScaler()
        n_samples, n_timesteps, n_features = X_combined.shape
        X_reshaped = X_combined.reshape(-1, n_features)
        X_scaled = scaler.fit_transform(X_reshaped)
        X_normalized = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Train/validation split stratified by domain
        X_train, X_val, y_train, y_val, domains_train, domains_val = train_test_split(
            X_normalized, y_remapped, domains_encoded, 
            test_size=0.2, random_state=42, stratify=domains_encoded
        )
        
        print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
        
        # Save encoders
        self.domain_encoder = domain_encoder
        self.scaler = scaler
        self.label_mapping = label_mapping
        
        return X_train, y_train, domains_train, X_val, y_val, domains_val


# Usage example
if __name__ == "__main__":
    processor = QuadrupedDatasetHAR(window_size=100, overlap=0.5)
    X_train, y_train, domains_train, X_val, y_val, domains_val = processor.preprocess_all_quadruped()
    print(f"Quadruped dataset ready: {X_train.shape} training samples")
    print(f"Behavior classes: {len(np.unique(y_train))}")