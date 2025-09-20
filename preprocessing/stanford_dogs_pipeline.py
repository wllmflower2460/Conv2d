import json
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any

class StanfordDogsDataset:
    """Stanford Dogs dataset with pose keypoints for TCN-VAE training"""
    
    def __init__(self, root_dir='datasets/stanford_dogs', window_size=100, overlap=0.5):
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.overlap = overlap
        self.data_loaded = False
        
        # Stanford Dogs has 24 keypoint slots (20 commonly used + 4 tail points)
        self.n_keypoints = 24
        # Match CVAT 24-point structure exactly
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'throat', 'withers', 'left_front_shoulder', 'left_front_elbow', 'left_front_paw',
            'right_front_shoulder', 'right_front_elbow', 'right_front_paw',
            'center', 'left_hip', 'left_knee', 'left_back_paw',
            'right_hip', 'right_knee', 'right_back_paw',
            'tail_base', 'tail_mid_1', 'tail_mid_2', 'tail_tip'
        ]
        
        # Map breed groups to behavioral categories
        self.breed_to_behavior = self._create_breed_behavior_mapping()
        
        # Canonical behavior labels matching quadruped pipeline
        self.canonical_labels = {
            'sit': 0, 'down': 1, 'stand': 2, 'stay': 3, 'lying': 4,
            'walking': 11, 'trotting': 12, 'running': 13,
            'alert': 18, 'sniffing': 19, 'looking': 20
        }
        
    def _create_breed_behavior_mapping(self):
        """Map dog breeds to likely behavioral patterns"""
        # Simplified mapping - in practice, behaviors would be annotated
        return {
            'working': ['stand', 'walking', 'alert'],
            'toy': ['sit', 'lying', 'looking'],
            'sporting': ['running', 'trotting', 'alert'],
            'hound': ['sniffing', 'walking', 'alert'],
            'terrier': ['alert', 'running', 'stand'],
            'herding': ['running', 'walking', 'alert'],
            'non-sporting': ['sit', 'stand', 'walking']
        }
    
    def load_data(self, split='train'):
        """Load Stanford Dogs keypoint data"""
        print(f"Loading Stanford Dogs {split} data...")
        
        # Load annotations
        json_path = self.root_dir / 'StanfordExtra_v12.json'
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        # Load split indices
        if split == 'train':
            indices = np.load(self.root_dir / 'train_stanford_StanfordExtra_v12.npy')
        elif split == 'val':
            indices = np.load(self.root_dir / 'val_stanford_StanfordExtra_v12.npy')
        else:  # test
            indices = np.load(self.root_dir / 'test_stanford_StanfordExtra_v12.npy')
        
        # Extract keypoint sequences
        sequences = []
        labels = []
        
        for idx in indices:
            if idx >= len(annotations):
                continue
                
            ann = annotations[idx]
            
            # Extract keypoints (joints field contains [x, y, visibility])
            keypoints = np.array(ann['joints'])[:self.n_keypoints]  # Take all 24
            
            # Normalize keypoints relative to image dimensions
            img_w, img_h = ann['img_width'], ann['img_height']
            keypoints[:, 0] /= img_w  # Normalize x
            keypoints[:, 1] /= img_h  # Normalize y
            
            # Create feature vector from keypoints
            # Using relative positions and angles for TCN-VAE
            features = self._extract_pose_features(keypoints)
            
            # For now, assign pseudo-labels based on pose characteristics
            # In practice, these would come from behavior annotations
            label = self._infer_behavior_from_pose(keypoints)
            
            sequences.append(features)
            labels.append(label)
        
        self.data = np.array(sequences)
        self.labels = np.array(labels)
        self.data_loaded = True
        
        print(f"Loaded {len(self.data)} samples from Stanford Dogs {split}")
        return self.data, self.labels
    
    def _extract_pose_features(self, keypoints):
        """Extract pose features for TCN-VAE input"""
        features = []
        
        # 1. Normalized keypoint positions (x, y for each keypoint)
        for kp in keypoints:
            features.extend([kp[0], kp[1]])  # x, y positions
        
        # 2. Key angles for quadruped pose
        # Spine angle (nose to tail base)
        spine_angle = self._compute_angle(keypoints[2], keypoints[7])  # nose to tail_base
        features.append(spine_angle)
        
        # Front leg angles
        left_front_angle = self._compute_angle(keypoints[10], keypoints[8])  # elbow to paw
        right_front_angle = self._compute_angle(keypoints[13], keypoints[11])
        features.extend([left_front_angle, right_front_angle])
        
        # Rear leg angles
        left_rear_angle = self._compute_angle(keypoints[16], keypoints[14])  # elbow to paw
        right_rear_angle = self._compute_angle(keypoints[19], keypoints[17])
        features.extend([left_rear_angle, right_rear_angle])
        
        # 3. Body posture metrics
        # Height ratio (withers height relative to body length)
        body_length = np.linalg.norm(keypoints[2][:2] - keypoints[7][:2])  # nose to tail
        withers_height = keypoints[6][1]  # y-coordinate of withers
        height_ratio = withers_height / (body_length + 1e-6)
        features.append(height_ratio)
        
        # Center of mass approximation
        com_x = np.mean([kp[0] for kp in keypoints])
        com_y = np.mean([kp[1] for kp in keypoints])
        features.extend([com_x, com_y])
        
        return np.array(features)
    
    def _compute_angle(self, point1, point2):
        """Compute angle between two points"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return np.arctan2(dy, dx)
    
    def _infer_behavior_from_pose(self, keypoints):
        """Infer behavior from pose (simplified heuristic)"""
        # This is a placeholder - real labels would come from annotations
        
        # Compute some pose metrics
        withers_y = keypoints[6][1]  # withers height
        paws_y = np.mean([keypoints[8][1], keypoints[11][1], 
                          keypoints[14][1], keypoints[17][1]])  # avg paw height
        
        # Simple heuristics
        if withers_y > 0.6 and abs(withers_y - paws_y) < 0.2:
            return self.canonical_labels['stand']
        elif withers_y < 0.4:
            return self.canonical_labels['down']
        elif 0.4 <= withers_y <= 0.6:
            return self.canonical_labels['sit']
        else:
            return self.canonical_labels['stand']
    
    def create_windows(self, data, labels):
        """Create sliding windows for temporal modeling"""
        if len(data.shape) == 2:
            # Single frame data - replicate to create temporal dimension
            # This handles static pose images
            n_samples = data.shape[0]
            windowed_data = []
            windowed_labels = []
            
            for i in range(n_samples):
                # Create synthetic temporal window by adding small variations
                window = np.tile(data[i], (self.window_size, 1))
                # Add small temporal noise for variation
                window += np.random.normal(0, 0.01, window.shape)
                windowed_data.append(window)
                windowed_labels.append(labels[i])
            
            return np.array(windowed_data), np.array(windowed_labels)
        
        # For actual temporal sequences (future enhancement)
        step = int(self.window_size * (1 - self.overlap))
        windows = []
        window_labels = []
        
        for i in range(0, len(data) - self.window_size + 1, step):
            window = data[i:i + self.window_size]
            windows.append(window)
            # Use majority label in window
            label = np.bincount(labels[i:i + self.window_size]).argmax()
            window_labels.append(label)
        
        return np.array(windows), np.array(window_labels)
    
    def get_data_loaders(self, batch_size=32, test_size=0.2):
        """Get PyTorch data loaders for training"""
        if not self.data_loaded:
            self.load_data('train')
        
        # Create windows
        X_windowed, y_windowed = self.create_windows(self.data, self.labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_windowed, y_windowed, test_size=test_size, random_state=42
        )
        
        # Normalize features
        scaler = StandardScaler()
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(n_samples, n_timesteps, n_features)
        
        n_samples_val = X_val.shape[0]
        X_val_flat = X_val.reshape(-1, n_features)
        X_val_flat = scaler.transform(X_val_flat)
        X_val = X_val_flat.reshape(n_samples_val, n_timesteps, n_features)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, scaler
    
    def get_combined_with_other_datasets(self, other_loaders):
        """Combine Stanford Dogs with other quadruped datasets"""
        # This method would merge Stanford Dogs pose data with
        # IMU-based datasets for multi-modal learning
        pass


class StanfordDogsVideoDataset(StanfordDogsDataset):
    """Extended dataset for video sequences (future enhancement)"""
    
    def __init__(self, root_dir='datasets/stanford_dogs', window_size=100, overlap=0.5, fps=30):
        super().__init__(root_dir, window_size, overlap)
        self.fps = fps
        
    def load_video_sequences(self, video_dir):
        """Load pose sequences from video annotations"""
        # Future: Load temporal pose sequences from video
        # This would enable true temporal behavior modeling
        pass


if __name__ == "__main__":
    # Test the dataset loader
    dataset = StanfordDogsDataset()
    data, labels = dataset.load_data('train')
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Get data loaders
    train_loader, val_loader, scaler = dataset.get_data_loaders(batch_size=32)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Check a batch
    for batch_data, batch_labels in train_loader:
        print(f"Batch shape: {batch_data.shape}")
        print(f"Labels in batch: {batch_labels[:10]}")
        break