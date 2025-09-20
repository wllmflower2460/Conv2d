"""
Enhanced Data Pipeline with Dual Approach Support
1. Cross-Species Behavioral Analysis with Hailo Conv2d compatibility (primary)
2. Traditional HAR Multi-Dataset approach (fallback)
"""

import pandas as pd
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline  # Moved to module level for performance
import warnings
import glob
import os
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# PRIMARY APPROACH: Cross-Species with Conv2d for Hailo
# ================================================================================

class EnhancedCrossSpeciesDataset(Dataset):
    """
    Enhanced dataset for cross-species transfer learning with phone+IMU support
    Implements Hailo-compatible data loading with static shapes
    """
    
    def __init__(self, config_path: str, mode: str = 'train', 
                 enforce_hailo_constraints: bool = True):
        """
        Initialize enhanced dataset with YAML configuration
        
        Args:
            config_path: Path to enhanced YAML configuration
            mode: 'train', 'val', or 'test'
            enforce_hailo_constraints: Validate Hailo compatibility
        """
        self.mode = mode
        self.enforce_hailo = enforce_hailo_constraints
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize data containers
        self.phone_data = []
        self.imu_data = []
        self.human_labels = []
        self.dog_labels = []
        self.sample_metadata = []
        
        # Load cross-species mappings
        self._load_cross_species_mappings()
        
        # Setup preprocessing
        self.phone_scaler = StandardScaler()
        self.imu_scaler = StandardScaler()
        
        # Load and process data
        self._load_data()
        
    def _validate_config(self):
        """Validate configuration for Hailo compatibility"""
        if self.enforce_hailo:
            constraints = self.config['hailo_deployment']['architecture_constraints']
            io_spec = self.config['hailo_deployment']['io_specification']
            
            # Check input shape
            expected_shape = io_spec['input_shape']
            assert len(expected_shape) == 4, f"Input must be 4D tensor, got {len(expected_shape)}D"
            assert expected_shape[2] == 2, f"Device dimension must be 2 (phone+IMU), got {expected_shape[2]}"
            
            # Check for unsupported operations
            unsupported = constraints['unsupported_ops']
            logger.info(f"✅ Avoiding unsupported operations: {unsupported}")
            
            # Verify static shapes
            assert io_spec['static_shape_required'], "Hailo requires static shapes"
            
            logger.info("✅ Configuration passes Hailo compatibility checks")
    
    def _load_cross_species_mappings(self):
        """Load behavioral mappings for cross-species transfer"""
        mappings = self.config['cross_species_mapping']['behavioral_correspondences']
        
        self.activity_to_behavior = {}
        self.behavior_to_activity = {}
        
        for mapping in mappings:
            source = mapping['source_activity']
            target = mapping['target_behavior']
            confidence = mapping['confidence']
            
            # Only use high-confidence mappings
            threshold = self.config['cross_species_mapping']['mapping_confidence_threshold']
            if confidence >= threshold:
                self.activity_to_behavior[source] = {
                    'target': target,
                    'confidence': confidence
                }
                
                if target not in self.behavior_to_activity:
                    self.behavior_to_activity[target] = []
                self.behavior_to_activity[target].append(source)
        
        logger.info(f"✅ Loaded {len(self.activity_to_behavior)} cross-species mappings")
    
    def _load_data(self):
        """Load and preprocess data based on configuration"""
        # This would load actual PAMAP2/WISDM data
        # For now, generate synthetic data for testing
        self._generate_synthetic_data()
        
        # Apply preprocessing
        self._preprocess_data()
        
        # Split data
        self._split_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic phone+IMU data for testing"""
        n_samples = 1000
        sequence_length = self.config['hailo_deployment']['io_specification']['input_shape'][3]
        n_channels = self.config['hailo_deployment']['io_specification']['input_shape'][1]
        
        # Generate phone sensor data (pocket placement)
        phone_samples = []
        for _ in range(n_samples):
            # Simulate different activity patterns
            activity_id = np.random.randint(0, 12)
            
            # Generate IMU signals with activity-specific patterns
            if activity_id in [0, 1, 2]:  # Static postures
                signal = np.random.randn(n_channels, sequence_length) * 0.1
            elif activity_id in [3, 4]:  # Walking/running
                t = np.linspace(0, 2*np.pi, sequence_length)
                signal = np.sin(t * (activity_id + 1)) + np.random.randn(n_channels, sequence_length) * 0.2
            else:  # Other activities
                signal = np.random.randn(n_channels, sequence_length) * 0.5
            
            phone_samples.append(signal)
        
        # Generate collar IMU data (chest/collar placement)
        imu_samples = []
        for i in range(n_samples):
            # Collar IMU has different characteristics than phone
            base_signal = phone_samples[i].copy()
            
            # Add placement-specific variations
            collar_variation = np.random.randn(*base_signal.shape) * 0.3
            collar_signal = base_signal * 0.8 + collar_variation
            
            imu_samples.append(collar_signal)
        
        # Generate labels
        human_activities = ['lying', 'sitting', 'standing', 'walking', 'running', 
                          'cycling', 'ascending_stairs', 'descending_stairs',
                          'computer_work', 'vacuum_cleaning', 'ironing', 'house_cleaning']
        
        for i in range(n_samples):
            activity_id = i % len(human_activities)
            activity = human_activities[activity_id]
            
            # Map to dog behavior if available
            if activity in self.activity_to_behavior:
                dog_behavior = self.activity_to_behavior[activity]['target']
                dog_label = ['sit', 'down', 'stand'].index(dog_behavior) if dog_behavior in ['sit', 'down', 'stand'] else -1
            else:
                dog_label = -1  # No mapping available
            
            self.phone_data.append(phone_samples[i])
            self.imu_data.append(imu_samples[i])
            self.human_labels.append(activity_id)
            self.dog_labels.append(dog_label)
            
            # Store metadata
            self.sample_metadata.append({
                'human_activity': activity,
                'dog_behavior': dog_behavior if activity in self.activity_to_behavior else None,
                'confidence': self.activity_to_behavior[activity]['confidence'] if activity in self.activity_to_behavior else 0
            })
        
        logger.info(f"✅ Generated {n_samples} synthetic samples for testing")
    
    def _preprocess_data(self):
        """Apply preprocessing pipeline from configuration"""
        preprocessing_steps = self.config['data_pipeline']['preprocessing']
        
        for step in preprocessing_steps:
            if step['step'] == 'normalization':
                # Normalize phone and IMU data separately
                phone_flat = np.array(self.phone_data).reshape(-1, 9)
                imu_flat = np.array(self.imu_data).reshape(-1, 9)
                
                phone_norm = self.phone_scaler.fit_transform(phone_flat)
                imu_norm = self.imu_scaler.fit_transform(imu_flat)
                
                # Reshape back
                n_samples = len(self.phone_data)
                seq_len = self.config['hailo_deployment']['io_specification']['input_shape'][3]
                
                self.phone_data = phone_norm.reshape(n_samples, 9, seq_len)
                self.imu_data = imu_norm.reshape(n_samples, 9, seq_len)
                
                logger.info("✅ Applied normalization")
                
            elif step['step'] == 'augmentation' and step.get('enabled', False):
                # Apply data augmentation for training set
                if self.mode == 'train':
                    self._apply_augmentation(step['techniques'])
    
    def _apply_augmentation(self, techniques: Dict):
        """Apply data augmentation techniques"""
        augmented_phone = []
        augmented_imu = []
        augmented_human_labels = []
        augmented_dog_labels = []
        
        for i in range(len(self.phone_data)):
            # Original sample
            augmented_phone.append(self.phone_data[i])
            augmented_imu.append(self.imu_data[i])
            augmented_human_labels.append(self.human_labels[i])
            augmented_dog_labels.append(self.dog_labels[i])
            
            # Time warping
            if techniques.get('time_warping', {}).get('enabled', False):
                warped_phone = self._time_warp(self.phone_data[i], techniques['time_warping']['sigma'])
                warped_imu = self._time_warp(self.imu_data[i], techniques['time_warping']['sigma'])
                
                augmented_phone.append(warped_phone)
                augmented_imu.append(warped_imu)
                augmented_human_labels.append(self.human_labels[i])
                augmented_dog_labels.append(self.dog_labels[i])
            
            # Magnitude warping
            if techniques.get('magnitude_warping', {}).get('enabled', False):
                warped_phone = self._magnitude_warp(self.phone_data[i], techniques['magnitude_warping']['sigma'])
                warped_imu = self._magnitude_warp(self.imu_data[i], techniques['magnitude_warping']['sigma'])
                
                augmented_phone.append(warped_phone)
                augmented_imu.append(warped_imu)
                augmented_human_labels.append(self.human_labels[i])
                augmented_dog_labels.append(self.dog_labels[i])
        
        self.phone_data = np.array(augmented_phone)
        self.imu_data = np.array(augmented_imu)
        self.human_labels = augmented_human_labels
        self.dog_labels = augmented_dog_labels
        
        logger.info(f"✅ Applied augmentation: {len(self.phone_data)} samples")
    
    def _time_warp(self, signal: np.ndarray, sigma: float) -> np.ndarray:
        """Apply time warping augmentation"""
        
        orig_steps = np.arange(signal.shape[1])
        random_points = np.random.normal(loc=1.0, scale=sigma, size=signal.shape[1])
        new_steps = np.cumsum(random_points)
        new_steps = (new_steps / new_steps[-1]) * (signal.shape[1] - 1)
        
        warped = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            cs = CubicSpline(orig_steps, signal[i, :])
            warped[i, :] = cs(new_steps)
        
        return warped
    
    def _magnitude_warp(self, signal: np.ndarray, sigma: float) -> np.ndarray:
        """Apply magnitude warping augmentation"""
        
        orig_steps = np.arange(signal.shape[1])
        random_points = np.random.normal(loc=1.0, scale=sigma, size=4)
        
        knot_points = np.linspace(0, signal.shape[1] - 1, 4)
        cs = CubicSpline(knot_points, random_points)
        warping_curve = cs(orig_steps)
        
        return signal * warping_curve[np.newaxis, :]
    
    def _split_data(self):
        """Split data according to configuration"""
        split_config = self.config['data_pipeline']['data_split']
        
        # Convert to numpy arrays
        phone_array = np.array(self.phone_data)
        imu_array = np.array(self.imu_data)
        human_array = np.array(self.human_labels)
        dog_array = np.array(self.dog_labels)
        
        # Get valid dog samples (not -1)
        valid_dog_mask = dog_array != -1
        
        # Split indices
        n_samples = len(phone_array)
        indices = np.arange(n_samples)
        
        train_size = split_config['train']
        val_size = split_config['validation']
        
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_size, stratify=human_array, random_state=42
        )
        
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size/(1-train_size), stratify=human_array[temp_idx], random_state=42
        )
        
        # Select data based on mode
        if self.mode == 'train':
            self.phone_data = phone_array[train_idx]
            self.imu_data = imu_array[train_idx]
            self.human_labels = human_array[train_idx]
            self.dog_labels = dog_array[train_idx]
        elif self.mode == 'val':
            self.phone_data = phone_array[val_idx]
            self.imu_data = imu_array[val_idx]
            self.human_labels = human_array[val_idx]
            self.dog_labels = dog_array[val_idx]
        else:  # test
            self.phone_data = phone_array[test_idx]
            self.imu_data = imu_array[test_idx]
            self.human_labels = human_array[test_idx]
            self.dog_labels = dog_array[test_idx]
        
        logger.info(f"✅ Data split - {self.mode}: {len(self.phone_data)} samples")
    
    def __len__(self) -> int:
        return len(self.phone_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with phone+IMU dual-device data
        
        Returns:
            Dictionary with:
                - input: (9, 2, 100) tensor - 9 channels, 2 devices, 100 timesteps
                - human_label: Human activity label
                - dog_label: Dog behavior label (-1 if not available)
                - metadata: Sample metadata
        """
        # Stack phone and IMU data to create dual-device tensor
        phone_tensor = torch.tensor(self.phone_data[idx], dtype=torch.float32)  # (9, 100)
        imu_tensor = torch.tensor(self.imu_data[idx], dtype=torch.float32)      # (9, 100)
        
        # Stack along device dimension: (9, 2, 100)
        dual_device = torch.stack([phone_tensor, imu_tensor], dim=1)
        
        # Validate shape for Hailo
        expected_shape = self.config['hailo_deployment']['io_specification']['input_shape'][1:]
        assert dual_device.shape == torch.Size(expected_shape), \
            f"Shape mismatch: {dual_device.shape} != {expected_shape}"
        
        return {
            'input': dual_device,
            'human_label': torch.tensor(self.human_labels[idx], dtype=torch.long),
            'dog_label': torch.tensor(self.dog_labels[idx], dtype=torch.long),
            'has_dog_label': self.dog_labels[idx] != -1
        }
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = None, 
                      num_workers: int = 4) -> DataLoader:
        """Get DataLoader for this dataset"""
        if shuffle is None:
            shuffle = (self.mode == 'train')
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )


class HailoDataValidator:
    """Validator for Hailo compatibility of data and models"""
    
    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, config: Dict) -> bool:
        """Validate tensor shape matches Hailo requirements"""
        expected = config['hailo_deployment']['io_specification']['input_shape']
        
        if len(tensor.shape) != len(expected):
            logger.error(f"Dimension mismatch: {len(tensor.shape)} != {len(expected)}")
            return False
        
        for i, (actual, exp) in enumerate(zip(tensor.shape, expected)):
            if exp != -1 and actual != exp:  # -1 means any size
                logger.error(f"Shape mismatch at dim {i}: {actual} != {exp}")
                return False
        
        return True
    
    @staticmethod
    def validate_model_ops(model: torch.nn.Module, config: Dict) -> bool:
        """Check model for unsupported operations"""
        unsupported = config['hailo_deployment']['architecture_constraints']['unsupported_ops']
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            
            if module_type in unsupported:
                logger.error(f"Unsupported operation found: {name} ({module_type})")
                return False
            
            # Check for Conv1d
            if isinstance(module, torch.nn.Conv1d):
                logger.error(f"Conv1d found at {name} - must use Conv2d for Hailo")
                return False
            
            # Check for grouped convolutions
            if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'groups'):
                if module.groups > 1 and module.groups != module.in_channels:
                    logger.error(f"Grouped convolution found at {name} - not supported")
                    return False
        
        logger.info("✅ Model passes Hailo compatibility check")
        return True


# ================================================================================
# FALLBACK APPROACH: Traditional HAR Multi-Dataset
# ================================================================================

class EnhancedMultiDatasetHAR:
    """Enhanced preprocessing pipeline with WISDM and HAPT integration"""
    
    def __init__(self, window_size=100, overlap=0.5, base_dataset_path=None):
        self.window_size = window_size  # ~1 second at 100Hz
        self.overlap = overlap
        self.datasets = {}
        
        # Configure base dataset path
        if base_dataset_path is None:
            # Default to project root + datasets
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            self.base_dataset_path = os.path.join(project_root, 'datasets')
        else:
            self.base_dataset_path = base_dataset_path
        
        # Enhanced label mappings including transitions from HAPT
        self.label_mappings = {
            'pamap2': self._pamap2_labels(),
            'uci_har': self._uci_har_labels(), 
            'tartan_imu': self._tartan_labels(),
            'wisdm': self._wisdm_labels(),
            'hapt': self._hapt_labels()
        }
        
        # Canonical label mapping as per Dataset Integration Roadmap
        self.canonical_labels = {
            'sit': 0, 'down': 1, 'stand': 2, 'stay': 3,
            'walking': 4, 'walking_upstairs': 5, 'walking_downstairs': 6,
            'sitting': 7, 'standing': 8, 'laying': 9,
            'sit_to_stand': 10, 'stand_to_sit': 11, 'sit_to_lie': 12,
            'lie_to_sit': 13, 'stand_to_lie': 14, 'lie_to_stand': 15
        }
    
    def _pamap2_labels(self):
        return {1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 
                5: 'running', 6: 'cycling', 7: 'nordic_walking', 
                9: 'watching_tv', 10: 'computer_work', 11: 'car_driving',
                12: 'ascending_stairs', 13: 'descending_stairs',
                16: 'vacuum_cleaning', 17: 'ironing', 18: 'folding_laundry',
                19: 'house_cleaning', 20: 'playing_soccer', 24: 'rope_jumping'}
    
    def _uci_har_labels(self):
        return {1: 'walking', 2: 'walking_upstairs', 3: 'walking_downstairs',
                4: 'sitting', 5: 'standing', 6: 'laying'}
    
    def _tartan_labels(self):
        return {0: 'stationary', 1: 'walking', 2: 'running', 3: 'turning'}
    
    def _wisdm_labels(self):
        # WISDM activity codes
        return {'A': 'walking', 'B': 'jogging', 'C': 'stairs', 
                'D': 'sitting', 'E': 'standing', 'F': 'typing',
                'G': 'brushing_teeth', 'H': 'eating', 'I': 'watching_tv'}
    
    def _hapt_labels(self):
        # HAPT includes basic activities + transitions
        return {1: 'walking', 2: 'walking_upstairs', 3: 'walking_downstairs',
                4: 'sitting', 5: 'standing', 6: 'laying',
                7: 'stand_to_sit', 8: 'sit_to_stand', 9: 'sit_to_lie',
                10: 'lie_to_sit', 11: 'stand_to_lie', 12: 'lie_to_stand'}
    
    def map_to_canonical(self, original_label, dataset_name):
        """Map dataset-specific labels to canonical taxonomy"""
        dataset_labels = self.label_mappings[dataset_name]
        
        # Get the descriptive label
        if dataset_name == 'wisdm':
            descriptive = dataset_labels.get(original_label, 'unknown')
        else:
            descriptive = dataset_labels.get(original_label, 'unknown')
        
        # Map to canonical labels
        canonical_mapping = {
            'lying': 'down', 'laying': 'down', 'lie_to_sit': 'down_to_sit',
            'lie_to_stand': 'down_to_stand', 'sit_to_lie': 'sit_to_down',
            'stand_to_lie': 'stand_to_down', 'sitting': 'sit',
            'standing': 'stand', 'jogging': 'walking', 'running': 'walking',
            'stairs': 'walking_upstairs', 'ascending_stairs': 'walking_upstairs',
            'descending_stairs': 'walking_downstairs', 'stationary': 'stand'
        }
        
        mapped_label = canonical_mapping.get(descriptive, descriptive)
        return self.canonical_labels.get(mapped_label, len(self.canonical_labels))
    
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
    
    # ... (include all the dataset loading methods from the master version)
    # load_wisdm, load_hapt, _create_wisdm_synthetic, _create_hapt_synthetic, etc.
    # These would be copied from the master version


# ================================================================================
# UNIFIED INTERFACE
# ================================================================================

def get_dataset(approach='cross_species', **kwargs):
    """
    Factory function to get the appropriate dataset
    
    Args:
        approach: 'cross_species' for Conv2d Hailo approach, 
                 'traditional_har' for multi-dataset HAR approach
        **kwargs: Arguments to pass to the dataset constructor
    
    Returns:
        Dataset instance
    """
    if approach == 'cross_species':
        return EnhancedCrossSpeciesDataset(**kwargs)
    elif approach == 'traditional_har':
        return EnhancedMultiDatasetHAR(**kwargs)
    else:
        raise ValueError(f"Unknown approach: {approach}")


if __name__ == "__main__":
    # Test the enhanced pipeline
    print("🧪 Testing Enhanced Cross-Species Data Pipeline...")
    
    # Default to cross-species approach
    use_cross_species = True
    
    if use_cross_species:
        # Load configuration
        config_path = "configs/enhanced_dataset_schema.yaml"
        
        # Create datasets
        print("\n📊 Creating cross-species datasets...")
        train_dataset = EnhancedCrossSpeciesDataset(config_path, mode='train')
        val_dataset = EnhancedCrossSpeciesDataset(config_path, mode='val')
        test_dataset = EnhancedCrossSpeciesDataset(config_path, mode='test')
        
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val: {len(val_dataset)} samples")
        print(f"   Test: {len(test_dataset)} samples")
        
        # Test data loading
        print("\n🔄 Testing data loading...")
        train_loader = train_dataset.get_dataloader(batch_size=4)
        
        for batch in train_loader:
            print(f"   Input shape: {batch['input'].shape}")
            print(f"   Human labels: {batch['human_label'].shape}")
            print(f"   Dog labels: {batch['dog_label'].shape}")
            break
        
        # Validate Hailo compatibility
        print("\n🔍 Validating Hailo compatibility...")
        validator = HailoDataValidator()
        sample = train_dataset[0]
        is_valid = validator.validate_tensor_shape(
            sample['input'].unsqueeze(0),  # Add batch dimension
            train_dataset.config
        )
        
        print(f"   Data shape validation: {'✅ Passed' if is_valid else '❌ Failed'}")
    else:
        # Fallback to traditional HAR approach
        print("\n📊 Using traditional HAR approach...")
        processor = EnhancedMultiDatasetHAR(window_size=100, overlap=0.5)
        # processor.preprocess_all_enhanced() would be called here
        print("Traditional HAR pipeline ready as fallback")
    
    print("\n✅ Enhanced pipeline testing complete!")