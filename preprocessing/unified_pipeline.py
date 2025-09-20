import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class MultiDatasetHAR:
    def __init__(self, window_size=100, overlap=0.5):
        self.window_size = window_size  # ~1 second at 100Hz
        self.overlap = overlap
        self.datasets = {}
        self.label_mappings = {
            'pamap2': self._pamap2_labels(),
            'uci_har': self._uci_har_labels(), 
            'tartan_imu': self._tartan_labels()
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
    
    def load_pamap2(self, data_path):
        """Load PAMAP2 dataset - 9-axis IMU data from chest sensor"""
        files = glob.glob(f"{data_path}/PAMAP2_Dataset/Protocol/*.dat")
        data_list, labels_list = [], []
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path, sep=' ', header=None, dtype=float)
                # Columns: timestamp, activity_id, heart_rate, IMU_hand(16cols), IMU_chest(17cols), IMU_ankle(17cols) 
                # Use chest IMU columns 4-12: acc_x,y,z + gyro_x,y,z + mag_x,y,z
                if df.shape[1] >= 20:  # Ensure minimum columns
                    imu_data = df.iloc[:, 4:13].values  # 9-axis from chest sensor
                    activities = df.iloc[:, 1].values
                    
                    # Filter out NaN and invalid activities (0 = transient)
                    valid_mask = (~np.isnan(imu_data).any(axis=1) & 
                                (activities > 0) & 
                                (np.isin(activities, list(self.label_mappings['pamap2'].keys()))))
                    
                    if valid_mask.sum() > 0:
                        data_list.append(imu_data[valid_mask])
                        labels_list.append(activities[valid_mask])
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if data_list:
            return np.vstack(data_list), np.hstack(labels_list)
        else:
            raise ValueError("No valid PAMAP2 data loaded")
    
    def load_uci_har(self, data_path):
        """Load UCI-HAR dataset - smartphone accelerometer + gyroscope"""
        try:
            # Load train and test data
            X_train = pd.read_csv(f"{data_path}/UCI HAR Dataset/train/X_train.txt", 
                                sep=r'\s+', header=None).values
            y_train = pd.read_csv(f"{data_path}/UCI HAR Dataset/train/y_train.txt", 
                                sep=r'\s+', header=None).values.ravel()
            
            X_test = pd.read_csv(f"{data_path}/UCI HAR Dataset/test/X_test.txt", 
                               sep=r'\s+', header=None).values  
            y_test = pd.read_csv(f"{data_path}/UCI HAR Dataset/test/y_test.txt", 
                               sep=r'\s+', header=None).values.ravel()
            
            # Combine train and test
            X_combined = np.vstack([X_train, X_test])
            y_combined = np.hstack([y_train, y_test])
            
            # UCI-HAR features are already processed - use first 6 (acc + gyro means)
            # Reshape to simulate time series: treat each sample as single timestep
            X_reshaped = X_combined[:, :6]  # Use first 6 features as pseudo-IMU
            
            return X_reshaped, y_combined
            
        except Exception as e:
            print(f"Error loading UCI-HAR: {e}")
            raise ValueError("Failed to load UCI-HAR data")
        
    def load_tartan_imu(self, data_path):
        """Load TartanIMU sample data - simplified for proof of concept"""
        try:
            # TartanIMU format is complex - create synthetic data for proof of concept
            print("Warning: Using synthetic TartanIMU data for proof of concept")
            
            # Generate synthetic 9-axis IMU data with 4 activity classes
            np.random.seed(42)
            n_samples = 10000
            
            # Simulate different motion patterns
            data = []
            labels = []
            
            for activity_id in range(4):  # 4 activities: stationary, walking, running, turning
                n_activity_samples = n_samples // 4
                
                if activity_id == 0:  # stationary
                    activity_data = np.random.normal(0, 0.1, (n_activity_samples, 9))
                elif activity_id == 1:  # walking  
                    activity_data = np.random.normal([0, 0, 9.8, 0, 0, 0, 0, 0, 0], 
                                                   [1, 1, 2, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2], 
                                                   (n_activity_samples, 9))
                elif activity_id == 2:  # running
                    activity_data = np.random.normal([0, 0, 9.8, 0, 0, 0, 0, 0, 0], 
                                                   [3, 3, 4, 2, 2, 2, 1, 1, 1], 
                                                   (n_activity_samples, 9))
                else:  # turning
                    activity_data = np.random.normal([0, 0, 9.8, 0, 0, 1, 0, 0, 0], 
                                                   [2, 2, 3, 1, 1, 3, 0.5, 0.5, 0.5], 
                                                   (n_activity_samples, 9))
                
                data.append(activity_data)
                labels.extend([activity_id] * n_activity_samples)
            
            return np.vstack(data), np.array(labels)
            
        except Exception as e:
            print(f"Error generating TartanIMU synthetic data: {e}")
            raise ValueError("Failed to generate TartanIMU data")
    
    def create_windows(self, data, labels, dataset_name):
        # Sliding window approach with overlap
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
    
    def preprocess_all(self):
        """Load all datasets and create unified training format"""
        all_windows = []
        all_labels = []
        all_domains = []
        
        # Load PAMAP2
        print("Loading PAMAP2 dataset...")
        pamap2_data, pamap2_labels = self.load_pamap2("/home/wllmflower/tcn-vae-training/datasets/pamap2")
        
        # Create windows for PAMAP2
        windows, labels, domains = self.create_windows(
            pamap2_data, pamap2_labels, 'pamap2'
        )
        all_windows.append(windows)
        all_labels.append(labels)
        all_domains.append(domains)
        print(f"PAMAP2: {len(windows)} windows")
        
        # Load UCI-HAR
        print("Loading UCI-HAR dataset...")
        uci_data, uci_labels = self.load_uci_har("/home/wllmflower/tcn-vae-training/datasets/uci_har")
        
        # UCI-HAR needs special handling - expand features to windows
        uci_windows = []
        uci_window_labels = []
        for i in range(len(uci_data)):
            # Expand each feature vector to a 100-timestep window by replication
            window = np.tile(uci_data[i:i+1], (self.window_size, 1))
            # Pad to 9 dimensions if needed
            if window.shape[1] < 9:
                padding = np.zeros((window.shape[0], 9 - window.shape[1]))
                window = np.hstack([window, padding])
            uci_windows.append(window)
            uci_window_labels.append(uci_labels[i])
        
        uci_windows = np.array(uci_windows)
        uci_window_labels = np.array(uci_window_labels) 
        uci_domains = np.full(len(uci_windows), 'uci_har', dtype=object)
        
        all_windows.append(uci_windows)
        all_labels.append(uci_window_labels)
        all_domains.append(uci_domains)
        print(f"UCI-HAR: {len(uci_windows)} windows")
        
        # Load TartanIMU  
        print("Loading TartanIMU dataset...")
        tartan_data, tartan_labels = self.load_tartan_imu("/home/wllmflower/tcn-vae-training/datasets/tartan_imu")
        
        windows, labels, domains = self.create_windows(
            tartan_data, tartan_labels, 'tartan_imu'
        )
        all_windows.append(windows)
        all_labels.append(labels)
        all_domains.append(domains)
        print(f"TartanIMU: {len(windows)} windows")
        
        # Combine all datasets
        X_combined = np.vstack(all_windows)
        y_combined = np.hstack(all_labels)
        domains_combined = np.hstack(all_domains)
        
        # Encode labels and domains
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_combined)
        
        domain_encoder = LabelEncoder() 
        domains_encoded = domain_encoder.fit_transform(domains_combined)
        
        # Normalize features
        scaler = StandardScaler()
        n_samples, n_timesteps, n_features = X_combined.shape
        X_reshaped = X_combined.reshape(-1, n_features)
        X_scaled = scaler.fit_transform(X_reshaped)
        X_normalized = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Train/validation split
        X_train, X_val, y_train, y_val, domains_train, domains_val = train_test_split(
            X_normalized, y_encoded, domains_encoded, 
            test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Total windows: {len(X_combined)}")
        print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
        print(f"Classes: {len(np.unique(y_encoded))}")
        
        # Save encoders for later use
        self.label_encoder = label_encoder
        self.domain_encoder = domain_encoder
        self.scaler = scaler
        
        return X_train, y_train, domains_train, X_val, y_val, domains_val

# Usage example
if __name__ == "__main__":
    processor = MultiDatasetHAR(window_size=100, overlap=0.5)
    X_train, y_train, domains_train, X_val, y_val, domains_val = processor.preprocess_all()