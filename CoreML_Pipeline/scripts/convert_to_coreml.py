#!/usr/bin/env python3
"""
Convert PyTorch TCN-VAE model to CoreML for iOS deployment
Supports 24-point dog pose estimation
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from pathlib import Path
import json


class MultiModalTCNVAE(nn.Module):
    """TCN-VAE model matching the trained Stanford Dogs model architecture"""
    
    def __init__(self, input_channels, sequence_length, latent_dim=32, num_classes=21):
        super().__init__()
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # Temporal Convolutional Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        
        # Calculate encoder output size
        enc_out_len = sequence_length // 4  # Due to two stride=2 convolutions
        self.flatten_size = 256 * enc_out_len
        
        # VAE latent layers
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, input_channels, kernel_size=3, stride=1, padding=1),
        )
        
        # Behavior classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, -1)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        classification = self.classifier(z)
        return recon, mu, logvar, classification


class InferenceOnlyModel(nn.Module):
    """Simplified model for inference only (no VAE sampling)"""
    
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model.encoder
        self.fc_mu = base_model.fc_mu
        self.classifier = base_model.classifier
        self.flatten_size = base_model.flatten_size
    
    def forward(self, x):
        # Encode to latent space
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        
        # Classify from latent representation
        classification = self.classifier(mu)
        return classification


def load_pytorch_model(model_path, input_channels=48, sequence_length=100):
    """Load the trained PyTorch model"""
    
    # Initialize model
    model = MultiModalTCNVAE(
        input_channels=input_channels,
        sequence_length=sequence_length,
        latent_dim=32,
        num_classes=21
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the dict is the state dict
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def convert_to_coreml(model_path, output_path, model_type='inference'):
    """Convert PyTorch model to CoreML"""
    
    print(f"Loading model from {model_path}")
    
    # For 24 keypoints with x,y coordinates
    n_keypoints = 24
    n_coords = 2  # x, y
    sequence_length = 100  # Temporal window
    input_channels = n_keypoints * n_coords  # 48 channels
    
    # Load PyTorch model
    pytorch_model = load_pytorch_model(model_path, input_channels, sequence_length)
    
    if model_type == 'inference':
        # Create inference-only version
        model = InferenceOnlyModel(pytorch_model)
        output_name = 'behavior_class'
    else:
        model = pytorch_model
        output_name = ['reconstruction', 'mu', 'logvar', 'behavior_class']
    
    model.eval()
    
    # Create dummy input for tracing
    # Shape: (batch_size, channels, sequence_length)
    dummy_input = torch.randn(1, input_channels, sequence_length)
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Define input type for CoreML
    inputs = [
        ct.TensorType(
            name='pose_sequence',
            shape=(1, input_channels, sequence_length),
            dtype=np.float32
        )
    ]
    
    # Define class labels
    class_labels = [
        'sit', 'down', 'stand', 'stay', 'lying',
        'heel', 'come', 'fetch', 'drop', 'wait',
        'leave_it', 'walking', 'trotting', 'running',
        'jumping', 'spinning', 'rolling', 'playing',
        'alert', 'sniffing', 'looking'
    ]
    
    # Convert to CoreML
    print("Converting to CoreML...")
    
    if model_type == 'inference':
        # For inference model, add classifier configuration
        classifier_config = ct.ClassifierConfig(
            class_labels=class_labels,
            predicted_feature_name='behavior_class_probs',
            predicted_probabilities_output='behavior_class_probs'
        )
        
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            classifier_config=classifier_config,
            convert_to='neuralnetwork',  # Use 'mlprogram' for iOS 15+
            minimum_deployment_target=ct.target.iOS14
        )
    else:
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            convert_to='neuralnetwork',
            minimum_deployment_target=ct.target.iOS14
        )
    
    # Add metadata
    mlmodel.author = 'TCN-VAE Dog Behavior Model'
    mlmodel.short_description = '24-point dog pose to behavior classification'
    mlmodel.version = '1.0'
    
    # Add input/output descriptions
    mlmodel.input_description['pose_sequence'] = 'Sequence of 24 keypoints (x,y) over 100 frames'
    
    if model_type == 'inference':
        mlmodel.output_description['behavior_class_probs'] = 'Probability distribution over 21 behavior classes'
        mlmodel.output_description['classLabel'] = 'Most likely behavior class'
    
    # Save the model
    mlmodel.save(output_path)
    print(f"✅ CoreML model saved to {output_path}")
    
    # Print model details
    print("\nModel Details:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Input channels: {input_channels} (24 keypoints × 2 coordinates)")
    print(f"  Sequence length: {sequence_length} frames")
    print(f"  Number of classes: {len(class_labels)}")
    print(f"  Model type: {model_type}")
    
    return mlmodel


def create_test_data():
    """Create test data matching 24-point dog skeleton"""
    
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "throat", "withers", "left_front_shoulder", "left_front_elbow", "left_front_paw",
        "right_front_shoulder", "right_front_elbow", "right_front_paw",
        "center", "left_hip", "left_knee", "left_back_paw",
        "right_hip", "right_knee", "right_back_paw",
        "tail_base", "tail_mid_1", "tail_mid_2", "tail_tip"
    ]
    
    # Create synthetic walking pattern
    sequence_length = 100
    n_keypoints = 24
    
    # Initialize pose sequence
    pose_sequence = np.zeros((sequence_length, n_keypoints, 2))
    
    # Add some realistic movement patterns
    for t in range(sequence_length):
        # Add sinusoidal movement to paws (walking pattern)
        phase = t * 0.1
        
        # Front paws alternate
        pose_sequence[t, 9, 0] = 0.3 + 0.1 * np.sin(phase)  # left_front_paw x
        pose_sequence[t, 9, 1] = 0.8 + 0.05 * np.cos(phase)  # left_front_paw y
        
        pose_sequence[t, 12, 0] = 0.7 + 0.1 * np.sin(phase + np.pi)  # right_front_paw x
        pose_sequence[t, 12, 1] = 0.8 + 0.05 * np.cos(phase + np.pi)  # right_front_paw y
        
        # Back paws alternate (with phase shift)
        pose_sequence[t, 16, 0] = 0.3 + 0.1 * np.sin(phase + np.pi/2)  # left_back_paw x
        pose_sequence[t, 16, 1] = 0.8 + 0.05 * np.cos(phase + np.pi/2)  # left_back_paw y
        
        pose_sequence[t, 19, 0] = 0.7 + 0.1 * np.sin(phase - np.pi/2)  # right_back_paw x
        pose_sequence[t, 19, 1] = 0.8 + 0.05 * np.cos(phase - np.pi/2)  # right_back_paw y
        
        # Body center stays relatively stable
        pose_sequence[t, 13, :] = [0.5, 0.5]  # center
        pose_sequence[t, 6, :] = [0.5, 0.4]   # withers
        
        # Head moves slightly
        pose_sequence[t, 0, :] = [0.5 + 0.02 * np.sin(phase * 2), 0.2]  # nose
    
    # Reshape to (channels, sequence_length)
    # Flatten keypoints: [x0, y0, x1, y1, ..., x23, y23]
    pose_flat = pose_sequence.reshape(sequence_length, -1).T  # (48, 100)
    
    return pose_flat, keypoint_names


def main():
    """Main conversion pipeline"""
    
    # Paths
    model_path = Path("models.old_20250913_134407/stanford_dogs_integration/best_stanford_dogs_model.pth")
    output_dir = Path("export")
    output_dir.mkdir(exist_ok=True)
    
    # Convert for inference (simplified model for iOS)
    coreml_path = output_dir / "DogBehaviorClassifier.mlmodel"
    
    if model_path.exists():
        print("Converting Stanford Dogs TCN-VAE model to CoreML")
        print("=" * 60)
        
        # Convert model
        mlmodel = convert_to_coreml(
            model_path,
            coreml_path,
            model_type='inference'  # Simplified for iOS
        )
        
        # Create test data
        print("\nCreating test data...")
        test_data, keypoint_names = create_test_data()
        
        # Test the converted model
        print("\nTesting CoreML model...")
        test_input = {'pose_sequence': test_data.reshape(1, 48, 100).astype(np.float32)}
        
        try:
            prediction = mlmodel.predict(test_input)
            print(f"Test prediction: {prediction['classLabel']}")
            
            # Show top 3 predictions
            if 'behavior_class_probs' in prediction:
                probs = prediction['behavior_class_probs']
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                print("\nTop 3 predictions:")
                for label, prob in sorted_probs:
                    print(f"  {label}: {prob:.2%}")
        except Exception as e:
            print(f"Test prediction failed: {e}")
        
        print(f"\n✅ Model ready for iOS deployment: {coreml_path}")
        print("\nTo use in iOS:")
        print("1. Drag the .mlmodel file into your Xcode project")
        print("2. Xcode will automatically generate a Swift class")
        print("3. Use with Vision framework for real-time inference")
        
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the Stanford Dogs model has been trained first.")


if __name__ == "__main__":
    main()