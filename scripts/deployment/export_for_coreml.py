#!/usr/bin/env python3
"""
Export PyTorch model to ONNX format as an intermediate step for CoreML conversion.
The actual CoreML conversion should be done on macOS.
"""

import torch
import torch.nn as nn
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
        enc_out_len = sequence_length // 4
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
    """Simplified model for inference only (no VAE sampling, just use mean)"""
    
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model.encoder
        self.fc_mu = base_model.fc_mu
        self.classifier = base_model.classifier
    
    def forward(self, x):
        # Encode to latent space
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        
        # Classify from latent mean
        logits = self.classifier(mu)
        probs = torch.softmax(logits, dim=-1)
        return probs


def export_to_onnx():
    """Export the model to ONNX format"""
    
    # Model parameters
    n_keypoints = 24
    n_coords = 2
    sequence_length = 100
    input_channels = n_keypoints * n_coords  # 48
    
    # Paths
    model_path = Path("models.old_20250913_134407/stanford_dogs_integration/best_stanford_dogs_model.pth")
    output_dir = Path("export")
    output_dir.mkdir(exist_ok=True)
    
    print("Dog Behavior Model Export for iOS")
    print("=" * 60)
    
    # Load PyTorch model
    print(f"Loading model from {model_path}")
    model = MultiModalTCNVAE(
        input_channels=input_channels,
        sequence_length=sequence_length,
        latent_dim=32,
        num_classes=21
    )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create inference-only version
    inference_model = InferenceOnlyModel(model)
    inference_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, input_channels, sequence_length)
    
    # Export to ONNX
    onnx_path = output_dir / "dog_behavior_classifier.onnx"
    print(f"\nExporting to ONNX: {onnx_path}")
    
    torch.onnx.export(
        inference_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pose_sequence'],
        output_names=['behavior_probs'],
        dynamic_axes={
            'pose_sequence': {0: 'batch_size'},
            'behavior_probs': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model exported: {onnx_path}")
    
    # Also export the full model state for reference
    torch_path = output_dir / "dog_behavior_classifier.pth"
    torch.save({
        'model_state_dict': inference_model.state_dict(),
        'input_channels': input_channels,
        'sequence_length': sequence_length,
        'n_keypoints': n_keypoints,
        'class_labels': [
            'sit', 'down', 'stand', 'stay', 'lying',
            'heel', 'come', 'fetch', 'drop', 'wait',
            'leave_it', 'walking', 'trotting', 'running',
            'jumping', 'spinning', 'rolling', 'playing',
            'alert', 'sniffing', 'looking'
        ]
    }, torch_path)
    
    print(f"✅ PyTorch model exported: {torch_path}")
    
    # Create metadata file
    metadata = {
        "model_info": {
            "name": "DogBehaviorClassifier",
            "version": "1.0",
            "description": "24-point dog pose to behavior classification",
            "trained_on": "Stanford Dogs Dataset",
            "accuracy": "94.46%"
        },
        "input": {
            "format": "pose_sequence",
            "shape": [1, 48, 100],
            "description": "24 keypoints (x,y) over 100 frames",
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "throat", "withers", "left_front_shoulder", "left_front_elbow", "left_front_paw",
                "right_front_shoulder", "right_front_elbow", "right_front_paw",
                "center", "left_hip", "left_knee", "left_back_paw",
                "right_hip", "right_knee", "right_back_paw",
                "tail_base", "tail_mid_1", "tail_mid_2", "tail_tip"
            ]
        },
        "output": {
            "format": "behavior_probs",
            "shape": [1, 21],
            "classes": [
                "sit", "down", "stand", "stay", "lying",
                "heel", "come", "fetch", "drop", "wait",
                "leave_it", "walking", "trotting", "running",
                "jumping", "spinning", "rolling", "playing",
                "alert", "sniffing", "looking"
            ]
        },
        "conversion_instructions": {
            "coreml": "Use coremltools on macOS: coremltools.convert(onnx_model, convert_to='mlprogram')",
            "ios_deployment": "Minimum iOS 14.0, recommended iOS 15.0+",
            "preprocessing": "Normalize keypoints to [0,1] range based on image dimensions"
        }
    }
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata exported: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("\nNext steps for iOS deployment:")
    print("1. Transfer the ONNX file to macOS")
    print("2. Install coremltools: pip install coremltools")
    print("3. Convert to CoreML:")
    print("   import coremltools as ct")
    print("   model = ct.convert('dog_behavior_classifier.onnx', convert_to='mlprogram')")
    print("   model.save('DogBehaviorClassifier.mlpackage')")
    print("4. Add to Xcode project")
    print("5. Use with Vision framework for inference")
    
    # Test the exported model
    print("\n" + "=" * 60)
    print("Testing exported model...")
    
    with torch.no_grad():
        test_input = torch.randn(1, 48, 100)
        output = inference_model(test_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Output sum: {output.sum().item():.4f} (should be ~1.0)")
        
        # Get top 3 predictions
        probs, indices = torch.topk(output[0], 3)
        class_labels = metadata['output']['classes']
        
        print("\nTop 3 predictions on random input:")
        for i, (prob, idx) in enumerate(zip(probs, indices)):
            print(f"  {i+1}. {class_labels[idx]}: {prob.item():.2%}")


if __name__ == "__main__":
    export_to_onnx()