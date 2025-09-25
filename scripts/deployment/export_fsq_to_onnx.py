#!/usr/bin/env python3
"""
Export Conv2d-FSQ model to ONNX for Hailo-8 compilation
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add models to path
sys.path.append(str(Path(__file__).parent))

from models.conv2d_fsq_model import Conv2dFSQ

def export_fsq_model():
    """Export the trained FSQ model to ONNX format"""
    
    # Load the trained model
    checkpoint_path = Path("models/conv2d_fsq_trained_20250921_225014.pth")
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    # Initialize model (matching training config)
    model = Conv2dFSQ(
        input_channels=9,      # IMU channels
        hidden_dim=128,        # Encoder output
        num_classes=10,        # Behaviors
        fsq_levels=[8,6,5,5,4]  # 4800 unique codes
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Loaded model from {checkpoint_path}")
    
    # Create dummy input matching expected shape
    # Shape: (batch, channels, height, width) = (1, 9, 2, 100)
    dummy_input = torch.randn(1, 9, 2, 100)
    
    # Export to ONNX
    output_path = Path("hailo_export/conv2d_fsq_hailo8.onnx")
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"📦 Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['imu_input'],
        output_names=['behavior_output', 'codes'],
        dynamic_axes={
            'imu_input': {0: 'batch_size'},
            'behavior_output': {0: 'batch_size'},
            'codes': {0: 'batch_size'}
        }
    )
    
    # Verify export
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ ONNX export successful!")
        print(f"   File: {output_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Input shape: (batch, 9, 2, 100)")
        print(f"   Architecture: Conv2d-FSQ (Hailo-compatible)")
        return True
    else:
        print(f"❌ ONNX export failed")
        return False

if __name__ == "__main__":
    success = export_fsq_model()
    sys.exit(0 if success else 1)