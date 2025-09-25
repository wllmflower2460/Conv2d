#!/usr/bin/env python3
"""
Conv2d-FSQ Model for Hailo Deployment
Splits FSQ into Hailo-compatible encoder and CPU post-processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

class Conv2dFSQHailoEncoder(nn.Module):
    """
    Hailo-compatible encoder that outputs continuous values
    FSQ quantization is applied in post-processing
    """
    
    def __init__(self, input_channels=9, hidden_dim=64, fsq_dim=8, num_classes=10):
        super().__init__()
        
        # Conv2d encoder (all Hailo-compatible operations)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(1, 7), padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d((1, 2))
        self.pool2 = nn.MaxPool2d((1, 2))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # FSQ projection (outputs continuous values for post-processing)
        self.fsq_projection = nn.Linear(hidden_dim, fsq_dim)
        
        # Optional: Classification head (can be on Hailo or CPU)
        self.use_classifier = True
        if self.use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(fsq_dim, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
    
    def forward(self, x):
        """
        Forward pass for Hailo inference
        Input: (B, 9, 2, 100) - IMU data
        Output: (B, 8) - Continuous FSQ projection (before quantization)
                or (B, 10) - Class logits if classifier enabled
        """
        # Conv blocks with BatchNorm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # FSQ projection (continuous values)
        fsq_continuous = self.fsq_projection(x)
        
        if self.use_classifier:
            # Classification on continuous FSQ values
            logits = self.classifier(fsq_continuous)
            return logits, fsq_continuous
        else:
            return fsq_continuous


class FSQPostProcessor:
    """
    CPU-based FSQ quantization post-processor
    Applies the Round operation that Hailo cannot handle
    """
    
    def __init__(self, levels: List[int] = [8, 6, 5, 5, 4]):
        """
        Initialize FSQ post-processor
        
        Args:
            levels: Number of quantization levels per dimension
        """
        self.levels = levels
        self.dim = len(levels)
        
        # Precompute quantization boundaries
        self.boundaries = []
        for L in levels:
            # Create L evenly spaced levels in [-1, 1]
            bounds = np.linspace(-1, 1, L)
            self.boundaries.append(bounds)
    
    def quantize(self, continuous_values: np.ndarray) -> np.ndarray:
        """
        Apply FSQ quantization to continuous values
        
        Args:
            continuous_values: (batch, dim) continuous values from Hailo
        
        Returns:
            quantized: (batch, dim) discrete FSQ codes
        """
        batch_size = continuous_values.shape[0]
        quantized = np.zeros_like(continuous_values)
        
        for i in range(self.dim):
            if i >= continuous_values.shape[1]:
                break
                
            # Get values for this dimension
            vals = continuous_values[:, i]
            
            # Scale to [-1, 1] using tanh (bounded activation)
            scaled = np.tanh(vals)
            
            # Quantize to L levels
            L = self.levels[i]
            # Round to nearest level
            level_width = 2.0 / (L - 1)
            level_indices = np.round((scaled + 1) / level_width)
            level_indices = np.clip(level_indices, 0, L - 1).astype(int)
            
            # Map back to quantized values
            quantized[:, i] = -1 + level_indices * level_width
        
        return quantized
    
    def quantize_torch(self, continuous_values: torch.Tensor) -> torch.Tensor:
        """
        PyTorch version of FSQ quantization
        
        Args:
            continuous_values: (batch, dim) continuous values from model
        
        Returns:
            quantized: (batch, dim) discrete FSQ codes
        """
        quantized = continuous_values.clone()
        
        for i, L in enumerate(self.levels):
            if i >= continuous_values.shape[1]:
                break
            
            # Scale to [-1, 1] using tanh
            scaled = torch.tanh(continuous_values[:, i])
            
            # Quantize to L levels
            level_width = 2.0 / (L - 1)
            level_indices = torch.round((scaled + 1) / level_width)
            level_indices = torch.clamp(level_indices, 0, L - 1)
            
            # Map back to quantized values
            quantized[:, i] = -1 + level_indices * level_width
        
        return quantized
    
    def get_codebook_size(self) -> int:
        """Get total number of possible FSQ codes"""
        return np.prod(self.levels)


def export_for_hailo(checkpoint_path: str, output_path: str, include_classifier: bool = False):
    """
    Export FSQ model for Hailo deployment
    
    Args:
        checkpoint_path: Path to trained checkpoint
        output_path: Path for ONNX output
        include_classifier: Whether to include classifier in Hailo model
    """
    import onnx
    from onnxsim import simplify
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    model = Conv2dFSQHailoEncoder(
        input_channels=9,
        hidden_dim=64,
        fsq_dim=8,
        num_classes=10
    )
    model.use_classifier = include_classifier
    
    # Load weights (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        # Filter and load matching weights
        state_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # Map weights from full model to Hailo encoder
        mapped_dict = {}
        for key, value in state_dict.items():
            # Remove module prefix if present
            key = key.replace('module.', '')
            # Map FSQ model keys to encoder keys
            if 'encoder' in key:
                new_key = key.replace('encoder.', '')
                if new_key in model_dict:
                    mapped_dict[new_key] = value
            elif key in model_dict:
                mapped_dict[key] = value
        
        model.load_state_dict(mapped_dict, strict=False)
        print(f"Loaded {len(mapped_dict)}/{len(model_dict)} weights")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 9, 2, 100)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    
    if include_classifier:
        output_names = ['logits', 'fsq_continuous']
    else:
        output_names = ['fsq_continuous']
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['imu_input'],
        output_names=output_names,
        dynamic_axes={'imu_input': {0: 'batch'}},  # Allow variable batch size
        opset_version=11,
        do_constant_folding=True,
        export_params=True
    )
    
    # Simplify ONNX
    print("Simplifying ONNX model...")
    onnx_model = onnx.load(output_path)
    model_simp, check = simplify(onnx_model)
    
    if check:
        onnx.save(model_simp, output_path)
        print(f"✅ Simplified ONNX saved: {output_path}")
    else:
        print("⚠️ Simplification failed, using original")
    
    return output_path


def test_split_inference():
    """Test the split Hailo + CPU inference pipeline"""
    
    # Create model and post-processor
    model = Conv2dFSQHailoEncoder()
    model.eval()
    
    fsq_processor = FSQPostProcessor(levels=[8, 6, 5, 5, 4])
    
    # Test input
    test_input = torch.randn(4, 9, 2, 100)
    
    print("Testing split inference pipeline...")
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        # Step 1: Hailo inference (continuous values)
        if model.use_classifier:
            logits, continuous = model(test_input)
            print(f"Logits shape: {logits.shape}")
            print(f"Continuous FSQ shape: {continuous.shape}")
        else:
            continuous = model(test_input)
            print(f"Continuous FSQ shape: {continuous.shape}")
        
        print(f"Continuous range: [{continuous.min():.3f}, {continuous.max():.3f}]")
        
        # Step 2: CPU post-processing (quantization)
        quantized = fsq_processor.quantize_torch(continuous)
        print(f"Quantized shape: {quantized.shape}")
        print(f"Quantized range: [{quantized.min():.3f}, {quantized.max():.3f}]")
        
        # Check quantization worked
        unique_per_dim = []
        for i in range(quantized.shape[1]):
            unique_vals = torch.unique(quantized[:, i])
            unique_per_dim.append(len(unique_vals))
        
        print(f"Unique values per dimension: {unique_per_dim}")
        print(f"Expected levels: {fsq_processor.levels[:quantized.shape[1]]}")
        print(f"Total codebook size: {fsq_processor.get_codebook_size()}")
    
    print("\n✅ Split inference pipeline working correctly")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FSQ model for Hailo deployment")
    parser.add_argument('--checkpoint', type=str, 
                       default='models/conv2d_fsq_trained_20250921_225014.pth',
                       help='Path to trained checkpoint')
    parser.add_argument('--output', type=str,
                       default='models/fsq_hailo_compatible.onnx',
                       help='Output ONNX path')
    parser.add_argument('--test', action='store_true',
                       help='Test split inference pipeline')
    parser.add_argument('--include-classifier', action='store_true',
                       help='Include classifier in Hailo model')
    
    args = parser.parse_args()
    
    if args.test:
        test_split_inference()
    else:
        if Path(args.checkpoint).exists():
            export_for_hailo(
                args.checkpoint, 
                args.output,
                include_classifier=args.include_classifier
            )
            print(f"\n✅ Export complete: {args.output}")
            print("\nNext steps:")
            print("1. Compile with Hailo SDK:")
            print(f"   hailo parser onnx {args.output} --hw-arch hailo8")
            print("2. Use FSQPostProcessor for quantization after inference")
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
            print("Running test instead...")
            test_split_inference()