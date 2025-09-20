#!/usr/bin/env python3
"""
macOS-specific CoreML conversion script.
Run this on a Mac to convert ONNX models to CoreML format.
"""

import coremltools as ct
import numpy as np
from pathlib import Path
import json
import sys


def convert_onnx_to_coreml(
    onnx_path,
    output_path,
    model_name="DogBehaviorClassifier",
    ios_version="iOS15"
):
    """
    Convert ONNX model to CoreML format with proper configuration.
    
    Args:
        onnx_path: Path to ONNX model file
        output_path: Output path for CoreML model
        model_name: Name for the model
        ios_version: Target iOS version (iOS14, iOS15, iOS16)
    """
    
    print(f"Converting {onnx_path} to CoreML...")
    print("=" * 60)
    
    # Set deployment target
    deployment_targets = {
        "iOS14": ct.target.iOS14,
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
    }
    target = deployment_targets.get(ios_version, ct.target.iOS15)
    
    # Load metadata if available
    metadata_path = Path(onnx_path).parent / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}")
    else:
        metadata = None
    
    # Define class labels
    class_labels = [
        'sit', 'down', 'stand', 'stay', 'lying',
        'heel', 'come', 'fetch', 'drop', 'wait',
        'leave_it', 'walking', 'trotting', 'running',
        'jumping', 'spinning', 'rolling', 'playing',
        'alert', 'sniffing', 'looking'
    ]
    
    try:
        # Convert the model
        print(f"Target iOS version: {ios_version}")
        print("Converting model...")
        
        # Basic conversion
        ml_model = ct.convert(
            onnx_path,
            convert_to='mlprogram',  # Use Neural Network for older iOS
            minimum_deployment_target=target,
        )
        
        # Add metadata
        ml_model.author = 'TCN-VAE Dog Behavior Model'
        ml_model.license = 'MIT'
        ml_model.short_description = '24-point dog pose to behavior classification'
        ml_model.version = '1.0'
        
        # Add input/output descriptions
        ml_model.input_description['pose_sequence'] = (
            'Sequence of 24 dog keypoints (x,y) over 100 frames. '
            'Shape: [1, 48, 100] where 48 = 24 keypoints × 2 coordinates'
        )
        
        ml_model.output_description['behavior_probs'] = (
            'Probability distribution over 21 dog behavior classes'
        )
        
        # Add user-defined metadata
        if metadata:
            ml_model.user_defined_metadata['training_accuracy'] = str(metadata['model_info'].get('accuracy', 'N/A'))
            ml_model.user_defined_metadata['keypoint_order'] = ', '.join(metadata['input']['keypoints'][:5]) + '...'
            ml_model.user_defined_metadata['total_keypoints'] = '24'
            ml_model.user_defined_metadata['temporal_window'] = '100 frames'
        
        # Save the model
        output_path = Path(output_path)
        if output_path.suffix not in ['.mlmodel', '.mlpackage']:
            output_path = output_path.with_suffix('.mlpackage')
        
        ml_model.save(str(output_path))
        print(f"✅ Model saved to: {output_path}")
        
        # Print model summary
        print("\nModel Summary:")
        print("-" * 40)
        print(f"  Type: ML Program (Neural Network)")
        print(f"  Input shape: [1, 48, 100]")
        print(f"  Output shape: [1, 21]")
        print(f"  Classes: {len(class_labels)}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"  iOS Target: {ios_version}+")
        
        return ml_model
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None


def test_coreml_model(model_path):
    """
    Test the converted CoreML model with sample data.
    """
    print("\nTesting CoreML model...")
    print("-" * 40)
    
    try:
        # Load the model
        model = ct.models.MLModel(model_path)
        
        # Create test input (random pose sequence)
        test_input = np.random.randn(1, 48, 100).astype(np.float32)
        
        # Make prediction
        prediction = model.predict({'pose_sequence': test_input})
        
        # Display results
        if 'behavior_probs' in prediction:
            probs = prediction['behavior_probs']
            print("✅ Model inference successful!")
            print(f"  Output shape: {probs.shape}")
            print(f"  Sum of probabilities: {np.sum(probs):.4f}")
            
            # Show top predictions if available
            if hasattr(probs, 'items'):
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                print("\n  Top 3 predictions:")
                for label, prob in sorted_probs:
                    print(f"    {label}: {prob:.2%}")
        else:
            print("  Output:", list(prediction.keys()))
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


def batch_convert(input_dir, output_dir, ios_version="iOS15"):
    """
    Convert all ONNX models in a directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    onnx_files = list(input_path.glob("*.onnx"))
    
    if not onnx_files:
        print(f"No ONNX files found in {input_dir}")
        return
    
    print(f"Found {len(onnx_files)} ONNX file(s) to convert")
    
    for onnx_file in onnx_files:
        output_name = onnx_file.stem + ".mlpackage"
        output_file = output_path / output_name
        
        print(f"\nConverting {onnx_file.name}...")
        model = convert_onnx_to_coreml(
            onnx_file,
            output_file,
            model_name=onnx_file.stem,
            ios_version=ios_version
        )
        
        if model and output_file.exists():
            test_coreml_model(output_file)


def main():
    """
    Main conversion pipeline for macOS.
    """
    print("CoreML Conversion Pipeline for macOS")
    print("=" * 60)
    
    # Check if running on macOS
    if sys.platform != 'darwin':
        print("⚠️  Warning: This script is designed for macOS.")
        print("   CoreML conversion may not work properly on other platforms.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Default paths
    models_dir = Path(__file__).parent.parent / "models"
    onnx_path = models_dir / "dog_behavior_classifier.onnx"
    output_path = models_dir / "DogBehaviorClassifier.mlpackage"
    
    # Check if ONNX file exists
    if not onnx_path.exists():
        print(f"❌ ONNX model not found: {onnx_path}")
        print("\nPlease run 'export_for_coreml.py' first to generate the ONNX model.")
        return
    
    # Convert the model
    model = convert_onnx_to_coreml(
        onnx_path,
        output_path,
        model_name="DogBehaviorClassifier",
        ios_version="iOS15"  # Change to iOS14 for broader compatibility
    )
    
    # Test if successful
    if model and output_path.exists():
        test_coreml_model(output_path)
        
        print("\n" + "=" * 60)
        print("✅ Conversion Complete!")
        print(f"\nModel ready for iOS: {output_path}")
        print("\nNext steps:")
        print("1. Open your Xcode project")
        print("2. Drag and drop the .mlpackage file into your project")
        print("3. Ensure 'Copy items if needed' is checked")
        print("4. Select your app target")
        print("5. Use the auto-generated Swift class to make predictions")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")


if __name__ == "__main__":
    main()