"""
Export Pipeline for Hailo-8 Deployment
Converts trained TCN-VAE model to Hailo-compatible format
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxsim import simplify
import numpy as np
from pathlib import Path
import yaml
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tcn_vae_hailo import HailoTCNVAE

# Configure logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HailoExporter:
    """
    Export trained models to Hailo-compatible format
    """
    
    def __init__(self, model_checkpoint: str, config_path: str):
        """
        Initialize exporter
        
        Args:
            model_checkpoint: Path to trained model checkpoint
            config_path: Path to configuration YAML
        """
        self.checkpoint_path = Path(model_checkpoint)
        self.config_path = Path(config_path)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        # Create model
        self.model = HailoTCNVAE(
            input_dim=9,
            hidden_dims=[64, 128, 256],
            latent_dim=64,
            sequence_length=100,
            num_human_activities=12,
            num_dog_behaviors=3,
            use_device_attention=True
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"✅ Model loaded from epoch {checkpoint['epoch']}")
        logger.info(f"   Metrics: {checkpoint['metrics']}")
    
    def export_to_onnx(self, output_path: str, simplify_model: bool = True):
        """
        Export model to ONNX format
        
        Args:
            output_path: Path for ONNX model
            simplify_model: Whether to simplify ONNX model
        """
        logger.info("Exporting to ONNX...")
        
        # Get input shape from config
        input_shape = self.config['hailo_deployment']['io_specification']['input_shape']
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        # Export encoder only for edge inference
        class EncoderWrapper(nn.Module):
            def __init__(self, full_model):
                super().__init__()
                self.input_projection = full_model.input_projection
                self.tcn_encoder = full_model.tcn_encoder
                self.fc_mu = full_model.fc_mu
                self.dog_classifier = full_model.dog_classifier
            
            def forward(self, x):
                # Encode input
                x = self.input_projection(x)
                h, attention = self.tcn_encoder(x)
                
                # Handle device attention output
                if len(h.shape) == 4:
                    h = h.mean(dim=2)
                
                # Global pooling
                h = torch.nn.functional.adaptive_avg_pool1d(h, 1).squeeze(-1)
                
                # Get latent and dog predictions
                mu = self.fc_mu(h)
                dog_logits = self.dog_classifier(mu)
                
                # Fallback attention tensor shape matches expected (B, num_devices, num_devices)
                default_attention = torch.zeros(x.shape[0], 2, 2, device=x.device)
                return dog_logits, attention if attention is not None else default_attention
        
        encoder_model = EncoderWrapper(self.model)
        encoder_model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            encoder_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['dog_behavior', 'attention_weights'],
            dynamic_axes=None,  # Static shapes for Hailo
            verbose=False
        )
        
        logger.info(f"✅ ONNX export complete: {output_path}")
        
        # Simplify if requested
        if simplify_model:
            self._simplify_onnx(output_path)
    
    def _simplify_onnx(self, onnx_path: str):
        """Simplify ONNX model"""
        logger.info("Simplifying ONNX model...")
        
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        
        if check:
            output_path = onnx_path.replace('.onnx', '_simplified.onnx')
            onnx.save(model_simp, output_path)
            logger.info(f"✅ Simplified model saved: {output_path}")
        else:
            logger.warning("⚠️ Simplification failed, using original model")
    
    def validate_onnx(self, onnx_path: str) -> bool:
        """
        Validate ONNX model for Hailo compatibility
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            True if valid, False otherwise
        """
        logger.info("Validating ONNX model...")
        
        # Load ONNX model
        model = onnx.load(onnx_path)
        
        # Check for unsupported operations
        unsupported_ops = self.config['hailo_deployment']['architecture_constraints']['unsupported_ops']
        found_unsupported = []
        
        for node in model.graph.node:
            if node.op_type in unsupported_ops:
                found_unsupported.append(node.op_type)
        
        if found_unsupported:
            logger.error(f"❌ Found unsupported operations: {found_unsupported}")
            return False
        
        # Verify Conv2D operations
        conv_count = sum(1 for node in model.graph.node if node.op_type == 'Conv')
        logger.info(f"   Found {conv_count} Conv operations")
        
        # Check input shape
        input_shape = []
        for input_tensor in model.graph.input:
            for dim in input_tensor.type.tensor_type.shape.dim:
                input_shape.append(dim.dim_value)
        
        expected_shape = self.config['hailo_deployment']['io_specification']['input_shape']
        if input_shape != expected_shape:
            logger.error(f"❌ Input shape mismatch: {input_shape} != {expected_shape}")
            return False
        
        logger.info(f"✅ ONNX model is Hailo-compatible")
        return True
    
    def test_onnx_inference(self, onnx_path: str):
        """Test ONNX model inference"""
        logger.info("Testing ONNX inference...")
        
        # Create ONNX runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Get input shape
        input_shape = self.config['hailo_deployment']['io_specification']['input_shape']
        
        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {'input': test_input})
        
        logger.info(f"   Output shapes: {[out.shape for out in outputs]}")
        
        # Verify output shape
        expected_output = self.config['hailo_deployment']['io_specification']['output_shape']
        if outputs[0].shape != tuple(expected_output):
            logger.warning(f"⚠️ Output shape mismatch: {outputs[0].shape} != {expected_output}")
        
        # Test latency
        import time
        num_tests = 100
        
        start = time.time()
        for _ in range(num_tests):
            _ = session.run(None, {'input': test_input})
        elapsed = (time.time() - start) / num_tests * 1000
        
        target_latency = self.config['hailo_deployment']['performance_targets']['inference_latency_ms']
        
        logger.info(f"   Average latency: {elapsed:.2f}ms (target: <{target_latency}ms)")
        
        if elapsed < target_latency:
            logger.info(f"✅ Latency target met!")
        else:
            logger.warning(f"⚠️ Latency exceeds target")
    
    def generate_hailo_script(self, onnx_path: str, output_dir: str):
        """
        Generate Hailo compilation script
        
        Args:
            onnx_path: Path to ONNX model
            output_dir: Directory for Hailo files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate calibration data
        calib_path = output_dir / "calibration_data.npy"
        self._generate_calibration_data(calib_path)
        
        # Generate Hailo model script
        script_content = f"""#!/usr/bin/env python3
\"\"\"
Hailo Model Script for Cross-Species TCN-VAE
Generated for model: {onnx_path}
\"\"\"

import numpy as np
from hailo_sdk_client import ClientRunner

def optimize_model(runner: ClientRunner):
    \"\"\"Apply Hailo-specific optimizations\"\"\"
    
    # Set quantization parameters
    runner.set_quantization_params(
        calibration_dataset_size={self.config['hailo_deployment']['quantization']['calibration_dataset_size']},
        quantization_method='{self.config['hailo_deployment']['quantization']['method']}',
        percentile={self.config['hailo_deployment']['quantization']['percentile']},
        per_channel_quantization={self.config['hailo_deployment']['quantization']['per_channel_quantization']}
    )
    
    # Set optimization level (0=size, 1=balanced, 2=latency)
    runner.set_optimization_level({self.config['hailo_deployment']['compilation']['optimization_level']})
    
    # Set batch size
    runner.set_batch_size({self.config['hailo_deployment']['compilation']['batch_size']})
    
    # Device-specific optimizations
    runner.optimize_for_device('{self.config['hailo_deployment']['target_hardware']}')
    
    # Enable profiling
    runner.enable_profiling({self.config['hailo_deployment']['compilation']['enable_profiling']})
    
    # Memory optimization
    runner.enable_memory_compression()
    
    print("✅ Hailo optimizations applied")
    
    return runner

if __name__ == "__main__":
    # This script is called by the Hailo compiler
    pass
"""
        
        script_path = output_dir / "hailo_model_script.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Generate compilation command
        compile_cmd = f"""#!/bin/bash
# Hailo Compilation Script for Cross-Species TCN-VAE

# Set paths
ONNX_MODEL="{onnx_path}"
CALIB_DATA="{calib_path}"
MODEL_SCRIPT="{script_path}"
OUTPUT_HEF="{output_dir}/tcn_vae_cross_species.hef"

# Run Hailo compiler
hailo compile \\
    $ONNX_MODEL \\
    --hw-arch hailo8 \\
    --calib-set-path $CALIB_DATA \\
    --model-script $MODEL_SCRIPT \\
    --output-path $OUTPUT_HEF \\
    --performance-param optimization-level=2 \\
    --performance-param batch-size=1

echo "✅ Compilation complete: $OUTPUT_HEF"
"""
        
        compile_script_path = output_dir / "compile_hailo.sh"
        with open(compile_script_path, 'w') as f:
            f.write(compile_cmd)
        
        # Make script executable
        os.chmod(compile_script_path, 0o755)
        
        logger.info(f"✅ Hailo scripts generated in {output_dir}")
        logger.info(f"   Run: {compile_script_path}")
    
    def _generate_calibration_data(self, output_path: str):
        """Generate calibration dataset for INT8 quantization"""
        logger.info("Generating calibration data...")
        
        input_shape = self.config['hailo_deployment']['io_specification']['input_shape']
        calib_size = self.config['hailo_deployment']['quantization']['calibration_dataset_size']
        
        # Generate representative data
        # In production, use real data samples
        calibration_data = np.random.randn(calib_size, *input_shape[1:]).astype(np.float32)
        
        # Normalize to match training data distribution
        calibration_data = (calibration_data - calibration_data.mean()) / calibration_data.std()
        
        np.save(output_path, calibration_data)
        logger.info(f"✅ Calibration data saved: {output_path}")
    
    def export_complete_pipeline(self, output_dir: str):
        """
        Complete export pipeline for Hailo deployment
        
        Args:
            output_dir: Directory for all export artifacts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Starting complete Hailo export pipeline...")
        logger.info("=" * 60)
        
        # Step 1: Export to ONNX
        onnx_path = output_dir / "tcn_vae_cross_species.onnx"
        self.export_to_onnx(str(onnx_path), simplify_model=True)
        
        # Step 2: Validate ONNX
        if not self.validate_onnx(str(onnx_path)):
            raise ValueError("ONNX validation failed!")
        
        # Step 3: Test ONNX inference
        self.test_onnx_inference(str(onnx_path))
        
        # Step 4: Generate Hailo scripts
        self.generate_hailo_script(str(onnx_path), str(output_dir / "hailo"))
        
        logger.info("=" * 60)
        logger.info("✅ Export pipeline complete!")
        logger.info(f"   Artifacts saved to: {output_dir}")
        logger.info("=" * 60)
        
        # Print next steps
        print("\n📋 Next Steps:")
        print("1. Copy files to Hailo device")
        print("2. Run compilation script: hailo/compile_hailo.sh")
        print("3. Deploy .hef file to edge device")
        print("4. Integrate with EdgeInfer API")


def main():
    """Main export script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export model for Hailo deployment")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/enhanced_dataset_schema.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='./hailo_export',
                       help='Output directory for export artifacts')
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = HailoExporter(args.checkpoint, args.config)
    
    # Run complete export pipeline
    exporter.export_complete_pipeline(args.output)


if __name__ == "__main__":
    main()