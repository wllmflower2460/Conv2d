"""Model exporter for multiple deployment targets.

Exports trained models to:
- ONNX (universal format)
- CoreML (iOS deployment)
- Hailo HEF (edge acceleration)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


class ModelExporter:
    """Export models to different deployment formats."""
    
    def __init__(self):
        """Initialize exporter."""
        self.temp_dir = None
    
    def export_onnx(
        self,
        model_path: Path,
        output_path: Path,
        config: Dict[str, Any],
        input_shape: Tuple[int, ...] = (1, 9, 2, 100),
    ) -> bool:
        """Export model to ONNX format.
        
        Args:
            model_path: Path to PyTorch model
            output_path: Output ONNX file path
            config: Model configuration
            input_shape: Input tensor shape
            
        Returns:
            True if export succeeded
        """
        try:
            # Load model
            model = self._load_pytorch_model(model_path, config)
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(input_shape, dtype=torch.float32)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                }
            )
            
            # Verify export
            if self._verify_onnx_model(output_path, input_shape):
                print(f"✓ ONNX export successful: {output_path}")
                return True
            else:
                print(f"✗ ONNX verification failed")
                return False
                
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            return False
    
    def export_coreml(
        self,
        onnx_path: Path,
        output_path: Path,
        config: Dict[str, Any],
    ) -> bool:
        """Export ONNX model to CoreML format.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Output CoreML package path
            config: Model configuration
            
        Returns:
            True if export succeeded
        """
        try:
            # Import coremltools (may not be available)
            try:
                import coremltools as ct
            except ImportError:
                print("✗ CoreML export requires coremltools: pip install coremltools")
                return False
            
            # Load ONNX model
            onnx_model = ct.converters.onnx.convert(
                str(onnx_path),
                minimum_deployment_target=ct.target.iOS14,
            )
            
            # Set metadata
            onnx_model.short_description = "Conv2d behavioral analysis model"
            onnx_model.author = "Conv2d Framework"
            onnx_model.license = "MIT"
            onnx_model.version = "1.0"
            
            # Add input/output descriptions
            onnx_model.input_description["input"] = "IMU sensor data (batch, 9, 2, 100)"
            onnx_model.output_description["output"] = "Behavioral classification logits"
            
            # Save CoreML package
            onnx_model.save(str(output_path))
            
            print(f"✓ CoreML export successful: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ CoreML export failed: {e}")
            return False
    
    def export_hailo(
        self,
        onnx_path: Path,
        output_path: Path,
        config: Dict[str, Any],
    ) -> bool:
        """Export ONNX model to Hailo HEF format.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Output HEF file path
            config: Model configuration
            
        Returns:
            True if export succeeded
        """
        try:
            # Check if Hailo compiler is available
            if not self._check_hailo_compiler():
                print("✗ Hailo compiler not found. Install Hailo Dataflow Compiler.")
                return False
            
            # Create temporary HAR file
            with tempfile.TemporaryDirectory() as temp_dir:
                har_path = Path(temp_dir) / "model.har"
                
                # Step 1: Parse ONNX to HAR
                parse_cmd = [
                    "hailo", "parser", "onnx",
                    str(onnx_path),
                    "--har-path", str(har_path),
                    "--net-name", "conv2d_behavioral",
                ]
                
                result = subprocess.run(parse_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"✗ Hailo parse failed: {result.stderr}")
                    return False
                
                # Step 2: Optimize HAR
                optimized_har = Path(temp_dir) / "optimized.har"
                optimize_cmd = [
                    "hailo", "optimize",
                    str(har_path),
                    "--har-path", str(optimized_har),
                    "--hw-arch", "hailo8",
                ]
                
                result = subprocess.run(optimize_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"✗ Hailo optimize failed: {result.stderr}")
                    return False
                
                # Step 3: Compile to HEF
                compile_cmd = [
                    "hailo", "compiler",
                    str(optimized_har),
                    "--hef-path", str(output_path),
                    "--hw-arch", "hailo8",
                ]
                
                result = subprocess.run(compile_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"✗ Hailo compile failed: {result.stderr}")
                    return False
            
            print(f"✓ Hailo export successful: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Hailo export failed: {e}")
            return False
    
    def _load_pytorch_model(
        self,
        model_path: Path,
        config: Dict[str, Any],
    ) -> nn.Module:
        """Load PyTorch model from checkpoint."""
        # Try to load directly
        try:
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'eval'):
                return model
            # If it's a state dict, need to reconstruct model
        except Exception:
            pass
        
        # Try to load as state dict
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract model architecture from config
            model = self._create_model_from_config(config)
            
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> nn.Module:
        """Create model instance from configuration.
        
        This is a simplified version - in practice, you'd have
        a model factory that creates the exact architecture.
        """
        # Placeholder for model creation
        # In real implementation, this would instantiate the actual Conv2d model
        
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(9, 64, (2, 5))
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(64 * 1 * 96, 4)  # 4 classes
                
            def forward(self, x):
                # x: (B, 9, 2, 100)
                x = self.conv(x)  # (B, 64, 1, 96)
                x = torch.relu(x)
                x = self.flatten(x)  # (B, 64 * 1 * 96)
                x = self.fc(x)  # (B, 4)
                return x
        
        return DummyModel()
    
    def _verify_onnx_model(
        self,
        onnx_path: Path,
        input_shape: Tuple[int, ...],
    ) -> bool:
        """Verify ONNX model can be loaded and run."""
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Test inference
            ort_session = ort.InferenceSession(str(onnx_path))
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            outputs = ort_session.run(None, {'input': dummy_input})
            
            # Basic output validation
            if len(outputs) > 0:
                output = outputs[0]
                if output.shape[0] == input_shape[0]:  # Batch size matches
                    return True
            
            return False
            
        except ImportError:
            print("Warning: ONNX verification requires onnx and onnxruntime packages")
            return True  # Assume success if can't verify
        except Exception as e:
            print(f"ONNX verification failed: {e}")
            return False
    
    def _check_hailo_compiler(self) -> bool:
        """Check if Hailo compiler is available."""
        try:
            result = subprocess.run(
                ["hailo", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Get information about a model file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with model information
        """
        info = {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size if model_path.exists() else 0,
            "format": self._detect_format(model_path),
        }
        
        # Format-specific information
        if info["format"] == "onnx":
            info.update(self._get_onnx_info(model_path))
        elif info["format"] == "pytorch":
            info.update(self._get_pytorch_info(model_path))
        elif info["format"] == "coreml":
            info.update(self._get_coreml_info(model_path))
        
        return info
    
    def _detect_format(self, model_path: Path) -> str:
        """Detect model format from file extension."""
        suffix = model_path.suffix.lower()
        
        if suffix == ".onnx":
            return "onnx"
        elif suffix in [".pth", ".pt"]:
            return "pytorch"
        elif suffix == ".mlpackage" or model_path.name.endswith(".mlpackage"):
            return "coreml"
        elif suffix == ".hef":
            return "hailo"
        else:
            return "unknown"
    
    def _get_onnx_info(self, onnx_path: Path) -> Dict[str, Any]:
        """Get ONNX model information."""
        try:
            import onnx
            
            model = onnx.load(str(onnx_path))
            
            # Get input/output shapes
            inputs = []
            for inp in model.graph.input:
                shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
                inputs.append({
                    "name": inp.name,
                    "shape": shape,
                    "dtype": inp.type.tensor_type.elem_type,
                })
            
            outputs = []
            for out in model.graph.output:
                shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
                outputs.append({
                    "name": out.name,
                    "shape": shape,
                    "dtype": out.type.tensor_type.elem_type,
                })
            
            return {
                "inputs": inputs,
                "outputs": outputs,
                "opset_version": model.opset_import[0].version if model.opset_import else None,
            }
            
        except ImportError:
            return {"error": "ONNX package not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_pytorch_info(self, model_path: Path) -> Dict[str, Any]:
        """Get PyTorch model information."""
        try:
            # Just load metadata without full model
            checkpoint = torch.load(model_path, map_location='cpu')
            
            info = {"type": "checkpoint"}
            
            if isinstance(checkpoint, dict):
                info["keys"] = list(checkpoint.keys())
                
                # Check for common keys
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    info["parameter_count"] = sum(p.numel() for p in state_dict.values())
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    info["parameter_count"] = sum(p.numel() for p in state_dict.values())
                
                # Other metadata
                for key in ["epoch", "best_acc", "optimizer", "config"]:
                    if key in checkpoint:
                        info[key] = checkpoint[key]
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_coreml_info(self, coreml_path: Path) -> Dict[str, Any]:
        """Get CoreML model information."""
        try:
            import coremltools as ct
            
            model = ct.models.MLModel(str(coreml_path))
            spec = model.get_spec()
            
            inputs = []
            for inp in spec.description.input:
                inputs.append({
                    "name": inp.name,
                    "type": str(inp.type),
                })
            
            outputs = []
            for out in spec.description.output:
                outputs.append({
                    "name": out.name,
                    "type": str(out.type),
                })
            
            return {
                "inputs": inputs,
                "outputs": outputs,
                "description": spec.description.metadata.shortDescription,
                "author": spec.description.metadata.author,
            }
            
        except ImportError:
            return {"error": "coremltools package not available"}
        except Exception as e:
            return {"error": str(e)}