"""Bundle validator for deployment packages.

Validates deployment bundles for:
- File presence and integrity
- Configuration compatibility
- Model format consistency
- Version requirements
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .bundler import DeploymentBundle
from .exporter import ModelExporter


class ValidationError(Exception):
    """Validation error exception."""
    pass


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        """Initialize validation result."""
        self.passed: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.errors.append(message)
        self.passed = False
    
    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)
    
    def add_info(self, message: str) -> None:
        """Add informational message."""
        self.info.append(message)
    
    def summary(self) -> str:
        """Generate validation summary."""
        lines = []
        
        if self.passed:
            lines.append("✅ VALIDATION PASSED")
        else:
            lines.append("❌ VALIDATION FAILED")
        
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        lines.append("")
        
        if self.errors:
            lines.append("ERRORS:")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
            lines.append("")
        
        if self.warnings:
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")
        
        if self.info:
            lines.append("INFO:")
            for info in self.info:
                lines.append(f"  ℹ️  {info}")
        
        return "\n".join(lines)


class BundleValidator:
    """Validates deployment bundles."""
    
    def __init__(self):
        """Initialize validator."""
        self.exporter = ModelExporter()
    
    def validate(self, bundle: DeploymentBundle) -> ValidationResult:
        """Validate a deployment bundle.
        
        Args:
            bundle: Bundle to validate
            
        Returns:
            ValidationResult with detailed results
        """
        result = ValidationResult()
        
        # Check bundle exists
        if not bundle.exists():
            result.add_error(f"Bundle directory does not exist: {bundle.bundle_dir}")
            return result
        
        # Validate file presence
        self._validate_file_presence(bundle, result)
        
        # Validate configuration
        config = self._validate_configuration(bundle, result)
        
        # Validate models
        if config:
            self._validate_models(bundle, config, result)
        
        # Validate metadata consistency
        self._validate_metadata(bundle, result)
        
        # Validate version compatibility
        self._validate_version_compatibility(bundle, result)
        
        return result
    
    def _validate_file_presence(
        self,
        bundle: DeploymentBundle,
        result: ValidationResult,
    ) -> None:
        """Validate required files are present."""
        # Required files
        required_files = [
            ("config.yaml", bundle.config_file),
            ("label_map.json", bundle.label_map),
            ("VERSION", bundle.version_file),
            ("COMMIT_SHA", bundle.commit_file),
        ]
        
        for name, path in required_files:
            if not path.exists():
                result.add_error(f"Required file missing: {name}")
            else:
                result.add_info(f"Found required file: {name}")
        
        # Model files (at least one should exist)
        model_files = [
            ("ONNX", bundle.model_onnx),
            ("CoreML", bundle.model_coreml),
            ("Hailo", bundle.model_hailo),
        ]
        
        model_count = 0
        for name, path in model_files:
            if path.exists():
                model_count += 1
                result.add_info(f"Found model: {name}")
        
        if model_count == 0:
            result.add_warning("No model files found")
        
        # Optional files
        if bundle.metrics_file.exists():
            result.add_info("Found metrics file")
        else:
            result.add_warning("Metrics file missing")
    
    def _validate_configuration(
        self,
        bundle: DeploymentBundle,
        result: ValidationResult,
    ) -> Optional[Dict[str, Any]]:
        """Validate configuration file."""
        if not bundle.config_file.exists():
            return None
        
        try:
            with open(bundle.config_file) as f:
                config = yaml.safe_load(f)
            
            # Check required config sections
            required_sections = ["model"]
            for section in required_sections:
                if section not in config:
                    result.add_error(f"Config missing required section: {section}")
                else:
                    result.add_info(f"Config has section: {section}")
            
            # Validate model config
            if "model" in config:
                model_config = config["model"]
                
                # Check model type
                if "name" in model_config:
                    result.add_info(f"Model type: {model_config['name']}")
                else:
                    result.add_warning("Model name not specified")
                
                # Check input/output specifications
                self._validate_model_io_spec(model_config, result)
            
            return config
            
        except yaml.YAMLError as e:
            result.add_error(f"Invalid YAML in config file: {e}")
            return None
        except Exception as e:
            result.add_error(f"Error reading config file: {e}")
            return None
    
    def _validate_model_io_spec(
        self,
        model_config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate model input/output specification."""
        # Expected input shape
        expected_input = (9, 2, 100)  # Channels, sensors, timesteps
        
        if "input_shape" in model_config:
            input_shape = model_config["input_shape"]
            if isinstance(input_shape, list) and len(input_shape) >= 3:
                if tuple(input_shape[-3:]) == expected_input:
                    result.add_info("Input shape matches expected format")
                else:
                    result.add_warning(f"Input shape {input_shape} doesn't match expected {expected_input}")
            else:
                result.add_warning("Input shape format invalid")
        else:
            result.add_warning("Input shape not specified in config")
        
        # Number of classes
        if "n_classes" in model_config:
            n_classes = model_config["n_classes"]
            if isinstance(n_classes, int) and 2 <= n_classes <= 20:
                result.add_info(f"Number of classes: {n_classes}")
            else:
                result.add_warning(f"Unusual number of classes: {n_classes}")
        else:
            result.add_warning("Number of classes not specified")
    
    def _validate_models(
        self,
        bundle: DeploymentBundle,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate model files."""
        # Validate ONNX model
        if bundle.model_onnx.exists():
            self._validate_onnx_model(bundle.model_onnx, config, result)
        
        # Validate CoreML model
        if bundle.model_coreml.exists():
            self._validate_coreml_model(bundle.model_coreml, config, result)
        
        # Validate Hailo model
        if bundle.model_hailo.exists():
            self._validate_hailo_model(bundle.model_hailo, result)
        
        # Cross-validate models if multiple exist
        self._cross_validate_models(bundle, result)
    
    def _validate_onnx_model(
        self,
        onnx_path: Path,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate ONNX model."""
        try:
            model_info = self.exporter.get_model_info(onnx_path)
            
            if "error" in model_info:
                result.add_error(f"ONNX model error: {model_info['error']}")
                return
            
            # Check inputs
            if "inputs" in model_info:
                inputs = model_info["inputs"]
                if len(inputs) == 1:
                    input_info = inputs[0]
                    shape = input_info["shape"]
                    
                    # Check shape (allowing dynamic batch dimension)
                    if len(shape) == 4 and shape[1:] == [9, 2, 100]:
                        result.add_info("ONNX input shape correct")
                    else:
                        result.add_warning(f"ONNX input shape unexpected: {shape}")
                else:
                    result.add_warning(f"ONNX has {len(inputs)} inputs, expected 1")
            
            # Check outputs
            if "outputs" in model_info:
                outputs = model_info["outputs"]
                if len(outputs) == 1:
                    result.add_info("ONNX has single output")
                    
                    # Check output shape matches number of classes
                    output_shape = outputs[0]["shape"]
                    if len(output_shape) >= 2:
                        n_classes = output_shape[-1]
                        config_classes = config.get("model", {}).get("n_classes", 4)
                        
                        if n_classes == config_classes:
                            result.add_info("ONNX output classes match config")
                        else:
                            result.add_error(
                                f"ONNX output classes ({n_classes}) != config ({config_classes})"
                            )
                else:
                    result.add_warning(f"ONNX has {len(outputs)} outputs")
            
            # Check ONNX version
            if "opset_version" in model_info:
                opset = model_info["opset_version"]
                if opset >= 11:
                    result.add_info(f"ONNX opset version {opset} is compatible")
                else:
                    result.add_warning(f"ONNX opset version {opset} may be outdated")
            
        except Exception as e:
            result.add_error(f"ONNX validation failed: {e}")
    
    def _validate_coreml_model(
        self,
        coreml_path: Path,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate CoreML model."""
        try:
            model_info = self.exporter.get_model_info(coreml_path)
            
            if "error" in model_info:
                result.add_warning(f"CoreML validation limited: {model_info['error']}")
                return
            
            # Check basic info
            if "inputs" in model_info and "outputs" in model_info:
                n_inputs = len(model_info["inputs"])
                n_outputs = len(model_info["outputs"])
                
                result.add_info(f"CoreML has {n_inputs} inputs, {n_outputs} outputs")
                
                if n_inputs == 1 and n_outputs == 1:
                    result.add_info("CoreML I/O structure looks correct")
                else:
                    result.add_warning("CoreML I/O structure may be complex")
            
            # Check metadata
            if "description" in model_info:
                result.add_info(f"CoreML description: {model_info['description']}")
            
        except Exception as e:
            result.add_error(f"CoreML validation failed: {e}")
    
    def _validate_hailo_model(
        self,
        hailo_path: Path,
        result: ValidationResult,
    ) -> None:
        """Validate Hailo model."""
        # Basic file validation
        if hailo_path.stat().st_size < 1024:
            result.add_warning("Hailo model file seems very small")
        else:
            result.add_info("Hailo model file size looks reasonable")
        
        # Check file extension
        if hailo_path.suffix == ".hef":
            result.add_info("Hailo model has correct extension (.hef)")
        else:
            result.add_warning(f"Hailo model extension unexpected: {hailo_path.suffix}")
        
        # Could add more detailed HEF validation if Hailo tools are available
        try:
            # Try to get HEF info (requires Hailo runtime)
            import subprocess
            result_check = subprocess.run(
                ["hailort", "run", "--print-hef-info", str(hailo_path)],
                capture_output=True,
                timeout=10,
            )
            if result_check.returncode == 0:
                result.add_info("Hailo model validated by HailoRT")
            else:
                result.add_warning("Hailo model validation inconclusive")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result.add_info("Hailo validation tools not available")
    
    def _cross_validate_models(
        self,
        bundle: DeploymentBundle,
        result: ValidationResult,
    ) -> None:
        """Cross-validate multiple model formats."""
        available_models = []
        
        if bundle.model_onnx.exists():
            available_models.append("ONNX")
        if bundle.model_coreml.exists():
            available_models.append("CoreML")
        if bundle.model_hailo.exists():
            available_models.append("Hailo")
        
        if len(available_models) > 1:
            result.add_info(f"Multiple model formats available: {', '.join(available_models)}")
            
            # Could add more sophisticated cross-validation here
            # e.g., comparing model outputs on same input
        
        elif len(available_models) == 1:
            result.add_info(f"Single model format: {available_models[0]}")
        
        else:
            result.add_warning("No model files found")
    
    def _validate_metadata(
        self,
        bundle: DeploymentBundle,
        result: ValidationResult,
    ) -> None:
        """Validate metadata consistency."""
        # Validate label mapping
        if bundle.label_map.exists():
            try:
                with open(bundle.label_map) as f:
                    label_data = json.load(f)
                
                if "mapping" in label_data:
                    mapping = label_data["mapping"]
                    n_labels = len(mapping)
                    result.add_info(f"Label mapping has {n_labels} behaviors")
                    
                    # Check label IDs are consecutive
                    ids = sorted(int(k) for k in mapping.keys())
                    expected_ids = list(range(len(ids)))
                    
                    if ids == expected_ids:
                        result.add_info("Label IDs are consecutive")
                    else:
                        result.add_warning("Label IDs are not consecutive")
                
                else:
                    result.add_error("Label mapping file missing 'mapping' key")
                    
            except json.JSONDecodeError as e:
                result.add_error(f"Invalid JSON in label mapping: {e}")
            except Exception as e:
                result.add_error(f"Error reading label mapping: {e}")
        
        # Validate metrics
        if bundle.metrics_file.exists():
            try:
                with open(bundle.metrics_file) as f:
                    metrics_data = json.load(f)
                
                if "metrics" in metrics_data:
                    metrics = metrics_data["metrics"]
                    
                    # Check for key metrics
                    key_metrics = ["accuracy", "macro_f1", "ece"]
                    found_metrics = [m for m in key_metrics if m in metrics]
                    
                    if found_metrics:
                        result.add_info(f"Found key metrics: {', '.join(found_metrics)}")
                    else:
                        result.add_warning("No key metrics found")
                    
                    # Check metric ranges
                    if "accuracy" in metrics:
                        acc = metrics["accuracy"]
                        if 0 <= acc <= 1:
                            result.add_info(f"Accuracy: {acc:.1%}")
                        else:
                            result.add_warning(f"Accuracy out of range: {acc}")
                
                else:
                    result.add_warning("Metrics file missing 'metrics' key")
                    
            except json.JSONDecodeError as e:
                result.add_error(f"Invalid JSON in metrics file: {e}")
            except Exception as e:
                result.add_error(f"Error reading metrics: {e}")
    
    def _validate_version_compatibility(
        self,
        bundle: DeploymentBundle,
        result: ValidationResult,
    ) -> None:
        """Validate version compatibility."""
        if not bundle.version_file.exists():
            return
        
        try:
            with open(bundle.version_file) as f:
                version_data = json.load(f)
            
            # Check framework version
            if "framework" in version_data:
                framework = version_data["framework"]
                if framework == "Conv2d":
                    result.add_info("Framework: Conv2d")
                else:
                    result.add_warning(f"Unknown framework: {framework}")
            
            # Check Python version
            if "python" in version_data:
                python_version = version_data["python"]
                result.add_info(f"Python version: {python_version}")
                
                # Check if it's a reasonable version
                major, minor = python_version.split(".")[:2]
                if int(major) >= 3 and int(minor) >= 8:
                    result.add_info("Python version compatible")
                else:
                    result.add_warning(f"Python version may be outdated: {python_version}")
            
            # Check dependencies
            if "dependencies" in version_data:
                deps = version_data["dependencies"]
                
                # Check key dependencies
                for dep_name, min_version in [
                    ("torch", "1.8.0"),
                    ("numpy", "1.19.0"),
                ]:
                    if dep_name in deps:
                        result.add_info(f"{dep_name}: {deps[dep_name]}")
                    else:
                        result.add_warning(f"Missing dependency info: {dep_name}")
            
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON in version file: {e}")
        except Exception as e:
            result.add_error(f"Error reading version file: {e}")
    
    def validate_for_target(
        self,
        bundle: DeploymentBundle,
        target: str,
    ) -> ValidationResult:
        """Validate bundle for specific deployment target.
        
        Args:
            bundle: Bundle to validate
            target: Target platform (ios, edge, hailo)
            
        Returns:
            ValidationResult with target-specific validation
        """
        result = self.validate(bundle)  # Base validation
        
        # Add target-specific checks
        if target.lower() == "ios":
            self._validate_for_ios(bundle, result)
        elif target.lower() in ["edge", "hailo"]:
            self._validate_for_hailo(bundle, result)
        elif target.lower() == "onnx":
            self._validate_for_onnx(bundle, result)
        else:
            result.add_warning(f"Unknown target: {target}")
        
        return result
    
    def _validate_for_ios(
        self,
        bundle: DeploymentBundle,
        result: ValidationResult,
    ) -> None:
        """Add iOS-specific validation."""
        if not bundle.model_coreml.exists():
            result.add_error("iOS deployment requires CoreML model")
        else:
            result.add_info("CoreML model available for iOS")
        
        # Check iOS-specific config
        try:
            with open(bundle.config_file) as f:
                config = yaml.safe_load(f)
            
            # Check for iOS-specific settings
            ios_config = config.get("ios", {})
            if "minimum_version" in ios_config:
                result.add_info(f"iOS minimum version: {ios_config['minimum_version']}")
            else:
                result.add_info("iOS minimum version not specified (using default)")
        
        except Exception:
            pass  # Config already validated
    
    def _validate_for_hailo(
        self,
        bundle: DeploymentBundle,
        result: ValidationResult,
    ) -> None:
        """Add Hailo-specific validation."""
        if not bundle.model_hailo.exists():
            result.add_error("Hailo deployment requires HEF model")
        else:
            result.add_info("HEF model available for Hailo")
        
        # Check Hailo-specific requirements
        if bundle.model_onnx.exists() and bundle.model_hailo.exists():
            # Compare file sizes - HEF should be smaller than ONNX
            onnx_size = bundle.model_onnx.stat().st_size
            hailo_size = bundle.model_hailo.stat().st_size
            
            if hailo_size < onnx_size:
                result.add_info("HEF model is optimized (smaller than ONNX)")
            else:
                result.add_warning("HEF model larger than ONNX (check optimization)")
    
    def _validate_for_onnx(
        self,
        bundle: DeploymentBundle,
        result: ValidationResult,
    ) -> None:
        """Add ONNX-specific validation."""
        if not bundle.model_onnx.exists():
            result.add_error("ONNX deployment requires ONNX model")
        else:
            result.add_info("ONNX model available")