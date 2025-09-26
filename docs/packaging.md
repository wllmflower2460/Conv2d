# Artifact Packaging Documentation

The artifact packaging system provides complete deployment bundles with multi-format model export (ONNX, CoreML, Hailo HEF), comprehensive metadata, and target-specific validation for production edge deployment.

## Overview

Key features:
- **Single bundle per experiment**: `artifacts/EXP_HASH/` with complete deployment package
- **Multi-format export**: ONNX, CoreML, Hailo HEF in one bundle
- **Complete metadata**: Config, labels, metrics, versions, commit SHA
- **Target validation**: Platform-specific deployment checks (iOS, Hailo, ONNX)
- **CLI integration**: `conv2d pack` commands for bundle management

## Architecture

```
Model + Config → Artifact Bundler → Complete Bundle
      ↓               ↓                    ↓
Model Exporter → Multi-format → artifacts/EXP_HASH/
      ↓               ↓                    ↓
Bundle Validator → Target Checks → Deployment Ready
```

## Core Components

### Artifact Bundler

```python
from conv2d.packaging import ArtifactBundler

bundler = ArtifactBundler(output_base="artifacts")

bundle = bundler.create_bundle(
    config=config,
    model_path="models/best.pth",
    metrics=metrics,
    exp_name="dogs_v1",
)

print(f"Bundle created: {bundle.exp_hash}")
print(f"Bundle path: {bundle.bundle_dir}")
```

### Model Exporter

```python
from conv2d.packaging import ModelExporter

exporter = ModelExporter()

# Export to multiple formats
formats = exporter.export_all(
    model=model,
    input_shape=(1, 9, 2, 100),
    output_dir=bundle.bundle_dir,
    model_name="conv2d_fsq",
)

print(f"Exported formats: {list(formats.keys())}")
```

### Bundle Validator

```python
from conv2d.packaging import BundleValidator

validator = BundleValidator()

# Validate for specific deployment target
result = validator.validate_for_target(bundle, target="hailo")

if result.passed:
    print("✓ Bundle ready for Hailo deployment")
else:
    print(f"✗ Validation failed: {result.errors}")
```

## Bundle Structure

### Complete Bundle Layout

```
artifacts/EXP_HASH/
├── models/                           # Model files
│   ├── model.onnx                   # ONNX format (universal)
│   ├── coreml.mlpackage/            # CoreML package (iOS)
│   └── hailo.hef                    # Hailo HEF (edge)
├── config.yaml                      # Complete experiment configuration
├── label_map.json                   # Class ID to label mapping
├── metrics.json                     # Evaluation metrics
├── metadata.json                    # Bundle metadata
├── VERSION                          # System version
├── COMMIT_SHA                       # Git commit hash
├── MANIFEST.json                    # Bundle contents manifest
├── validation/                      # Validation reports
│   ├── general_validation.json     
│   ├── onnx_validation.json
│   ├── ios_validation.json
│   └── hailo_validation.json
└── docs/                           # Documentation
    ├── README.md                   # Bundle overview
    ├── deployment_guide.md         # Target-specific guides
    └── api_reference.md            # Model API documentation
```

### Bundle Metadata

```json
{
    "exp_hash": "a1b2c3d4e5f6",
    "exp_name": "dogs_behavioral_analysis",
    "created_at": "2024-01-15T10:30:45.123456Z",
    "created_by": "user@example.com",
    "system_version": "1.0.0",
    "commit_sha": "abc123def456",
    "model": {
        "name": "conv2d_fsq",
        "architecture": "Conv2d-VQ-HDP-HSMM",
        "parameters": 313000,
        "input_shape": [1, 9, 2, 100],
        "output_classes": 4,
        "accuracy": 0.7812
    },
    "formats": {
        "onnx": {
            "file": "models/model.onnx",
            "opset_version": 11,
            "file_size_mb": 1.2
        },
        "coreml": {
            "file": "models/coreml.mlpackage",
            "minimum_ios_version": "13.0",
            "file_size_mb": 1.1
        },
        "hailo": {
            "file": "models/hailo.hef",
            "hailo_version": "4.12.0",
            "file_size_mb": 0.8
        }
    },
    "validation": {
        "general": {"passed": true, "errors": 0, "warnings": 1},
        "onnx": {"passed": true, "errors": 0, "warnings": 0},
        "ios": {"passed": true, "errors": 0, "warnings": 0},
        "hailo": {"passed": true, "errors": 0, "warnings": 0}
    }
}
```

## Model Export System

### Multi-Format Export

```python
class ModelExporter:
    """Export models to multiple deployment formats."""
    
    def export_all(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_dir: Path,
        model_name: str = "model",
    ) -> Dict[str, Path]:
        """Export model to all supported formats."""
        
        formats = {}
        
        # ONNX Export (universal)
        onnx_path = self.export_onnx(
            model, input_shape, output_dir / f"{model_name}.onnx"
        )
        formats["onnx"] = onnx_path
        
        # CoreML Export (iOS)
        try:
            coreml_path = self.export_coreml(
                model, input_shape, output_dir / f"{model_name}.mlpackage"
            )
            formats["coreml"] = coreml_path
        except ImportError:
            logger.warning("CoreML export skipped: coremltools not available")
        
        # Hailo HEF Export (edge)
        try:
            hailo_path = self.export_hailo(
                onnx_path, output_dir / f"{model_name}.hef"
            )
            formats["hailo"] = hailo_path
        except Exception as e:
            logger.warning(f"Hailo export failed: {e}")
        
        return formats
```

### ONNX Export

```python
def export_onnx(
    self,
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    output_path: Path,
) -> Path:
    """Export model to ONNX format."""
    
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
        input_names=["imu_input"],
        output_names=["behavioral_codes", "confidence"],
        dynamic_axes={
            "imu_input": {0: "batch_size"},
            "behavioral_codes": {0: "batch_size"},
            "confidence": {0: "batch_size"},
        },
    )
    
    # Validate ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    logger.info(f"ONNX model exported: {output_path}")
    return output_path
```

### CoreML Export

```python
def export_coreml(
    self,
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    output_path: Path,
) -> Path:
    """Export model to CoreML format."""
    
    import coremltools as ct
    
    model.eval()
    
    # Trace model
    dummy_input = torch.randn(input_shape, dtype=torch.float32)
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="imu_input",
                shape=input_shape,
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="behavioral_codes", dtype=np.int32),
            ct.TensorType(name="confidence", dtype=np.float32),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS13,
    )
    
    # Add metadata
    coreml_model.short_description = "Behavioral synchrony analysis model"
    coreml_model.input_description["imu_input"] = "IMU sensor data (accel + gyro)"
    coreml_model.output_description["behavioral_codes"] = "Discrete behavioral codes"
    coreml_model.output_description["confidence"] = "Prediction confidence scores"
    
    # Save model
    coreml_model.save(str(output_path))
    
    logger.info(f"CoreML model exported: {output_path}")
    return output_path
```

### Hailo HEF Export

```python
def export_hailo(
    self,
    onnx_path: Path,
    output_path: Path,
) -> Path:
    """Compile ONNX model to Hailo HEF format."""
    
    # Use Hailo SDK to compile ONNX to HEF
    compile_command = [
        "hailo", "compiler",
        "--onnx", str(onnx_path),
        "--hef", str(output_path),
        "--hw-arch", "hailo8",
        "--batch-size", "1",
        "--optimization-level", "O2",
    ]
    
    result = subprocess.run(
        compile_command,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Hailo compilation failed: {result.stderr}")
    
    logger.info(f"Hailo HEF exported: {output_path}")
    return output_path
```

## Bundle Validation System

### Validation Framework

```python
@dataclass
class ValidationResult:
    """Bundle validation result."""
    
    passed: bool
    errors: List[str]
    warnings: List[str]
    checks_run: List[str]
    target: Optional[str] = None
    
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        self.passed = False
    
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)

class BundleValidator:
    """Validate deployment bundles for target platforms."""
    
    def validate(self, bundle: DeploymentBundle) -> ValidationResult:
        """Run general bundle validation."""
        
        result = ValidationResult(passed=True, errors=[], warnings=[], checks_run=[])
        
        # Check required files
        self._check_required_files(bundle, result)
        
        # Check metadata consistency
        self._check_metadata_consistency(bundle, result)
        
        # Check model formats
        self._check_model_formats(bundle, result)
        
        # Check configuration
        self._check_configuration(bundle, result)
        
        return result
```

### Target-Specific Validation

```python
def validate_for_target(
    self,
    bundle: DeploymentBundle,
    target: str,
) -> ValidationResult:
    """Validate bundle for specific deployment target."""
    
    # Run general validation first
    result = self.validate(bundle)
    result.target = target
    
    # Target-specific checks
    if target == "onnx":
        self._validate_onnx(bundle, result)
    elif target == "ios":
        self._validate_ios(bundle, result)
    elif target == "hailo":
        self._validate_hailo(bundle, result)
    else:
        result.add_error(f"Unknown target: {target}")
    
    return result

def _validate_onnx(self, bundle: DeploymentBundle, result: ValidationResult):
    """Validate ONNX deployment readiness."""
    
    onnx_path = bundle.bundle_dir / "models" / "model.onnx"
    
    if not onnx_path.exists():
        result.add_error("ONNX model file missing")
        return
    
    try:
        # Load and validate ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        # Check input/output shapes
        input_shape = onnx_model.graph.input[0].type.tensor_type.shape
        expected_shape = [1, 9, 2, 100]
        
        actual_shape = [dim.dim_value for dim in input_shape.dim]
        if actual_shape[1:] != expected_shape[1:]:  # Allow dynamic batch
            result.add_error(f"ONNX input shape mismatch: {actual_shape} vs {expected_shape}")
        
        # Check opset version
        opset_version = onnx_model.opset_import[0].version
        if opset_version < 11:
            result.add_warning(f"ONNX opset version {opset_version} < 11 (recommended)")
        
        result.checks_run.extend([
            "onnx_model_loadable",
            "onnx_model_valid",
            "onnx_input_shape",
            "onnx_opset_version",
        ])
        
    except Exception as e:
        result.add_error(f"ONNX validation failed: {e}")

def _validate_ios(self, bundle: DeploymentBundle, result: ValidationResult):
    """Validate iOS (CoreML) deployment readiness."""
    
    coreml_path = bundle.bundle_dir / "models" / "coreml.mlpackage"
    
    if not coreml_path.exists():
        result.add_error("CoreML model package missing")
        return
    
    try:
        import coremltools as ct
        
        # Load CoreML model
        coreml_model = ct.models.MLModel(str(coreml_path))
        
        # Check iOS version compatibility
        min_ios = coreml_model.get_spec().specificationVersion
        if min_ios < 5:  # iOS 13+
            result.add_warning("CoreML model may not support iOS 13+")
        
        # Check input specifications
        input_spec = coreml_model.get_spec().description.input[0]
        if input_spec.type.multiArrayType.dataType != 65568:  # FLOAT32
            result.add_error("CoreML input type must be FLOAT32")
        
        # Check model size
        model_size_mb = coreml_path.stat().st_size / (1024 * 1024)
        if model_size_mb > 100:
            result.add_warning(f"Large CoreML model: {model_size_mb:.1f}MB")
        
        result.checks_run.extend([
            "coreml_model_loadable",
            "coreml_ios_compatibility",
            "coreml_input_type",
            "coreml_model_size",
        ])
        
    except Exception as e:
        result.add_error(f"CoreML validation failed: {e}")

def _validate_hailo(self, bundle: DeploymentBundle, result: ValidationResult):
    """Validate Hailo deployment readiness."""
    
    hef_path = bundle.bundle_dir / "models" / "hailo.hef"
    
    if not hef_path.exists():
        result.add_error("Hailo HEF file missing")
        return
    
    try:
        # Check HEF file is valid
        hef_info = subprocess.run(
            ["hailo", "hef-info", str(hef_path)],
            capture_output=True,
            text=True,
        )
        
        if hef_info.returncode != 0:
            result.add_error("Invalid HEF file")
            return
        
        # Parse HEF info
        hef_data = hef_info.stdout
        
        # Check architecture
        if "hailo8" not in hef_data.lower():
            result.add_warning("HEF not compiled for Hailo-8")
        
        # Check batch size
        if "batch_size: 1" not in hef_data:
            result.add_warning("HEF batch size != 1 (edge deployment)")
        
        # Check model size
        hef_size_mb = hef_path.stat().st_size / (1024 * 1024)
        if hef_size_mb > 10:
            result.add_warning(f"Large HEF file: {hef_size_mb:.1f}MB")
        
        result.checks_run.extend([
            "hef_file_valid",
            "hef_architecture",
            "hef_batch_size",
            "hef_file_size",
        ])
        
    except FileNotFoundError:
        result.add_error("Hailo CLI tools not available")
    except Exception as e:
        result.add_error(f"Hailo validation failed: {e}")
```

## CLI Integration

### Bundle Management Commands

```bash
# Create deployment bundle
conv2d pack create --config conf/exp/dogs.yaml --model models/best.pth --name dogs_v1

# Validate bundle for deployment
conv2d pack verify a1b2c3d4 --target hailo

# List all bundles
conv2d pack list --verbose

# Archive bundle
conv2d pack archive a1b2c3d4 --format zip

# Clean old bundles (keep latest 5)
conv2d pack clean --keep 5

# Compare bundles
conv2d pack compare a1b2c3d4 e5f6g7h8 --output comparison.md
```

### CLI Implementation

```python
# src/conv2d/cli_pack.py
import click
from conv2d.packaging import ArtifactBundler, BundleValidator

@click.group()
def pack():
    """Artifact packaging commands."""
    pass

@pack.command()
@click.option("--config", required=True, help="Configuration file")
@click.option("--model", required=True, help="Model checkpoint path")
@click.option("--name", help="Bundle name")
@click.option("--output-dir", default="artifacts", help="Output directory")
def create(config, model, name, output_dir):
    """Create deployment bundle."""
    
    # Load configuration
    config_data = load_config(config)
    
    # Create bundler
    bundler = ArtifactBundler(output_base=output_dir)
    
    # Create bundle
    bundle = bundler.create_bundle(
        config=config_data,
        model_path=model,
        exp_name=name,
    )
    
    click.echo(f"✓ Bundle created: {bundle.exp_hash}")
    click.echo(f"  Path: {bundle.bundle_dir}")
    click.echo(f"  Formats: {', '.join(bundle.get_available_formats())}")

@pack.command()
@click.argument("bundle_hash")
@click.option("--target", help="Deployment target (onnx, ios, hailo)")
def verify(bundle_hash, target):
    """Validate deployment bundle."""
    
    # Find bundle
    bundler = ArtifactBundler()
    bundle = bundler.get_bundle(bundle_hash)
    
    if not bundle:
        click.echo(f"✗ Bundle not found: {bundle_hash}")
        return
    
    # Validate bundle
    validator = BundleValidator()
    
    if target:
        result = validator.validate_for_target(bundle, target)
        click.echo(f"Validating for {target.upper()} deployment...")
    else:
        result = validator.validate(bundle)
        click.echo("Running general validation...")
    
    if result.passed:
        click.echo("✓ Validation passed")
    else:
        click.echo("✗ Validation failed")
        for error in result.errors:
            click.echo(f"  ERROR: {error}")
    
    for warning in result.warnings:
        click.echo(f"  WARNING: {warning}")
    
    click.echo(f"Checks run: {len(result.checks_run)}")

@pack.command()
@click.option("--verbose", is_flag=True, help="Verbose output")
def list(verbose):
    """List deployment bundles."""
    
    bundler = ArtifactBundler()
    bundles = bundler.list_bundles()
    
    if not bundles:
        click.echo("No bundles found")
        return
    
    click.echo(f"Found {len(bundles)} bundles:")
    
    for bundle in sorted(bundles, key=lambda b: b.created_at, reverse=True):
        click.echo(f"  {bundle.exp_hash[:8]} - {bundle.exp_name}")
        
        if verbose:
            click.echo(f"    Created: {bundle.created_at}")
            click.echo(f"    Path: {bundle.bundle_dir}")
            click.echo(f"    Formats: {', '.join(bundle.get_available_formats())}")
            if hasattr(bundle, 'metadata') and 'model' in bundle.metadata:
                accuracy = bundle.metadata['model'].get('accuracy', 'N/A')
                click.echo(f"    Accuracy: {accuracy}")
```

## Usage Examples

### Complete Bundle Creation

```python
import torch
from pathlib import Path
from conv2d.packaging import ArtifactBundler

# Load trained model
model = torch.load("models/best_conv2d_vq.pth")

# Configuration
config = {
    "model": {
        "name": "conv2d_fsq",
        "architecture": "Conv2d-VQ-HDP-HSMM",
        "input_shape": [1, 9, 2, 100],
        "n_classes": 4,
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.001,
    },
}

# Evaluation metrics
metrics = {
    "accuracy": 0.7812,
    "macro_f1": 0.7654,
    "ece": 0.035,
    "perplexity": 125.6,
}

# Create bundle
bundler = ArtifactBundler(output_base="artifacts")
bundle = bundler.create_bundle(
    config=config,
    model=model,
    metrics=metrics,
    exp_name="dogs_behavioral_analysis",
)

print(f"Bundle created: {bundle.exp_hash}")
print(f"Available formats: {bundle.get_available_formats()}")
```

### Bundle Validation Pipeline

```python
from conv2d.packaging import BundleValidator

validator = BundleValidator()

# Validate for all targets
targets = ["onnx", "ios", "hailo"]
results = {}

for target in targets:
    result = validator.validate_for_target(bundle, target)
    results[target] = result
    
    print(f"\n{target.upper()} Validation:")
    print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    
    for error in result.errors:
        print(f"    ERROR: {error}")
    
    for warning in result.warnings:
        print(f"    WARNING: {warning}")

# Overall deployment readiness
ready_targets = [t for t, r in results.items() if r.passed]
print(f"\nDeployment ready: {ready_targets}")
```

### Bundle Deployment

```python
# Deploy to edge device
def deploy_to_edge(bundle_hash: str, device_ip: str):
    """Deploy bundle to edge device."""
    
    bundler = ArtifactBundler()
    bundle = bundler.get_bundle(bundle_hash)
    
    if not bundle:
        raise ValueError(f"Bundle not found: {bundle_hash}")
    
    # Validate for Hailo deployment
    validator = BundleValidator()
    result = validator.validate_for_target(bundle, "hailo")
    
    if not result.passed:
        raise RuntimeError(f"Bundle validation failed: {result.errors}")
    
    # Copy bundle to device
    bundle_archive = bundle.create_archive(format="tar.gz")
    
    rsync_cmd = [
        "rsync", "-avz",
        str(bundle_archive),
        f"pi@{device_ip}:/opt/models/"
    ]
    
    subprocess.run(rsync_cmd, check=True)
    
    # Extract on device
    extract_cmd = [
        "ssh", f"pi@{device_ip}",
        f"cd /opt/models && tar -xzf {bundle_archive.name}"
    ]
    
    subprocess.run(extract_cmd, check=True)
    
    print(f"✓ Bundle {bundle_hash} deployed to {device_ip}")

# Usage
deploy_to_edge("a1b2c3d4", "192.168.1.100")
```

### Bundle Comparison

```python
def compare_bundles(hash1: str, hash2: str) -> Dict[str, Any]:
    """Compare two deployment bundles."""
    
    bundler = ArtifactBundler()
    bundle1 = bundler.get_bundle(hash1)
    bundle2 = bundler.get_bundle(hash2)
    
    comparison = {
        "bundles": {
            "bundle1": {"hash": hash1, "name": bundle1.exp_name},
            "bundle2": {"hash": hash2, "name": bundle2.exp_name},
        },
        "metrics": {},
        "config": {},
        "formats": {},
    }
    
    # Compare metrics
    if bundle1.metrics and bundle2.metrics:
        for metric in ["accuracy", "macro_f1", "ece"]:
            if metric in bundle1.metrics and metric in bundle2.metrics:
                val1 = bundle1.metrics[metric]
                val2 = bundle2.metrics[metric]
                comparison["metrics"][metric] = {
                    "bundle1": val1,
                    "bundle2": val2,
                    "delta": val2 - val1,
                    "improvement": val2 > val1,
                }
    
    # Compare available formats
    formats1 = set(bundle1.get_available_formats())
    formats2 = set(bundle2.get_available_formats())
    
    comparison["formats"] = {
        "bundle1_only": list(formats1 - formats2),
        "bundle2_only": list(formats2 - formats1),
        "common": list(formats1 & formats2),
    }
    
    return comparison
```

## Best Practices

1. **Always validate bundles**: Run validation before deployment to catch issues early
2. **Use meaningful names**: Bundle names should indicate experiment purpose and version
3. **Include comprehensive metadata**: Document model architecture, training config, and metrics
4. **Test all target formats**: Validate ONNX, CoreML, and Hailo exports work correctly
5. **Monitor bundle sizes**: Keep models under deployment size limits (10MB for edge)
6. **Archive old bundles**: Clean up artifacts directory regularly to save space
7. **Version everything**: Include git commit SHA and system version in bundles
8. **Document deployment**: Include target-specific deployment guides in bundles

This packaging system ensures complete, validated, and deployment-ready model bundles for production behavioral analysis systems across multiple target platforms.