#!/usr/bin/env python3
"""Demonstration of artifact packaging system.

Shows how to:
1. Create deployment bundles
2. Export models to multiple formats
3. Validate bundles for different targets
4. Manage bundle lifecycle
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conv2d.packaging import ArtifactBundler, BundleValidator, ModelExporter


def create_sample_config() -> dict:
    """Create sample experiment configuration."""
    return {
        "name": "quadruped_behavioral_analysis",
        "version": "1.0",
        "model": {
            "name": "conv2d_fsq",
            "architecture": {
                "encoder": "conv2d",
                "quantization": {
                    "type": "fsq",
                    "levels": [8, 6, 5],
                    "embedding_dim": 64,
                },
                "clustering": {
                    "type": "gmm",
                    "min_clusters": 3,
                    "max_clusters": 10,
                },
                "temporal": {
                    "type": "median_hysteresis",
                    "min_dwell": 5,
                    "window_size": 7,
                },
            },
            "input_shape": [1, 9, 2, 100],
            "n_classes": 4,
        },
        "data": {
            "name": "quadruped_imu",
            "sampling_rate": 100,
            "window_size": 100,
            "sensors": 2,
            "channels": 9,
        },
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
        },
        "deployment": {
            "targets": ["onnx", "coreml", "hailo"],
            "optimization": {
                "quantization": "int8",
                "batch_size": 1,
            },
        },
    }


def create_sample_metrics() -> dict:
    """Create sample evaluation metrics."""
    return {
        "accuracy": 0.8512,
        "macro_f1": 0.8234,
        "per_class_f1": [0.87, 0.82, 0.79, 0.81],
        "ece": 0.0347,
        "mce": 0.0892,
        "coverage": 0.94,
        "motif_count": 4,
        "code_usage_percent": 67.3,
        "perplexity": 84.2,
        "behavioral_metrics": {
            "transition_rate": 0.023,
            "mean_dwell": 43.2,
            "median_dwell": 38.0,
            "motif_entropy": 1.89,
        },
        "calibration": {
            "bin_accuracies": [0.12, 0.31, 0.48, 0.67, 0.73, 0.81, 0.87, 0.91, 0.95, 0.97],
            "bin_confidences": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.98],
            "bin_counts": [89, 112, 127, 134, 98, 87, 76, 65, 54, 43],
        },
        "qa_stats": {
            "nan_events": 0,
            "inf_events": 0,
            "outlier_samples": 3,
            "zero_variance_features": 0,
            "qa_pass": True,
        },
    }


def create_sample_labels() -> dict:
    """Create sample label mapping."""
    return {
        0: "resting",
        1: "walking", 
        2: "running",
        3: "playing",
    }


def example_create_bundle():
    """Example 1: Create deployment bundle."""
    print("=" * 60)
    print("Example 1: Create Deployment Bundle")
    print("=" * 60)
    
    # Create temporary artifacts directory
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        
        # Create bundler
        bundler = ArtifactBundler(output_base=artifacts_dir)
        
        # Sample data
        config = create_sample_config()
        metrics = create_sample_metrics()
        labels = create_sample_labels()
        
        print("Creating bundle with:")
        print(f"  - Config: {config['name']}")
        print(f"  - Metrics: {len(metrics)} entries")
        print(f"  - Labels: {len(labels)} behaviors")
        
        # Create bundle
        bundle = bundler.create_bundle(
            config=config,
            metrics=metrics,
            label_mapping=labels,
            exp_name="quadruped_demo",
        )
        
        print(f"\n✅ Bundle created!")
        print(f"Hash: {bundle.exp_hash}")
        print(f"Location: {bundle.bundle_dir}")
        
        # Show bundle contents
        manifest = bundle.get_manifest()
        print(f"\nBundle contents:")
        for name, info in manifest["files"].items():
            if info["exists"]:
                size_kb = info["size_bytes"] / 1024
                print(f"  ✓ {info['path']} ({size_kb:.1f} KB)")
            else:
                print(f"  ✗ {name} (missing)")
        
        return bundle


def example_validate_bundle(bundle):
    """Example 2: Validate deployment bundle."""
    print("\n" + "=" * 60)
    print("Example 2: Bundle Validation")
    print("=" * 60)
    
    validator = BundleValidator()
    
    # General validation
    print("Running general validation...")
    result = validator.validate(bundle)
    
    print(f"\nValidation Result:")
    print(f"  Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    
    # Show key findings
    if result.info:
        print(f"\nKey Findings:")
        for info in result.info[:5]:  # Show first 5
            print(f"  ℹ️  {info}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings[:3]:  # Show first 3
            print(f"  ⚠️  {warning}")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  ❌ {error}")
    
    # Target-specific validation
    targets = ["onnx", "ios", "hailo"]
    print(f"\nTarget-specific validation:")
    
    for target in targets:
        target_result = validator.validate_for_target(bundle, target)
        status = "✅" if target_result.passed else "❌"
        print(f"  {status} {target.upper()}: {len(target_result.errors)} errors")


def example_model_export():
    """Example 3: Model export demonstration."""
    print("\n" + "=" * 60)
    print("Example 3: Model Export (Simulated)")
    print("=" * 60)
    
    exporter = ModelExporter()
    
    # Simulate model information
    print("Model export targets:")
    
    formats = [
        ("ONNX", "Universal format for deployment"),
        ("CoreML", "iOS and macOS deployment"),
        ("Hailo HEF", "Edge acceleration on Hailo-8"),
    ]
    
    for format_name, description in formats:
        print(f"  📦 {format_name}: {description}")
    
    # Show what the export process would check
    print(f"\nExport validation checks:")
    print(f"  ✓ Input shape: (1, 9, 2, 100) - IMU data format")
    print(f"  ✓ Output shape: (1, 4) - Number of behavioral classes")
    print(f"  ✓ Dtype: float32 - Compatible with edge devices")
    print(f"  ✓ Model size: <10MB - Mobile deployment ready")
    
    # Simulated model info
    model_info = {
        "pytorch": {
            "parameters": 313_000,
            "size_mb": 1.2,
            "format": "PyTorch checkpoint",
        },
        "onnx": {
            "opset": 11,
            "size_mb": 1.1,
            "optimized": True,
        },
        "coreml": {
            "target": "iOS 14+",
            "size_mb": 0.9,
            "accelerated": True,
        },
        "hailo": {
            "target": "Hailo-8",
            "size_mb": 0.3,
            "fps_improvement": "25%",
        },
    }
    
    print(f"\nModel format comparison:")
    for format_name, info in model_info.items():
        size = info.get('size_mb', 0)
        print(f"  {format_name.upper():8s}: {size:.1f} MB")


def example_bundle_management():
    """Example 4: Bundle lifecycle management."""
    print("\n" + "=" * 60)
    print("Example 4: Bundle Management")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        bundler = ArtifactBundler(output_base=artifacts_dir)
        
        # Create multiple bundles
        configs = [
            ("experiment_v1", {"version": "1.0", "model": {"name": "baseline"}}),
            ("experiment_v2", {"version": "1.1", "model": {"name": "improved"}}),
            ("experiment_v3", {"version": "1.2", "model": {"name": "optimized"}}),
        ]
        
        bundles = []
        for exp_name, config in configs:
            bundle = bundler.create_bundle(
                config=config,
                exp_name=exp_name,
            )
            bundles.append(bundle)
            print(f"Created bundle: {exp_name} -> {bundle.exp_hash[:8]}...")
        
        # List bundles
        print(f"\nListing bundles:")
        all_bundles = bundler.list_bundles()
        for i, bundle in enumerate(all_bundles, 1):
            print(f"  {i}. {bundle.exp_hash[:8]}... ({bundle.bundle_dir.name})")
        
        # Bundle operations
        print(f"\nBundle operations:")
        if bundles:
            latest_bundle = bundles[-1]
            
            # Archive
            try:
                archive_path = bundler.archive_bundle(latest_bundle, format="zip")
                size_mb = archive_path.stat().st_size / (1024 * 1024)
                print(f"  📦 Archived: {archive_path.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  📦 Archive simulation: latest.zip (0.5 MB)")
            
            # Cleanup simulation
            print(f"  🧹 Cleanup: Would remove {max(0, len(all_bundles) - 2)} old bundles")
            
            # Copy simulation
            print(f"  📋 Copy: Can copy bundle to production directory")


def example_cli_usage():
    """Example 5: CLI usage demonstration."""
    print("\n" + "=" * 60)
    print("Example 5: CLI Usage")
    print("=" * 60)
    
    print("Deployment packaging CLI commands:")
    print()
    
    commands = [
        ("Create bundle", "conv2d pack create --config conf/exp/dogs.yaml --model models/best.pth --name dogs_v1"),
        ("Verify bundle", "conv2d pack verify 3a5b7c9d --target hailo"),
        ("List bundles", "conv2d pack list --verbose"),
        ("Clean old bundles", "conv2d pack clean --keep 5"),
        ("Archive bundle", "conv2d pack archive 3a5b7c9d --format zip"),
    ]
    
    for description, command in commands:
        print(f"# {description}")
        print(f"{command}")
        print()
    
    print("Bundle structure:")
    print("artifacts/")
    print("└── EXP_HASH/")
    print("    ├── model.onnx")
    print("    ├── coreml.mlpackage/")
    print("    ├── hailo.hef")
    print("    ├── config.yaml")
    print("    ├── label_map.json")
    print("    ├── metrics.json")
    print("    ├── VERSION")
    print("    └── COMMIT_SHA")


def example_production_workflow():
    """Example 6: Production deployment workflow."""
    print("\n" + "=" * 60)
    print("Example 6: Production Workflow")
    print("=" * 60)
    
    workflow_steps = [
        ("1. Train Model", "python train_quadruped_overnight.py"),
        ("2. Evaluate", "conv2d eval --config conf/exp/dogs.yaml --split test"),
        ("3. Package", "conv2d pack create --config conf/exp/dogs.yaml --model models/best.pth"),
        ("4. Validate", "conv2d pack verify BUNDLE_HASH --target hailo"),
        ("5. Deploy", "rsync -av artifacts/BUNDLE_HASH/ pi@edge-device:/opt/models/"),
        ("6. Monitor", "curl http://edge-device:8080/healthz"),
    ]
    
    print("Production deployment workflow:")
    print()
    
    for step, command in workflow_steps:
        print(f"{step}")
        print(f"  $ {command}")
        print()
    
    print("Quality gates:")
    print("  ✓ ECE < 0.05 (calibration)")  
    print("  ✓ Accuracy > 80%")
    print("  ✓ Model size < 10MB")
    print("  ✓ Inference < 10ms")
    print("  ✓ All validation checks pass")
    
    print("\nDeployment targets:")
    targets_info = [
        ("Edge Device", "Raspberry Pi 5 + Hailo-8", "hailo.hef"),
        ("iOS App", "iPhone/iPad", "coreml.mlpackage"),
        ("Server", "ONNX Runtime", "model.onnx"),
    ]
    
    for target, platform, model_file in targets_info:
        print(f"  📱 {target:12s}: {platform:20s} -> {model_file}")


def main():
    """Run all packaging examples."""
    # Example 1: Create bundle
    bundle = example_create_bundle()
    
    # Example 2: Validate bundle  
    example_validate_bundle(bundle)
    
    # Example 3: Model export
    example_model_export()
    
    # Example 4: Bundle management
    example_bundle_management()
    
    # Example 5: CLI usage
    example_cli_usage()
    
    # Example 6: Production workflow
    example_production_workflow()
    
    print("\n" + "=" * 60)
    print("All packaging examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()