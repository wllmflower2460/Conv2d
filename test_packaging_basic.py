#!/usr/bin/env python3
"""Basic test of packaging system functionality."""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from conv2d.packaging import ArtifactBundler, BundleValidator


def test_basic_packaging():
    """Test basic packaging functionality."""
    print("Testing basic packaging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        
        # Create bundler
        bundler = ArtifactBundler(output_base=artifacts_dir)
        
        # Simple config
        config = {
            "model": {
                "name": "test_model",
                "n_classes": 4,
                "input_shape": [1, 9, 2, 100],
            },
        }
        
        # Create bundle
        bundle = bundler.create_bundle(
            config=config,
            exp_name="test",
        )
        
        print(f"✓ Bundle created: {bundle.exp_hash}")
        
        # Check required files exist
        assert bundle.config_file.exists(), "Config file missing"
        assert bundle.label_map.exists(), "Label map missing"
        assert bundle.version_file.exists(), "Version file missing"
        
        print("✓ Required files present")
        
        # Test validation
        validator = BundleValidator()
        result = validator.validate(bundle)
        
        print(f"✓ Validation completed: {len(result.errors)} errors, {len(result.warnings)} warnings")
        
        # Test bundle listing
        bundles = bundler.list_bundles()
        assert len(bundles) == 1, "Bundle not listed"
        assert bundles[0].exp_hash == bundle.exp_hash, "Bundle hash mismatch"
        
        print("✓ Bundle listing works")
        
        # Test manifest
        manifest = bundle.get_manifest()
        assert "files" in manifest, "Manifest missing files"
        
        print("✓ Manifest generation works")
        
        return True


def test_validation_system():
    """Test validation system."""
    print("\nTesting validation system...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        bundler = ArtifactBundler(output_base=artifacts_dir)
        
        # Create bundle with comprehensive config
        config = {
            "model": {
                "name": "conv2d_fsq",
                "n_classes": 4,
                "input_shape": [1, 9, 2, 100],
            },
            "data": {
                "name": "test_data",
            },
        }
        
        metrics = {
            "accuracy": 0.85,
            "macro_f1": 0.82,
            "ece": 0.05,
        }
        
        bundle = bundler.create_bundle(
            config=config,
            metrics=metrics,
            exp_name="validation_test",
        )
        
        # Test different validation targets
        validator = BundleValidator()
        
        targets = ["onnx", "ios", "hailo"]
        for target in targets:
            result = validator.validate_for_target(bundle, target)
            print(f"✓ {target.upper()} validation: {len(result.errors)} errors")
        
        # Test general validation
        result = validator.validate(bundle)
        print(f"✓ General validation: {'PASSED' if result.passed else 'FAILED'}")
        
        return True


def main():
    """Run basic packaging tests."""
    print("🎯 BASIC PACKAGING TESTS")
    print("=" * 50)
    
    try:
        test_basic_packaging()
        test_validation_system()
        
        print("\n✅ ALL TESTS PASSED")
        print("Packaging system: FUNCTIONAL")
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())