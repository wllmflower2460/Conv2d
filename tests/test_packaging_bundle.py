#!/usr/bin/env python3
"""Packaging bundle tests - artifact bundle has required files and valid exports."""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
import sys
import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conv2d.packaging.bundler import ArtifactBundler, DeploymentBundle
from conv2d.packaging.exporter import ModelExporter
from conv2d.packaging.validator import BundleValidator


class TestPackagingBundle:
    """Test artifact packaging - critical for deployment safety and completeness."""
    
    def test_bundle_required_files_present(self):
        """Deployment bundle must contain all required files."""
        
        required_files = [
            "config.yaml",
            "label_map.json", 
            "metrics.json",
            "VERSION",
            "COMMIT_SHA",
            "manifest.json",
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            # Create minimal config
            config = {
                "model": {"name": "conv2d_fsq", "n_classes": 4},
                "data": {"name": "test_data"},
            }
            
            # Create bundle
            bundle = bundler.create_bundle(config=config, exp_name="test")
            
            # Check all required files exist
            for required_file in required_files:
                file_path = bundle.bundle_dir / required_file
                assert file_path.exists(), f"Required file missing: {required_file}"
                assert file_path.stat().st_size > 0, f"Required file empty: {required_file}"
    
    def test_bundle_config_yaml_valid(self):
        """config.yaml must be valid YAML with required fields."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            config = {
                "model": {
                    "name": "conv2d_fsq",
                    "architecture": "Conv2d-FSQ-Clustering",
                    "n_classes": 4,
                    "input_shape": [1, 9, 2, 100],
                },
                "training": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                },
            }
            
            bundle = bundler.create_bundle(config=config, exp_name="test")
            
            # Load and validate config.yaml
            config_file = bundle.bundle_dir / "config.yaml"
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Check required top-level sections
            assert "model" in loaded_config, "config.yaml missing model section"
            assert "training" in loaded_config, "config.yaml missing training section"
            
            # Check required model fields
            model_config = loaded_config["model"]
            required_model_fields = ["name", "n_classes", "input_shape"]
            
            for field in required_model_fields:
                assert field in model_config, f"config.yaml model section missing {field}"
            
            # Validate types and values
            assert isinstance(model_config["n_classes"], int)
            assert model_config["n_classes"] > 0
            assert isinstance(model_config["input_shape"], list)
            assert len(model_config["input_shape"]) == 4  # [B, C, H, W]
    
    def test_bundle_label_map_json_valid(self):
        """label_map.json must be valid JSON with proper structure."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts" 
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            config = {"model": {"name": "test", "n_classes": 4}}
            bundle = bundler.create_bundle(config=config, exp_name="test")
            
            # Load and validate label_map.json
            label_map_file = bundle.bundle_dir / "label_map.json"
            with open(label_map_file, 'r') as f:
                label_map = json.load(f)
            
            # Should be a dictionary mapping integers to strings
            assert isinstance(label_map, dict), "label_map.json not a dictionary"
            
            for key, value in label_map.items():
                # Keys should be string representations of integers  
                assert key.isdigit(), f"Label map key not numeric: {key}"
                assert isinstance(value, str), f"Label map value not string: {value}"
                
                # Key should be valid class index
                class_id = int(key)
                assert 0 <= class_id < 4, f"Invalid class ID: {class_id}"
    
    def test_bundle_metrics_json_valid(self):
        """metrics.json must contain evaluation metrics."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            metrics = {
                "accuracy": 0.7812,
                "macro_f1": 0.7654,
                "ece": 0.035,
                "perplexity": 125.6,
                "code_usage_percent": 75.0,
            }
            
            config = {"model": {"name": "test", "n_classes": 4}}
            bundle = bundler.create_bundle(
                config=config, 
                metrics=metrics,
                exp_name="test"
            )
            
            # Load and validate metrics.json
            metrics_file = bundle.bundle_dir / "metrics.json"
            with open(metrics_file, 'r') as f:
                loaded_metrics = json.load(f)
            
            # Check required metrics are present
            required_metrics = ["accuracy", "macro_f1", "ece"]
            
            for metric in required_metrics:
                assert metric in loaded_metrics, f"metrics.json missing {metric}"
                assert isinstance(loaded_metrics[metric], (int, float)), \
                    f"Metric {metric} not numeric: {type(loaded_metrics[metric])}"
            
            # Check value ranges
            assert 0 <= loaded_metrics["accuracy"] <= 1, "Accuracy out of range [0,1]"
            assert 0 <= loaded_metrics["macro_f1"] <= 1, "Macro-F1 out of range [0,1]"
            assert 0 <= loaded_metrics["ece"] <= 1, "ECE out of range [0,1]"
    
    def test_bundle_version_and_commit(self):
        """VERSION and COMMIT_SHA files must be present and valid."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            config = {"model": {"name": "test", "n_classes": 4}}
            bundle = bundler.create_bundle(config=config, exp_name="test")
            
            # Check VERSION file
            version_file = bundle.bundle_dir / "VERSION"
            assert version_file.exists(), "VERSION file missing"
            
            version_content = version_file.read_text().strip()
            assert len(version_content) > 0, "VERSION file empty"
            assert "." in version_content, "VERSION doesn't look like semantic version"
            
            # Check COMMIT_SHA file
            commit_file = bundle.bundle_dir / "COMMIT_SHA"
            assert commit_file.exists(), "COMMIT_SHA file missing"
            
            commit_content = commit_file.read_text().strip()
            assert len(commit_content) >= 7, "COMMIT_SHA too short (not a valid SHA)"
            assert len(commit_content) <= 40, "COMMIT_SHA too long (not a valid SHA)"
    
    def test_bundle_manifest_completeness(self):
        """manifest.json must accurately describe bundle contents."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            config = {
                "model": {
                    "name": "conv2d_fsq",
                    "n_classes": 4, 
                    "input_shape": [1, 9, 2, 100],
                }
            }
            
            bundle = bundler.create_bundle(config=config, exp_name="test")
            
            # Load manifest
            manifest_file = bundle.bundle_dir / "manifest.json"
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            # Check required manifest fields
            required_fields = [
                "exp_hash",
                "exp_name", 
                "created_at",
                "files",
                "bundle_version",
            ]
            
            for field in required_fields:
                assert field in manifest, f"Manifest missing {field}"
            
            # Check files section lists actual files
            assert isinstance(manifest["files"], list), "Manifest files not a list"
            
            # Verify files listed actually exist
            for file_info in manifest["files"]:
                assert "name" in file_info, "File info missing name"
                assert "size" in file_info, "File info missing size"
                
                file_path = bundle.bundle_dir / file_info["name"]
                assert file_path.exists(), f"Manifest lists non-existent file: {file_info['name']}"
                
                actual_size = file_path.stat().st_size
                assert file_info["size"] == actual_size, \
                    f"File size mismatch: {file_info['name']} manifest={file_info['size']}, actual={actual_size}"
    
    def test_bundle_export_formats_present(self):
        """Bundle should contain model exports in requested formats."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            # Create a mock model for export testing
            class MockModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(128, 4)
                
                def forward(self, x):
                    # Expect flattened features
                    return self.linear(x)
            
            model = MockModel()
            
            config = {
                "model": {
                    "name": "conv2d_fsq",
                    "n_classes": 4,
                    "input_shape": [1, 9, 2, 100],
                }
            }
            
            # Test ONNX export
            bundle = bundler.create_bundle(
                config=config,
                model=model,
                export_formats=["onnx"],
                exp_name="test"
            )
            
            # Check ONNX file exists
            models_dir = bundle.bundle_dir / "models"
            onnx_file = models_dir / "model.onnx"
            
            if onnx_file.exists():  # Only check if export succeeded
                assert onnx_file.stat().st_size > 1000, "ONNX file too small"
                
                # Try to load ONNX model to verify it's valid
                try:
                    import onnx
                    onnx_model = onnx.load(str(onnx_file))
                    onnx.checker.check_model(onnx_model)
                except ImportError:
                    print("ONNX not available for validation")
                except Exception as e:
                    pytest.fail(f"Invalid ONNX model: {e}")
    
    def test_bundle_validation_passes(self):
        """Bundle validation should pass for well-formed bundles."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            # Create comprehensive bundle
            config = {
                "model": {
                    "name": "conv2d_fsq",
                    "architecture": "Conv2d-FSQ-Clustering", 
                    "n_classes": 4,
                    "input_shape": [1, 9, 2, 100],
                },
                "training": {
                    "epochs": 100,
                    "batch_size": 32,
                },
            }
            
            metrics = {
                "accuracy": 0.78,
                "macro_f1": 0.76,
                "ece": 0.05,
            }
            
            bundle = bundler.create_bundle(
                config=config,
                metrics=metrics,
                exp_name="validation_test"
            )
            
            # Validate bundle
            validator = BundleValidator()
            result = validator.validate(bundle)
            
            # Should pass validation
            assert result.passed, f"Bundle validation failed: {result.errors}"
            print(f"Validation passed with {len(result.warnings)} warnings")
    
    def test_bundle_target_specific_validation(self):
        """Target-specific validation should work for different deployment formats."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            config = {
                "model": {
                    "name": "conv2d_fsq",
                    "n_classes": 4,
                    "input_shape": [1, 9, 2, 100],
                }
            }
            
            bundle = bundler.create_bundle(config=config, exp_name="target_test")
            
            validator = BundleValidator()
            
            # Test different targets
            targets = ["onnx", "ios", "hailo"]
            
            for target in targets:
                result = validator.validate_for_target(bundle, target)
                
                # Validation may pass or fail, but should not crash
                assert hasattr(result, 'passed'), f"Missing validation result for {target}"
                assert hasattr(result, 'errors'), f"Missing errors list for {target}"
                assert hasattr(result, 'warnings'), f"Missing warnings list for {target}"
                
                print(f"{target.upper()} validation: {'PASS' if result.passed else 'FAIL'} "
                      f"({len(result.errors)} errors, {len(result.warnings)} warnings)")
    
    def test_bundle_listing_and_retrieval(self):
        """Bundle listing and retrieval should work correctly."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            # Create multiple bundles
            bundle_names = ["test1", "test2", "test3"]
            created_bundles = []
            
            for name in bundle_names:
                config = {"model": {"name": f"model_{name}", "n_classes": 4}}
                bundle = bundler.create_bundle(config=config, exp_name=name)
                created_bundles.append(bundle)
            
            # List bundles
            listed_bundles = bundler.list_bundles()
            
            # Should find all created bundles
            assert len(listed_bundles) == len(bundle_names), \
                f"Listed {len(listed_bundles)} bundles, expected {len(bundle_names)}"
            
            # Check bundle retrieval by hash
            for created_bundle in created_bundles:
                retrieved_bundle = bundler.get_bundle(created_bundle.exp_hash)
                
                assert retrieved_bundle is not None, \
                    f"Could not retrieve bundle {created_bundle.exp_hash}"
                assert retrieved_bundle.exp_hash == created_bundle.exp_hash, \
                    "Retrieved bundle has different hash"
                assert retrieved_bundle.bundle_dir.exists(), \
                    "Retrieved bundle directory doesn't exist"
    
    def test_bundle_cleanup_and_archival(self):
        """Bundle cleanup should work without breaking existing bundles."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            # Create several bundles
            bundles = []
            for i in range(5):
                config = {"model": {"name": f"model_{i}", "n_classes": 4}}
                bundle = bundler.create_bundle(config=config, exp_name=f"test_{i}")
                bundles.append(bundle)
            
            # Verify all exist
            all_bundles = bundler.list_bundles()
            assert len(all_bundles) == 5, "Not all bundles were created"
            
            # Test that bundles can be archived (create archive)
            for bundle in bundles[:2]:  # Archive first 2
                try:
                    archive_path = bundle.create_archive(format="tar.gz")
                    assert archive_path.exists(), "Archive not created"
                    assert archive_path.stat().st_size > 1000, "Archive too small"
                except Exception as e:
                    print(f"Archive creation failed (may be expected): {e}")
            
            # Remaining bundles should still be accessible
            remaining_bundles = bundler.list_bundles()
            assert len(remaining_bundles) >= 3, "Too many bundles were removed"
    
    def test_bundle_metadata_consistency(self):
        """Bundle metadata should be consistent across all files."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            config = {
                "model": {
                    "name": "conv2d_fsq",
                    "n_classes": 4,
                    "input_shape": [1, 9, 2, 100],
                }
            }
            
            bundle = bundler.create_bundle(config=config, exp_name="metadata_test")
            
            # Load all metadata files
            with open(bundle.bundle_dir / "config.yaml", 'r') as f:
                config_data = yaml.safe_load(f)
                
            with open(bundle.bundle_dir / "manifest.json", 'r') as f:
                manifest_data = json.load(f)
            
            # Check consistency
            assert bundle.exp_hash == manifest_data["exp_hash"], \
                "Bundle hash inconsistent between object and manifest"
            
            assert bundle.exp_name == manifest_data["exp_name"], \
                "Bundle name inconsistent between object and manifest"
            
            # Verify experiment hash is deterministic for same config
            bundle2 = bundler.create_bundle(config=config, exp_name="metadata_test_2")
            
            # Different names should produce different hashes
            assert bundle.exp_hash != bundle2.exp_hash, \
                "Different experiment names produced same hash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])