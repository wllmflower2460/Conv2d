#!/usr/bin/env python3
"""Quick test of critical protection tests to verify they work."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_fsq_basic():
    """Test basic FSQ functionality."""
    try:
        import torch
        from conv2d.features.fsq_contract import encode_fsq
        
        # Basic determinism test
        torch.manual_seed(42)
        x = torch.randn(4, 9, 2, 100, dtype=torch.float32)
        
        result1 = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        result2 = encode_fsq(x, levels=[8, 6, 5], reset_stats=True)
        
        # Check determinism
        codes_match = torch.equal(result1.codes, result2.codes)
        features_match = torch.allclose(result1.features, result2.features, atol=1e-8)
        
        # Check dtypes
        codes_int32 = result1.codes.dtype == torch.int32
        features_float32 = result1.features.dtype == torch.float32
        
        # Check shapes
        correct_codes_shape = result1.codes.shape == (4, 64)
        correct_features_batch = result1.features.shape[0] == 4
        
        print("✓ FSQ Basic Tests:")
        print(f"  - Determinism (codes): {codes_match}")
        print(f"  - Determinism (features): {features_match}")
        print(f"  - Codes dtype int32: {codes_int32}")
        print(f"  - Features dtype float32: {features_float32}")
        print(f"  - Codes shape correct: {correct_codes_shape}")
        print(f"  - Features batch correct: {correct_features_batch}")
        
        return all([codes_match, features_match, codes_int32, features_float32, 
                   correct_codes_shape, correct_features_batch])
        
    except Exception as e:
        print(f"❌ FSQ test failed: {e}")
        return False

def test_clustering_basic():
    """Test basic clustering functionality."""
    try:
        import numpy as np
        from conv2d.clustering.kmeans import KMeansClusterer
        
        # Create test data
        np.random.seed(42)
        features = np.random.randn(100, 64).astype(np.float32)
        
        # Test determinism
        clusterer1 = KMeansClusterer(random_state=42)
        labels1 = clusterer1.fit_predict(features, k=4)
        
        clusterer2 = KMeansClusterer(random_state=42)
        labels2 = clusterer2.fit_predict(features, k=4)
        
        # Check properties
        deterministic = np.array_equal(labels1, labels2)
        correct_dtype = labels1.dtype == np.int32
        correct_shape = labels1.shape == (100,)
        valid_range = (labels1.min() >= 0) and (labels1.max() < 4)
        correct_k = len(np.unique(labels1)) == 4
        
        print("✓ Clustering Basic Tests:")
        print(f"  - Deterministic: {deterministic}")
        print(f"  - Correct dtype (int32): {correct_dtype}")
        print(f"  - Correct shape: {correct_shape}")
        print(f"  - Valid label range: {valid_range}")
        print(f"  - Correct K clusters: {correct_k}")
        
        return all([deterministic, correct_dtype, correct_shape, valid_range, correct_k])
        
    except Exception as e:
        print(f"❌ Clustering test failed: {e}")
        return False

def test_temporal_basic():
    """Test basic temporal policy functionality."""
    try:
        import numpy as np
        from conv2d.temporal.median import MedianHysteresisPolicy
        
        # Create sequence with flickers
        labels = np.array([
            [0, 0, 1, 0, 0, 1, 1, 1, 2, 0],  # 1-frame flickers
        ])
        
        policy = MedianHysteresisPolicy(min_dwell=3, window_size=5)
        smoothed = policy.smooth(labels)
        
        # Check properties
        shape_preserved = smoothed.shape == labels.shape
        dtype_preserved = smoothed.dtype == labels.dtype
        
        # Check no new states
        original_states = set(labels.flatten())
        smoothed_states = set(smoothed.flatten()) 
        no_new_states = smoothed_states <= original_states
        
        # Check for reduced transitions
        transitions_before = np.sum(np.diff(labels[0]) != 0)
        transitions_after = np.sum(np.diff(smoothed[0]) != 0)
        reduced_transitions = transitions_after <= transitions_before
        
        print("✓ Temporal Basic Tests:")
        print(f"  - Shape preserved: {shape_preserved}")
        print(f"  - Dtype preserved: {dtype_preserved}")
        print(f"  - No new states: {no_new_states}")
        print(f"  - Transitions reduced: {reduced_transitions} ({transitions_before}→{transitions_after})")
        
        return all([shape_preserved, dtype_preserved, no_new_states, reduced_transitions])
        
    except Exception as e:
        print(f"❌ Temporal test failed: {e}")
        return False

def test_packaging_basic():
    """Test basic packaging functionality."""
    try:
        import tempfile
        import json
        import yaml
        from pathlib import Path
        from conv2d.packaging.bundler import ArtifactBundler
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            bundler = ArtifactBundler(output_base=artifacts_dir)
            
            config = {
                "model": {"name": "test_model", "n_classes": 4},
                "training": {"epochs": 50}
            }
            
            bundle = bundler.create_bundle(config=config, exp_name="test")
            
            # Check required files exist
            required_files = ["config.yaml", "label_map.json", "metrics.json", 
                            "VERSION", "COMMIT_SHA", "manifest.json"]
            
            files_exist = all((bundle.bundle_dir / f).exists() for f in required_files)
            files_non_empty = all((bundle.bundle_dir / f).stat().st_size > 0 for f in required_files)
            
            # Check config is valid YAML
            config_valid = True
            try:
                with open(bundle.bundle_dir / "config.yaml", 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    config_valid = "model" in loaded_config
            except:
                config_valid = False
                
            # Check label_map is valid JSON
            label_map_valid = True
            try:
                with open(bundle.bundle_dir / "label_map.json", 'r') as f:
                    label_map = json.load(f)
                    label_map_valid = isinstance(label_map, dict)
            except:
                label_map_valid = False
            
            print("✓ Packaging Basic Tests:")
            print(f"  - Required files exist: {files_exist}")
            print(f"  - Files non-empty: {files_non_empty}")
            print(f"  - Config YAML valid: {config_valid}")
            print(f"  - Label map JSON valid: {label_map_valid}")
            
            return all([files_exist, files_non_empty, config_valid, label_map_valid])
            
    except Exception as e:
        print(f"❌ Packaging test failed: {e}")
        return False

def main():
    """Run basic verification of critical tests."""
    print("🧪 CRITICAL TEST VERIFICATION")
    print("=" * 50)
    print("Verifying that our critical protection tests work correctly...\n")
    
    tests = [
        ("FSQ Determinism & Contracts", test_fsq_basic),
        ("Clustering Determinism", test_clustering_basic), 
        ("Temporal Policy Enforcement", test_temporal_basic),
        ("Packaging Bundle Creation", test_packaging_basic),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Running {name}...")
        result = test_func()
        results.append(result)
        print(f"{'✅ PASS' if result else '❌ FAIL'}\n")
    
    all_passed = all(results)
    
    print("=" * 50)
    print(f"VERIFICATION SUMMARY: {sum(results)}/{len(results)} tests passed")
    
    if all_passed:
        print("🎉 Critical tests are working correctly!")
        print("The production gate system is functional.")
    else:
        print("⚠️  Some critical tests need attention.")
        print("Fix these before relying on the gate system.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())