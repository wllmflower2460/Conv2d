#!/usr/bin/env python3
"""Run comprehensive regression tests that catch real failures.

These tests ensure:
1. Shape & dtype enforcement (edge deployment safety)
2. Deterministic behavior (reproducibility) 
3. Temporal coherence (no flickering)
4. Calibration quality (trustworthiness)
5. Performance targets (real-time capability)
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_test_module(module_name: str, description: str) -> bool:
    """Run a test module and report results."""
    print("\n" + "=" * 60)
    print(f"Running: {description}")
    print("=" * 60)
    
    try:
        # Import and run the test module
        if module_name == "test_shape_dtype":
            from tests.test_shape_dtype_enforcement import (
                test_input_validation_strict,
                test_fsq_output_shapes_strict,
                test_clustering_preserves_dtype,
                test_temporal_preserves_dtype,
                test_metrics_handle_dtypes,
                test_pipeline_end_to_end_shapes,
                test_mixed_precision_rejection,
                test_batch_size_edge_cases,
                test_output_range_validation,
            )
            
            test_input_validation_strict()
            print("✓ Input validation strict")
            test_fsq_output_shapes_strict()
            print("✓ FSQ output shapes strict")
            test_clustering_preserves_dtype()
            print("✓ Clustering preserves dtype")
            test_temporal_preserves_dtype()
            print("✓ Temporal preserves dtype")
            test_metrics_handle_dtypes()
            print("✓ Metrics handle dtypes")
            test_pipeline_end_to_end_shapes()
            print("✓ Pipeline end-to-end shapes")
            test_mixed_precision_rejection()
            print("✓ Mixed precision rejection")
            test_batch_size_edge_cases()
            print("✓ Batch size edge cases")
            test_output_range_validation()
            print("✓ Output range validation")
            
        elif module_name == "test_determinism":
            from tests.test_determinism import (
                test_fsq_determinism,
                test_fsq_code_usage,
                test_clustering_determinism_with_seed,
                test_hungarian_matching_stability,
                test_temporal_determinism,
                test_pipeline_determinism_end_to_end,
                test_determinism_with_different_batch_sizes,
                test_config_hash_determinism,
                test_numpy_torch_consistency,
            )
            
            test_fsq_determinism()
            test_fsq_code_usage()
            test_clustering_determinism_with_seed()
            test_hungarian_matching_stability()
            test_temporal_determinism()
            test_pipeline_determinism_end_to_end()
            test_determinism_with_different_batch_sizes()
            test_config_hash_determinism()
            test_numpy_torch_consistency()
            
        elif module_name == "test_temporal":
            from tests.test_temporal_assertions import (
                test_min_dwell_enforcement,
                test_no_single_frame_flickers,
                test_hysteresis_thresholds,
                test_transition_monotonicity,
                test_state_preservation,
                test_batch_consistency,
                test_edge_case_handling,
                test_temporal_causality,
                test_real_world_patterns,
            )
            
            test_min_dwell_enforcement()
            test_no_single_frame_flickers()
            test_hysteresis_thresholds()
            test_transition_monotonicity()
            test_state_preservation()
            test_batch_consistency()
            test_edge_case_handling()
            test_temporal_causality()
            test_real_world_patterns()
            
        elif module_name == "test_calibration":
            from tests.test_calibration_improvements import (
                test_ece_calculation,
                test_smoothing_improves_calibration,
                test_mce_bounds,
                test_confidence_histogram,
                test_reliability_diagram_bins,
                test_calibration_with_class_imbalance,
                test_temperature_scaling_effect,
                test_brier_score_decomposition,
            )
            
            test_ece_calculation()
            test_smoothing_improves_calibration()
            test_mce_bounds()
            test_confidence_histogram()
            test_reliability_diagram_bins()
            test_calibration_with_class_imbalance()
            test_temperature_scaling_effect()
            test_brier_score_decomposition()
            
        elif module_name == "test_speed":
            from tests.test_speed_benchmarks import (
                test_fsq_encode_speed,
                test_clustering_speed,
                test_temporal_smoothing_speed,
                test_metrics_computation_speed,
                test_pipeline_end_to_end_speed,
                test_memory_efficiency,
                test_parallelization_speedup,
                generate_performance_report,
            )
            
            test_fsq_encode_speed()
            test_clustering_speed()
            test_temporal_smoothing_speed()
            test_metrics_computation_speed()
            test_pipeline_end_to_end_speed()
            test_memory_efficiency()
            test_parallelization_speedup()
            generate_performance_report()
        
        print(f"\n✅ {description} PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ {description} FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all regression tests."""
    print("🎯 COMPREHENSIVE REGRESSION TEST SUITE")
    print("=" * 60)
    print("Testing critical invariants for production deployment")
    
    test_suites = [
        ("test_shape_dtype", "Shape & Dtype Enforcement"),
        ("test_determinism", "Deterministic Behavior"),
        ("test_temporal", "Temporal Policy Assertions"),
        ("test_calibration", "Calibration Improvements"),
        ("test_speed", "Speed Micro-benchmarks"),
    ]
    
    results = {}
    for module, description in test_suites:
        results[description] = run_test_module(module, description)
    
    # Summary
    print("\n" + "=" * 60)
    print("REGRESSION TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for description, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {description}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL REGRESSION TESTS PASSED!")
        print("Production deployment: SAFE")
        print("Edge latency: PRESERVED")
        print("Reproducibility: GUARANTEED")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Fix failures before deployment!")
        return 1


if __name__ == "__main__":
    sys.exit(main())