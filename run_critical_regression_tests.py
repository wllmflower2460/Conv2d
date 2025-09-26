#!/usr/bin/env python3
"""Run critical regression tests that protect production deployment.

These tests catch the failures that actually matter:
1. FSQ Determinism - same input + levels + seed ⇒ identical codes
2. Clustering Determinism - fixed seed + K ⇒ identical labels with Hungarian matching  
3. Temporal Policy - min-dwell/hysteresis are respected (no 1-2 frame flicker)
4. Shape/Dtype Contracts - float32 and canonical shapes across all stages
5. Packaging Bundle - artifact bundles have all required files for deployment

If these pass, the system is ready for production. If they fail, deployment will break.
"""

import sys
import subprocess
import time
from pathlib import Path

def run_test_suite():
    """Run critical test suite with reporting."""
    
    print("🎯 CRITICAL PRODUCTION REGRESSION TESTS")
    print("=" * 60)
    print("These tests protect against real deployment failures")
    
    # Critical tests that must pass for production
    critical_tests = [
        ("FSQ Determinism", "tests/test_fsq_determinism.py"),
        ("Clustering Determinism", "tests/test_clustering_determinism.py"),
        ("Temporal Policy Enforcement", "tests/test_temporal_policy.py"),
        ("Shape & Dtype Contracts", "tests/test_dtype_shapes.py"), 
        ("Packaging Bundle Validation", "tests/test_packaging_bundle.py"),
    ]
    
    results = {}
    total_start = time.time()
    
    for category, test_file in critical_tests:
        print(f"\n▶️  Running {category}...")
        
        start_time = time.time()
        
        # Run pytest with verbose output and fail fast
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            test_file,
            "-v", 
            "--tb=short",
            "--maxfail=1",  # Stop on first failure
        ], capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.2f}s)")
            results[category] = "PASSED"
        else:
            print(f"❌ FAILED ({duration:.2f}s)")
            print("CRITICAL FAILURE:")
            print(result.stdout)
            print(result.stderr)
            results[category] = "FAILED"
            
            # For critical tests, stop on first failure
            print(f"\n💥 CRITICAL TEST FAILED: {category}")
            print("This failure WILL break production deployment!")
            print("Fix this issue before proceeding.")
            break
    
    total_duration = time.time() - total_start
    
    # Summary report
    print(f"\n📊 CRITICAL TEST RESULTS ({total_duration:.1f}s total)")
    print("=" * 60)
    
    passed = sum(1 for status in results.values() if status == "PASSED")
    failed = len(results) - passed
    
    for category, status in results.items():
        status_icon = "✅" if status == "PASSED" else "❌" 
        print(f"{status_icon} {category}: {status}")
    
    print(f"\nOverall: {passed}/{len(critical_tests)} critical tests passed")
    
    if failed > 0:
        print(f"\n🚨 {failed} CRITICAL TEST(S) FAILED")
        print("DEPLOYMENT STATUS: BLOCKED")
        print("\nProduction deployment is NOT SAFE until these pass:")
        for category, status in results.items():
            if status == "FAILED":
                print(f"  - {category}")
        
        print("\nWhy these tests matter:")
        print("  • FSQ Determinism: Same inputs must produce identical codes")
        print("  • Clustering Determinism: Reproducible behavioral analysis")  
        print("  • Temporal Policy: No behavioral flickering artifacts")
        print("  • Shape/Dtype: Edge deployment will crash with wrong types")
        print("  • Packaging: Deployment bundles missing critical files")
        
        return 1
    else:
        print("\n🎉 ALL CRITICAL TESTS PASSED")
        print("DEPLOYMENT STATUS: SAFE ✅")
        print("\nSystem ready for production:")
        print("  ✓ Deterministic and reproducible")
        print("  ✓ No temporal artifacts")
        print("  ✓ Edge deployment safe")
        print("  ✓ Complete packaging")
        print("\nProceeding with deployment...")
        return 0


def run_supplementary_tests():
    """Run supplementary tests (non-blocking)."""
    
    print("\n🔍 SUPPLEMENTARY REGRESSION TESTS")
    print("=" * 60)
    print("Additional quality checks (non-blocking)")
    
    supplementary_tests = [
        ("Shape & Dtype Enforcement (Legacy)", "tests/test_shape_dtype_enforcement.py"),
        ("Determinism Validation (Legacy)", "tests/test_determinism.py"),
        ("Temporal Assertions (Legacy)", "tests/test_temporal_assertions.py"), 
        ("Calibration Improvements", "tests/test_calibration_improvements.py"),
        ("Speed Benchmarks", "tests/test_speed_benchmarks.py"),
    ]
    
    for category, test_file in supplementary_tests:
        print(f"\n▶️  Running {category}...")
        
        start_time = time.time()
        
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=line",
        ], capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.2f}s)")
        else:
            print(f"⚠️  FAILED ({duration:.2f}s) - Non-blocking")
    
    print("\n✓ Supplementary tests completed")


def main():
    """Run critical tests followed by supplementary tests."""
    
    # Run critical tests first - these must pass
    critical_result = run_test_suite()
    
    if critical_result == 0:
        # Only run supplementary tests if critical tests pass
        run_supplementary_tests()
        
        print("\n🏁 FINAL STATUS: PRODUCTION READY")
        print("All critical protection tests passed!")
        
    return critical_result


if __name__ == "__main__":
    sys.exit(main())