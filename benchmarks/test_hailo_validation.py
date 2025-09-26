#!/usr/bin/env python3
"""Test script to demonstrate Hailo-8 speedup factor validation."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.latency_benchmark import HailoSimulator

def test_validation():
    """Test different speedup factor configurations."""
    
    print("="*70)
    print("HAILO-8 SPEEDUP FACTOR VALIDATION TEST")
    print("="*70)
    
    # Test 1: Default factors (should be valid)
    print("\n1. Testing DEFAULT factors:")
    print("-"*40)
    sim_default = HailoSimulator(device='cpu')
    validation = sim_default.validate_factors()
    print(f"   Base speedup: {sim_default.hailo_speedup}x")
    print(f"   Conv2d optimization: {sim_default.conv2d_optimization}x")
    print(f"   Quantization speedup: {sim_default.quantization_speedup}x")
    total = sim_default.hailo_speedup * sim_default.conv2d_optimization * sim_default.quantization_speedup
    print(f"   Total: {total:.1f}x")
    print(f"   Status: {validation['status']}")
    if validation['warnings']:
        print("   Warnings:", validation['warnings'])
    else:
        print("   ✅ No warnings - factors within reasonable range")
    
    # Test 2: Unrealistic factors (should trigger warnings)
    print("\n2. Testing UNREALISTIC factors:")
    print("-"*40)
    unrealistic = {
        'base_speedup': 100.0,      # Too high
        'conv2d_optimization': 5.0,  # Too high
        'quantization_speedup': 6.0  # Too high
    }
    sim_unrealistic = HailoSimulator(device='cpu', custom_factors=unrealistic)
    validation = sim_unrealistic.validate_factors()
    print(f"   Base speedup: {sim_unrealistic.hailo_speedup}x")
    print(f"   Conv2d optimization: {sim_unrealistic.conv2d_optimization}x")
    print(f"   Quantization speedup: {sim_unrealistic.quantization_speedup}x")
    total = sim_unrealistic.hailo_speedup * sim_unrealistic.conv2d_optimization * sim_unrealistic.quantization_speedup
    print(f"   Total: {total:.1f}x")
    print(f"   Status: {validation['status']}")
    if validation['warnings']:
        print("   ⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"      - {warning}")
    
    # Test 3: Conservative factors (should be valid)
    print("\n3. Testing CONSERVATIVE factors:")
    print("-"*40)
    conservative = {
        'base_speedup': 5.0,
        'conv2d_optimization': 1.2,
        'quantization_speedup': 1.5
    }
    sim_conservative = HailoSimulator(device='cpu', custom_factors=conservative)
    validation = sim_conservative.validate_factors()
    print(f"   Base speedup: {sim_conservative.hailo_speedup}x")
    print(f"   Conv2d optimization: {sim_conservative.conv2d_optimization}x")
    print(f"   Quantization speedup: {sim_conservative.quantization_speedup}x")
    total = sim_conservative.hailo_speedup * sim_conservative.conv2d_optimization * sim_conservative.quantization_speedup
    print(f"   Total: {total:.1f}x")
    print(f"   Status: {validation['status']}")
    if validation['warnings']:
        print("   Warnings:", validation['warnings'])
    else:
        print("   ✅ No warnings - conservative but valid")
    
    # Test 4: Demonstrate calibration file creation
    print("\n4. Creating CALIBRATION file (example):")
    print("-"*40)
    example_measurements = {
        'cpu': {
            'baseline_ms': 450.0,  # Measured on RPi5
            'hailo_ms': 18.0       # Measured with Hailo-8
        }
    }
    
    # Create calibration file
    HailoSimulator.create_calibration_file(
        example_measurements,
        "example_calibration.json"
    )
    
    # Load and use calibrated values
    print("\n   Loading calibrated values...")
    import json
    with open("example_calibration.json", 'r') as f:
        cal_data = json.load(f)
    
    print(f"   Calibrated total speedup: {cal_data['cpu']['total_measured']:.1f}x")
    print(f"   Timestamp: {cal_data['cpu']['timestamp']}")
    
    # Clean up
    import os
    os.remove("example_calibration.json")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("1. Use default factors for initial estimates")
    print("2. Validate with real hardware when available")
    print("3. Document all measurements and sources")
    print("4. Be conservative for production planning")
    print("5. Monitor actual performance in deployment")
    print("\nSee HAILO_CALIBRATION.md for detailed guidance")

if __name__ == "__main__":
    test_validation()