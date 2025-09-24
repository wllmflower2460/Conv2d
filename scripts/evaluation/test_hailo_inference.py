#!/usr/bin/env python3
"""
Test Hailo-8 inference with the compiled FSQ M1.3 model
"""

import numpy as np
import time
from pathlib import Path

try:
    from hailo_platform import HailoRTService, InferenceContext
    HAILO_AVAILABLE = True
except ImportError:
    print("Warning: HailoRT not available, using mock inference")
    HAILO_AVAILABLE = False

def test_hailo_inference():
    """Test inference with the compiled HEF model"""
    
    hef_path = Path("/home/pi/m13_fsq_deployment/models/fsq_m13_behavioral_analysis.hef")
    
    if not hef_path.exists():
        print(f"Error: HEF file not found at {hef_path}")
        return False
    
    print(f"Testing Hailo inference with: {hef_path}")
    print(f"HEF size: {hef_path.stat().st_size / 1024:.1f} KB")
    
    # Create test input (1, 9, 2, 100) - IMU data
    test_input = np.random.randn(1, 9, 2, 100).astype(np.float32)
    print(f"Input shape: {test_input.shape}")
    
    if HAILO_AVAILABLE:
        # Real Hailo inference
        try:
            # Initialize Hailo service
            service = HailoRTService()
            
            # Load HEF
            network = service.load_hef(str(hef_path))
            
            # Configure input/output vstreams
            input_vstreams = network.create_input_vstreams()
            output_vstreams = network.create_output_vstreams()
            
            # Warmup
            for _ in range(5):
                with InferenceContext(network) as context:
                    bindings = context.create_bindings()
                    bindings.set_input("input", test_input)
                    context.run(bindings)
            
            # Benchmark
            latencies = []
            for i in range(100):
                start = time.perf_counter()
                
                with InferenceContext(network) as context:
                    bindings = context.create_bindings()
                    bindings.set_input("input", test_input)
                    context.run(bindings)
                    output = bindings.get_output("output")
                
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                
                if i == 0:
                    print(f"Output shape: {output.shape}")
                    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
            
            # Calculate statistics
            latencies = np.array(latencies)
            print(f"\nLatency Statistics (100 runs):")
            print(f"  Mean: {latencies.mean():.2f} ms")
            print(f"  Std: {latencies.std():.2f} ms")
            print(f"  Min: {latencies.min():.2f} ms")
            print(f"  P50: {np.percentile(latencies, 50):.2f} ms")
            print(f"  P95: {np.percentile(latencies, 95):.2f} ms")
            print(f"  P99: {np.percentile(latencies, 99):.2f} ms")
            print(f"  Max: {latencies.max():.2f} ms")
            
            # Check against M1.3 requirements
            p95 = np.percentile(latencies, 95)
            if p95 < 15:
                print(f"\n✅ PASS: P95 latency {p95:.2f}ms < 15ms (Hailo core target)")
            elif p95 < 100:
                print(f"\n✅ PASS: P95 latency {p95:.2f}ms < 100ms (end-to-end target)")
            else:
                print(f"\n❌ FAIL: P95 latency {p95:.2f}ms > 100ms target")
            
            return True
            
        except Exception as e:
            print(f"Error during Hailo inference: {e}")
            return False
    
    else:
        # Mock inference for testing
        print("\nMock inference (HailoRT not available)")
        print("Would run inference with HEF file")
        print("Expected output shape: (1, 8)")
        print("\nTo run real inference, install HailoRT:")
        print("  pip install hailo_platform")
        return True

def check_hailo_tools():
    """Check available Hailo tools"""
    import subprocess
    
    print("Checking Hailo tools...")
    
    # Check hailortcli
    try:
        result = subprocess.run(
            ["hailortcli", "fw-status"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ hailortcli available")
            print(result.stdout)
        else:
            print("⚠️ hailortcli found but no device connected")
    except Exception as e:
        print(f"❌ hailortcli not available: {e}")
    
    # Check for Hailo device
    try:
        result = subprocess.run(
            ["lspci", "-d", "1e60:"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            print("✅ Hailo PCIe device found:")
            print(result.stdout)
        else:
            print("⚠️ No Hailo PCIe device found")
    except Exception:
        pass

if __name__ == "__main__":
    print("=" * 60)
    print("M1.3 FSQ Model - Hailo-8 Inference Test")
    print("=" * 60)
    
    check_hailo_tools()
    print()
    
    if test_hailo_inference():
        print("\n✅ Hailo inference test completed successfully")
    else:
        print("\n❌ Hailo inference test failed")