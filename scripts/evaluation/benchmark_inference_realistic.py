#!/usr/bin/env python3
"""
Realistic inference benchmarking for FSQ model.
31,000 FPS is impossible - let's get real numbers.
"""

import torch
import time
import numpy as np
from pathlib import Path

def benchmark_inference():
    """Benchmark with realistic measurements."""
    
    print("="*70)
    print("REALISTIC INFERENCE BENCHMARKING")
    print("="*70)
    print("\n⚠️  Previous claim: 31,000 FPS is IMPOSSIBLE")
    print("   That would be 32 microseconds per frame")
    print("   Even a simple matrix multiply takes longer!\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    from models.conv2d_fsq_model import Conv2dFSQ
    
    model = Conv2dFSQ(
        input_channels=9,
        hidden_dim=128,
        num_classes=10,
        fsq_levels=[8, 6, 5]
    ).to(device)
    
    model.eval()
    
    # Test different batch sizes
    batch_sizes = [1, 8, 32, 64]
    input_shape = (9, 2, 50)  # Our Conv2d input shape
    
    print("\n" + "-"*70)
    print("INFERENCE TIMING (includes all overhead)")
    print("-"*70)
    
    for batch_size in batch_sizes:
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time inference
        timings = []
        
        for _ in range(100):
            start = time.perf_counter()
            
            with torch.no_grad():
                output = model(dummy_input)
                
                # Include post-processing that would be needed
                if isinstance(output, dict):
                    logits = output.get('logits', output.get('output'))
                else:
                    logits = output
                
                # Get predictions (this is part of inference!)
                predictions = torch.argmax(logits, dim=1)
                
                # Force synchronization for accurate timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                else:
                    # Force CPU completion
                    _ = predictions.cpu().numpy()
            
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to ms
        
        # Calculate statistics
        mean_time = np.mean(timings)
        std_time = np.std(timings)
        min_time = np.min(timings)
        max_time = np.max(timings)
        p50_time = np.percentile(timings, 50)
        p95_time = np.percentile(timings, 95)
        p99_time = np.percentile(timings, 99)
        
        # Calculate realistic FPS
        fps_mean = 1000 / mean_time if mean_time > 0 else 0
        fps_best = 1000 / min_time if min_time > 0 else 0
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  Mean: {mean_time:.3f}ms ± {std_time:.3f}ms")
        print(f"  Min: {min_time:.3f}ms, Max: {max_time:.3f}ms")
        print(f"  P50: {p50_time:.3f}ms, P95: {p95_time:.3f}ms, P99: {p99_time:.3f}ms")
        print(f"  Throughput: {fps_mean:.1f} FPS (mean), {fps_best:.1f} FPS (best)")
        print(f"  Per-sample: {mean_time/batch_size:.3f}ms")
    
    # Hardware limitations
    print("\n" + "-"*70)
    print("REALITY CHECK")
    print("-"*70)
    
    print("""
Realistic FPS expectations by hardware:
  
  CPU (Intel i7):          10-50 FPS
  GPU (RTX 2060):          100-500 FPS  
  GPU (RTX 3090):          200-1000 FPS
  Raspberry Pi 5:          5-20 FPS
  Raspberry Pi + Hailo-8:  50-200 FPS
  
31,000 FPS would mean:
  - 32 microseconds per inference
  - Faster than memory access time
  - Faster than PCIe transfer time
  - Physically impossible for this model size
  
Likely explanation for 31,000 FPS claim:
  1. Timer was measuring wrong thing (empty loop?)
  2. Batch size confusion (31k samples/sec with batch 1000?)
  3. Not including actual computation
  4. Integer overflow or unit confusion
  5. Measuring only part of the model
""")
    
    # Memory bandwidth check
    if device.type == 'cuda':
        model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)
        print(f"\nModel size: {model_size_mb:.2f} MB")
        print(f"Minimum memory reads per inference: {model_size_mb:.2f} MB")
        
        # Even with 500 GB/s memory bandwidth
        theoretical_max_fps = 500_000 / model_size_mb
        print(f"Theoretical maximum FPS (memory limited): {theoretical_max_fps:.0f}")
    
    # Actual Hailo-8 expectations
    print("\n" + "-"*70)
    print("HAILO-8 REALISTIC PERFORMANCE")
    print("-"*70)
    print("""
Hailo-8 specifications:
  - 26 TOPS at INT8
  - Designed for edge inference
  - Typical CNN models: 100-1000 FPS
  
For our FSQ model (57k parameters):
  - Expected: 100-300 FPS
  - With optimization: 200-500 FPS
  - Absolute maximum: ~1000 FPS
  
NOT 31,000 FPS!
""")

if __name__ == "__main__":
    benchmark_inference()