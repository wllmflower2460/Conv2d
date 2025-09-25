#!/usr/bin/env python3
"""
Benchmark script for Conv2d-VQ-HDP-HSMM model
- Latency testing on CPU and GPU
- ONNX export for Hailo compilation
- Memory profiling and optimization verification
"""

import torch
import torch.nn as nn
import time
import numpy as np
import argparse
from pathlib import Path
import json
import yaml
import warnings
warnings.filterwarnings('ignore')

# Model imports
from models.conv2d_vq_hdp_hsmm import Conv2dVQHDPHSMM
from models.conv2d_vq_model import Conv2dVQModel


def measure_latency(model, input_tensor, num_warmup=10, num_runs=100):
    """Measure model inference latency."""
    device = next(model.parameters()).device
    
    # Warmup runs
    print(f"Warming up with {num_warmup} runs...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # Synchronize for GPU timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    print(f"Measuring latency over {num_runs} runs...")
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return latencies


def profile_memory(model, input_tensor):
    """Profile memory usage."""
    device = next(model.parameters()).device
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
        
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
        
        return {
            'allocated_mb': memory_allocated,
            'reserved_mb': memory_reserved
        }
    else:
        # CPU memory profiling (approximate)
        import tracemalloc
        tracemalloc.start()
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'current_mb': current / 1024**2,
            'peak_mb': peak / 1024**2
        }


def export_onnx(model, input_shape, output_path, opset_version=11, simplify=True):
    """Export model to ONNX format for Hailo compilation."""
    print(f"\nExporting model to ONNX...")
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    model.eval()
    
    # Move to CPU for export
    model_cpu = model.cpu()
    dummy_input = dummy_input.cpu()
    
    # Export to ONNX
    torch.onnx.export(
        model_cpu,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output', 'indices', 'clusters', 'states'],
        dynamic_axes=None,  # Static shapes for Hailo
        verbose=False
    )
    
    print(f"✓ ONNX exported to: {output_path}")
    
    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify
            
            print("Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            simplified_model, check = simplify(onnx_model)
            
            if check:
                onnx.save(simplified_model, output_path)
                print("✓ ONNX model simplified")
            else:
                print("⚠ Simplification check failed, keeping original")
        except ImportError:
            print("ℹ onnx-simplifier not installed, skipping simplification")
    
    # Verify ONNX model
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
        
        # Test inference
        ort_session = ort.InferenceSession(output_path)
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: dummy_input.numpy()})
        print(f"✓ ONNX inference test passed, output shape: {outputs[0].shape}")
        
    except ImportError:
        print("ℹ onnxruntime not installed, skipping verification")
    except Exception as e:
        print(f"⚠ ONNX verification failed: {e}")
    
    return output_path


def run_benchmark(args):
    """Run complete benchmark suite."""
    print("=" * 60)
    print("Conv2d-VQ-HDP-HSMM Benchmark Suite")
    print("=" * 60)
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            print(f"✓ Loaded config from: {args.config}")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Model selection
    if args.model == 'full':
        model = Conv2dVQHDPHSMM(
            input_channels=9,
            input_height=2,
            num_codes=config.get('vq', {}).get('num_codes', 512),
            code_dim=config.get('vq', {}).get('code_dim', 64),
            max_clusters=config.get('hdp', {}).get('max_clusters', 20),
            num_states=config.get('hsmm', {}).get('num_states', 10)
        )
        model_name = "Conv2d-VQ-HDP-HSMM (Full)"
    else:  # vq_only
        model = Conv2dVQModel(
            input_channels=9,
            input_height=2,
            num_codes=config.get('vq', {}).get('num_codes', 512),
            code_dim=config.get('vq', {}).get('code_dim', 64)
        )
        model_name = "Conv2d-VQ (VQ Only)"
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    # Test input
    input_shape = (args.batch_size, 9, 2, 100)  # (B, C, H, T)
    input_tensor = torch.randn(*input_shape).to(device)
    print(f"\nInput shape: {input_shape}")
    
    # Measure latency
    print("\n" + "="*40)
    print("LATENCY BENCHMARKS")
    print("="*40)
    
    latencies = measure_latency(model, input_tensor, args.warmup, args.runs)
    
    # Calculate statistics
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"\nLatency Statistics (ms):")
    print(f"  Mean:    {mean_latency:.3f} ± {std_latency:.3f}")
    print(f"  Min:     {min_latency:.3f}")
    print(f"  Max:     {max_latency:.3f}")
    print(f"  P50:     {p50_latency:.3f}")
    print(f"  P95:     {p95_latency:.3f}")
    print(f"  P99:     {p99_latency:.3f}")
    print(f"  FPS:     {1000/mean_latency:.1f}")
    
    # Memory profiling
    print("\n" + "="*40)
    print("MEMORY PROFILING")
    print("="*40)
    
    memory_info = profile_memory(model, input_tensor)
    for key, value in memory_info.items():
        print(f"  {key}: {value:.2f} MB")
    
    # Test different batch sizes
    if args.test_batch_sizes:
        print("\n" + "="*40)
        print("BATCH SIZE SCALING")
        print("="*40)
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        for bs in batch_sizes:
            if bs > args.batch_size:
                break
            
            test_input = torch.randn(bs, 9, 2, 100).to(device)
            latencies = measure_latency(model, test_input, num_warmup=5, num_runs=20)
            mean_lat = np.mean(latencies)
            print(f"  Batch {bs:2d}: {mean_lat:7.3f} ms ({mean_lat/bs:7.3f} ms/sample)")
    
    # Export ONNX if requested
    if args.export_onnx:
        print("\n" + "="*40)
        print("ONNX EXPORT")
        print("="*40)
        
        onnx_path = Path(args.onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use batch size 1 for edge deployment
        export_shape = (1, 9, 2, 100)
        export_onnx(model, export_shape, str(onnx_path), 
                   opset_version=args.onnx_opset, 
                   simplify=args.onnx_simplify)
    
    # Save results
    if args.save_results:
        results = {
            'model': model_name,
            'device': str(device),
            'parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'size_mb': total_params * 4 / 1024**2
            },
            'input_shape': list(input_shape),
            'latency_ms': {
                'mean': mean_latency,
                'std': std_latency,
                'min': min_latency,
                'max': max_latency,
                'p50': p50_latency,
                'p95': p95_latency,
                'p99': p99_latency
            },
            'throughput_fps': 1000/mean_latency,
            'memory': memory_info
        }
        
        results_path = Path(args.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Conv2d-VQ-HDP-HSMM model')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='full', 
                       choices=['full', 'vq_only'],
                       help='Model variant to benchmark')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to model configuration YAML')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to run benchmark on')
    
    # Benchmark settings
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for benchmarking')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of benchmark runs')
    parser.add_argument('--test-batch-sizes', action='store_true',
                       help='Test multiple batch sizes')
    
    # ONNX export
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export model to ONNX format')
    parser.add_argument('--onnx-path', type=str, 
                       default='models/conv2d_vq_hdp_hsmm.onnx',
                       help='Path for ONNX export')
    parser.add_argument('--onnx-opset', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--onnx-simplify', action='store_true', default=True,
                       help='Simplify ONNX model')
    
    # Output
    parser.add_argument('--save-results', action='store_true',
                       help='Save benchmark results to JSON')
    parser.add_argument('--results-path', type=str, 
                       default='benchmark_results.json',
                       help='Path for results JSON')
    
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    main()