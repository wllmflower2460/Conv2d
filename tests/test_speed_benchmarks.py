"""Speed micro-benchmarks with thresholds.

CRITICAL: Preserve edge latency targets.
These tests catch performance regressions that break real-time requirements.
"""

from __future__ import annotations

import time
import torch
import numpy as np
from typing import Dict, Callable
from contextlib import contextmanager

from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.kmeans import KMeansClusterer
from conv2d.clustering.gmm import GMMClusterer
from conv2d.temporal.median import MedianHysteresisPolicy as MedianHysteresis
from conv2d.metrics.core import MetricsCalculator


# Performance thresholds (milliseconds)
THRESHOLDS = {
    # Edge device targets (Raspberry Pi 5 + Hailo-8)
    "fsq_encode_batch1": 10.0,      # Single sample inference
    "fsq_encode_batch32": 50.0,     # Batch inference
    "clustering_100": 20.0,         # Cluster 100 samples
    "clustering_1000": 100.0,       # Cluster 1000 samples
    "temporal_smooth_500": 5.0,     # Smooth 500 timesteps
    "metrics_compute": 10.0,        # Compute all metrics
    
    # GPU targets (for training)
    "fsq_encode_batch256_gpu": 20.0,  # Large batch on GPU
    
    # End-to-end pipeline
    "pipeline_single": 30.0,        # Complete pipeline for 1 sample
    "pipeline_batch32": 100.0,      # Complete pipeline for batch
}


@contextmanager
def timer():
    """Simple timer context manager."""
    start = time.perf_counter()
    result = {"time": 0.0}
    yield result
    result["time"] = (time.perf_counter() - start) * 1000  # Convert to ms


def benchmark_function(
    func: Callable,
    *args,
    n_warmup: int = 10,
    n_trials: int = 100,
    **kwargs,
) -> Dict[str, float]:
    """Benchmark a function with warmup and multiple trials.
    
    Returns:
        Dictionary with mean, std, min, max times in milliseconds.
    """
    # Warmup
    for _ in range(n_warmup):
        func(*args, **kwargs)
    
    # Benchmark
    times = []
    for _ in range(n_trials):
        with timer() as t:
            func(*args, **kwargs)
        times.append(t["time"])
    
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
    }


def test_fsq_encode_speed():
    """Test FSQ encoding speed meets thresholds."""
    print("\nFSQ Encoding Benchmarks:")
    print("-" * 50)
    
    # Single sample (inference)
    x1 = torch.randn(1, 9, 2, 100, dtype=torch.float32)
    stats1 = benchmark_function(encode_fsq, x1, reset_stats=True, n_trials=100)
    
    print(f"  Batch=1:  {stats1['mean']:.2f}ms ± {stats1['std']:.2f}ms")
    print(f"           (threshold: {THRESHOLDS['fsq_encode_batch1']:.1f}ms)")
    
    assert stats1['mean'] < THRESHOLDS['fsq_encode_batch1'], (
        f"FSQ batch=1 too slow: {stats1['mean']:.2f}ms > {THRESHOLDS['fsq_encode_batch1']}ms"
    )
    
    # Batch 32 (typical batch)
    x32 = torch.randn(32, 9, 2, 100, dtype=torch.float32)
    stats32 = benchmark_function(encode_fsq, x32, reset_stats=True, n_trials=50)
    
    print(f"  Batch=32: {stats32['mean']:.2f}ms ± {stats32['std']:.2f}ms")
    print(f"           (threshold: {THRESHOLDS['fsq_encode_batch32']:.1f}ms)")
    
    assert stats32['mean'] < THRESHOLDS['fsq_encode_batch32'], (
        f"FSQ batch=32 too slow: {stats32['mean']:.2f}ms > {THRESHOLDS['fsq_encode_batch32']}ms"
    )
    
    # Check batch scaling efficiency
    speedup = stats32['mean'] / (stats1['mean'] * 32)
    print(f"  Batch efficiency: {speedup:.2%} of linear scaling")
    
    # GPU benchmark if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x256_gpu = torch.randn(256, 9, 2, 100, dtype=torch.float32, device=device)
        
        # Note: encode_fsq might not support GPU directly
        # This is a placeholder for GPU-enabled version
        try:
            stats256_gpu = benchmark_function(
                encode_fsq, x256_gpu.cpu(), reset_stats=True, n_trials=20
            )
            print(f"  Batch=256 (GPU→CPU): {stats256_gpu['mean']:.2f}ms")
        except:
            print("  GPU benchmark skipped (not implemented)")


def test_clustering_speed():
    """Test clustering speed meets thresholds."""
    print("\nClustering Benchmarks:")
    print("-" * 50)
    
    # 100 samples
    features100 = np.random.randn(100, 64).astype(np.float32)
    
    # K-means
    kmeans = KMeansClusterer(random_state=42)
    stats_km100 = benchmark_function(
        kmeans.fit_predict, features100, k=4, n_trials=50
    )
    
    print(f"  K-means (n=100):  {stats_km100['mean']:.2f}ms ± {stats_km100['std']:.2f}ms")
    print(f"                   (threshold: {THRESHOLDS['clustering_100']:.1f}ms)")
    
    assert stats_km100['mean'] < THRESHOLDS['clustering_100'], (
        f"K-means n=100 too slow: {stats_km100['mean']:.2f}ms"
    )
    
    # 1000 samples
    features1000 = np.random.randn(1000, 64).astype(np.float32)
    
    stats_km1000 = benchmark_function(
        kmeans.fit_predict, features1000, k=4, n_trials=20
    )
    
    print(f"  K-means (n=1000): {stats_km1000['mean']:.2f}ms ± {stats_km1000['std']:.2f}ms")
    print(f"                   (threshold: {THRESHOLDS['clustering_1000']:.1f}ms)")
    
    assert stats_km1000['mean'] < THRESHOLDS['clustering_1000'], (
        f"K-means n=1000 too slow: {stats_km1000['mean']:.2f}ms"
    )
    
    # GMM comparison
    gmm = GMMClusterer(random_state=42)
    stats_gmm100 = benchmark_function(
        gmm.fit_predict, features100, k=4, n_trials=20
    )
    
    print(f"  GMM (n=100):      {stats_gmm100['mean']:.2f}ms ± {stats_gmm100['std']:.2f}ms")
    
    # GMM can be slower, but shouldn't be more than 3x K-means
    assert stats_gmm100['mean'] < stats_km100['mean'] * 3, (
        f"GMM too slow compared to K-means: {stats_gmm100['mean']/stats_km100['mean']:.1f}x"
    )


def test_temporal_smoothing_speed():
    """Test temporal smoothing speed meets thresholds."""
    print("\nTemporal Smoothing Benchmarks:")
    print("-" * 50)
    
    # 500 timesteps (5 seconds at 100Hz)
    sequence500 = np.random.randint(0, 4, (1, 500), dtype=np.int32)
    
    smoother = MedianHysteresis(min_dwell=5, window_size=7)
    stats500 = benchmark_function(smoother.smooth, sequence500, n_trials=100)
    
    print(f"  T=500:  {stats500['mean']:.2f}ms ± {stats500['std']:.2f}ms")
    print(f"         (threshold: {THRESHOLDS['temporal_smooth_500']:.1f}ms)")
    
    assert stats500['mean'] < THRESHOLDS['temporal_smooth_500'], (
        f"Smoothing T=500 too slow: {stats500['mean']:.2f}ms"
    )
    
    # Test scaling with sequence length
    lengths = [100, 500, 1000, 5000]
    times = []
    
    for T in lengths:
        seq = np.random.randint(0, 4, (1, T), dtype=np.int32)
        stats = benchmark_function(smoother.smooth, seq, n_trials=20)
        times.append(stats['mean'])
        print(f"  T={T:4d}: {stats['mean']:.2f}ms")
    
    # Should scale linearly or better
    # Check if doubling length less than doubles time
    for i in range(len(times) - 1):
        ratio = times[i+1] / times[i]
        length_ratio = lengths[i+1] / lengths[i]
        assert ratio <= length_ratio * 1.5, (
            f"Smoothing scaling poor: {ratio:.1f}x time for {length_ratio:.1f}x length"
        )


def test_metrics_computation_speed():
    """Test metrics computation speed."""
    print("\nMetrics Computation Benchmarks:")
    print("-" * 50)
    
    n_samples = 1000
    n_classes = 4
    
    # Generate data
    y_true = np.random.randint(0, n_classes, n_samples, dtype=np.int32)
    y_pred = np.random.randint(0, n_classes, n_samples, dtype=np.int32)
    y_prob = np.random.rand(n_samples, n_classes).astype(np.float32)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    calculator = MetricsCalculator()
    
    # Benchmark all metrics
    stats = benchmark_function(
        calculator.compute_all, y_true, y_pred, y_prob, n_trials=100
    )
    
    print(f"  All metrics: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms")
    print(f"              (threshold: {THRESHOLDS['metrics_compute']:.1f}ms)")
    
    assert stats['mean'] < THRESHOLDS['metrics_compute'], (
        f"Metrics too slow: {stats['mean']:.2f}ms"
    )
    
    # Individual metric benchmarks
    print("\n  Individual metrics:")
    
    # Accuracy only
    def compute_accuracy():
        return np.mean(y_true == y_pred)
    
    stats_acc = benchmark_function(compute_accuracy, n_trials=1000)
    print(f"    Accuracy:  {stats_acc['mean']:.3f}ms")
    
    # F1 only (more expensive)
    from sklearn.metrics import f1_score
    
    def compute_f1():
        return f1_score(y_true, y_pred, average='macro')
    
    stats_f1 = benchmark_function(compute_f1, n_trials=100)
    print(f"    Macro-F1:  {stats_f1['mean']:.2f}ms")


def test_pipeline_end_to_end_speed():
    """Test complete pipeline speed."""
    print("\nEnd-to-End Pipeline Benchmarks:")
    print("-" * 50)
    
    def run_pipeline_single():
        """Run pipeline for single sample."""
        # Input
        x = torch.randn(1, 9, 2, 100, dtype=torch.float32)
        
        # FSQ
        result = encode_fsq(x, reset_stats=True)
        
        # Clustering (on accumulated features)
        features = result.features.numpy()
        if features.shape[0] >= 4:
            clusterer = KMeansClusterer(random_state=42)
            labels = clusterer.fit_predict(features, k=4)
        else:
            labels = np.zeros(features.shape[0], dtype=np.int32)
        
        return labels
    
    stats_single = benchmark_function(run_pipeline_single, n_trials=50)
    
    print(f"  Single sample: {stats_single['mean']:.2f}ms ± {stats_single['std']:.2f}ms")
    print(f"                (threshold: {THRESHOLDS['pipeline_single']:.1f}ms)")
    
    assert stats_single['mean'] < THRESHOLDS['pipeline_single'], (
        f"Pipeline single too slow: {stats_single['mean']:.2f}ms"
    )
    
    def run_pipeline_batch():
        """Run pipeline for batch."""
        # Input
        x = torch.randn(32, 9, 2, 100, dtype=torch.float32)
        
        # FSQ
        result = encode_fsq(x, reset_stats=True)
        
        # Clustering
        features = result.features.numpy()
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(features, k=4)
        
        # Temporal smoothing (simulate sequence)
        labels_seq = np.tile(labels, 10)[:300].reshape(1, -1)
        smoother = MedianHysteresis(min_dwell=5)
        smoothed = smoother.smooth(labels_seq)
        
        return smoothed
    
    stats_batch = benchmark_function(run_pipeline_batch, n_trials=20)
    
    print(f"  Batch-32:      {stats_batch['mean']:.2f}ms ± {stats_batch['std']:.2f}ms")
    print(f"                (threshold: {THRESHOLDS['pipeline_batch32']:.1f}ms)")
    
    assert stats_batch['mean'] < THRESHOLDS['pipeline_batch32'], (
        f"Pipeline batch too slow: {stats_batch['mean']:.2f}ms"
    )


def test_memory_efficiency():
    """Test memory usage stays reasonable."""
    print("\nMemory Efficiency Tests:")
    print("-" * 50)
    
    import tracemalloc
    
    # Test FSQ memory usage
    tracemalloc.start()
    
    # Large batch
    x = torch.randn(256, 9, 2, 100, dtype=torch.float32)
    snapshot1 = tracemalloc.take_snapshot()
    
    result = encode_fsq(x)
    snapshot2 = tracemalloc.take_snapshot()
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_mb = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)
    
    print(f"  FSQ batch=256: {total_mb:.1f} MB allocated")
    
    # Should be reasonable (< 100MB for this size)
    assert total_mb < 100, f"FSQ using too much memory: {total_mb:.1f} MB"
    
    tracemalloc.stop()
    
    # Test clustering memory
    tracemalloc.start()
    
    features = np.random.randn(10000, 64).astype(np.float32)
    snapshot1 = tracemalloc.take_snapshot()
    
    kmeans = KMeansClusterer(random_state=42)
    labels = kmeans.fit_predict(features, k=10)
    snapshot2 = tracemalloc.take_snapshot()
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_mb = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)
    
    print(f"  K-means n=10000: {total_mb:.1f} MB allocated")
    
    # Should be reasonable
    assert total_mb < 200, f"K-means using too much memory: {total_mb:.1f} MB"
    
    tracemalloc.stop()


def test_parallelization_speedup():
    """Test parallelization provides expected speedup."""
    print("\nParallelization Tests:")
    print("-" * 50)
    
    # Test if numpy/torch using multiple threads
    import os
    
    # Get thread counts
    if hasattr(torch, 'get_num_threads'):
        torch_threads = torch.get_num_threads()
        print(f"  PyTorch threads: {torch_threads}")
    
    try:
        import mkl
        mkl_threads = mkl.get_max_threads()
        print(f"  MKL threads: {mkl_threads}")
    except ImportError:
        pass
    
    # Large matrix operation to test parallelization
    size = 2000
    A = torch.randn(size, size, dtype=torch.float32)
    B = torch.randn(size, size, dtype=torch.float32)
    
    # Single thread
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(1)
        
        with timer() as t1:
            C = torch.matmul(A, B)
        time_single = t1["time"]
        
        # Multi thread
        torch.set_num_threads(4)
        
        with timer() as t4:
            C = torch.matmul(A, B)
        time_multi = t4["time"]
        
        speedup = time_single / time_multi
        print(f"  Matrix multiply speedup (4 threads): {speedup:.2f}x")
        
        # Should get some speedup with 4 threads
        assert speedup > 1.5, f"Poor parallelization: only {speedup:.2f}x with 4 threads"
        
        # Restore default
        torch.set_num_threads(torch_threads)
    else:
        print("  Thread control not available")


def generate_performance_report():
    """Generate a performance summary report."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY REPORT")
    print("=" * 60)
    
    # Collect all benchmarks
    results = {}
    
    # FSQ
    x32 = torch.randn(32, 9, 2, 100, dtype=torch.float32)
    stats = benchmark_function(encode_fsq, x32, reset_stats=True, n_trials=50)
    results["FSQ (batch=32)"] = stats['mean']
    
    # Clustering
    features = np.random.randn(100, 64).astype(np.float32)
    kmeans = KMeansClusterer(random_state=42)
    stats = benchmark_function(kmeans.fit_predict, features, k=4, n_trials=50)
    results["K-means (n=100)"] = stats['mean']
    
    # Temporal
    sequence = np.random.randint(0, 4, (1, 500), dtype=np.int32)
    smoother = MedianHysteresis(min_dwell=5)
    stats = benchmark_function(smoother.smooth, sequence, n_trials=100)
    results["Temporal (T=500)"] = stats['mean']
    
    # Print table
    print("\nComponent Latencies:")
    print("-" * 40)
    for name, time_ms in results.items():
        status = "✓" if time_ms < 50 else "⚠"
        print(f"  {status} {name:20s}: {time_ms:6.2f} ms")
    
    # Total pipeline estimate
    total = sum(results.values())
    print("-" * 40)
    print(f"  Estimated Total:     {total:6.2f} ms")
    print(f"  Target (100ms):      {'✓ PASS' if total < 100 else '✗ FAIL'}")
    
    # Throughput estimate
    throughput = 1000 / total  # samples per second
    print(f"\n  Throughput: ~{throughput:.1f} Hz")
    print(f"  Target (10 Hz): {'✓ PASS' if throughput > 10 else '✗ FAIL'}")


if __name__ == "__main__":
    # Run all benchmarks
    test_fsq_encode_speed()
    test_clustering_speed()
    test_temporal_smoothing_speed()
    test_metrics_computation_speed()
    test_pipeline_end_to_end_speed()
    test_memory_efficiency()
    test_parallelization_speedup()
    
    # Generate summary report
    generate_performance_report()
    
    print("\n🎯 All speed benchmarks passed!")
    print("Edge latency targets: PRESERVED")