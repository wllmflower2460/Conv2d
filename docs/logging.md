# Structured Logging Documentation

The structured logging system provides comprehensive JSON-based logging with metrics collection, analysis tools, and complete audit trails for production behavioral analysis pipelines.

## Overview

Key features:
- **JSON output**: Structured logs for aggregation and analysis
- **Key event tracking**: Config hashes, seeds, FSQ levels, clustering results  
- **Metrics counters**: Track interpolation, fallbacks, processing stats
- **Forensic value**: Complete audit trail with timestamps and context
- **Analysis tools**: CLI utilities for log parsing and reporting

## Architecture

```
Application Events → Structured Logger → JSON Log Files → Analysis Tools
                           ↓                ↓               ↓
                   Metrics Collector   Log Storage    Reports & Dashboards
```

## Core Components

### Structured Logger

```python
from conv2d.logging.structured import setup_logging, get_logger

# Setup structured logging
logger = setup_logging(name="conv2d", output_file="training.log")

# Log key events
logger.info("Training started", extra={
    "event": "training_start",
    "config_hash": "a1b2c3d4",
    "seed": 42,
    "batch_size": 32,
})
```

### Metrics Collector

```python
from conv2d.logging.structured import MetricsCollector

collector = MetricsCollector()

# Track counters
collector.increment("num_rows_interpolated", 15)
collector.increment("num_rows_all_nan", 3)

# Track timers
with collector.timer("fsq_encoding"):
    result = encode_fsq(x)

# Get metrics summary
summary = collector.get_summary()
```

### Log Analysis Tools

```bash
# Analyze training logs
conv2d-logs summary training.log

# Find errors and warnings
conv2d-logs errors training.log --verbose

# Search for specific events
conv2d-logs grep "clustering" training.log

# Compare experiments
conv2d-logs compare logs/ --output comparison.csv
```

## Log Event Types

### Configuration Events

```python
from conv2d.logging.structured import log_config, log_fsq_config

# Log complete configuration
log_config(config_dict, config_hash="a1b2c3d4")

# Log FSQ-specific configuration
log_fsq_config(
    levels=[8, 6, 5],
    embedding_dim=64,
    codebook_size=240,
)

# Example output:
{
    "timestamp": "2024-01-15T10:30:45.123456",
    "level": "INFO",
    "event": "config",
    "config_hash": "a1b2c3d4",
    "model_name": "conv2d_fsq",
    "fsq_levels": [8, 6, 5],
    "embedding_dim": 64,
    "codebook_size": 240,
    "random_seed": 42
}
```

### Training Events

```python
from conv2d.logging.structured import log_training_epoch

# Log epoch progress
log_training_epoch(
    epoch=15,
    loss=0.123,
    accuracy=0.856,
    lr=0.001,
    duration=47.3,
)

# Example output:
{
    "timestamp": "2024-01-15T10:31:32.789012",
    "level": "INFO", 
    "event": "epoch_complete",
    "epoch": 15,
    "loss": 0.123,
    "accuracy": 0.856,
    "learning_rate": 0.001,
    "duration_seconds": 47.3,
    "samples_per_second": 682.4
}
```

### FSQ Encoding Events

```python
from conv2d.logging.structured import log_fsq_encoding

# Log FSQ encoding results
log_fsq_encoding(
    batch_size=32,
    perplexity=125.6,
    code_usage=0.75,
    duration=0.023,
)

# Example output:
{
    "timestamp": "2024-01-15T10:31:33.234567",
    "level": "INFO",
    "event": "fsq_encoding",
    "batch_size": 32,
    "perplexity": 125.6,
    "code_usage_percent": 75.0,
    "codebook_utilization": "good",
    "encoding_duration_ms": 23.0,
    "codes_per_second": 1391.3
}
```

### Clustering Events

```python
from conv2d.logging.structured import log_clustering

# Log clustering results
log_clustering(
    algorithm="gmm",
    k=4,
    n_samples=1000,
    silhouette_score=0.68,
    merge_operations=2,
)

# Example output:
{
    "timestamp": "2024-01-15T10:31:35.567890",
    "level": "INFO",
    "event": "clustering",
    "algorithm": "gmm",
    "k_clusters": 4,
    "n_samples": 1000,
    "silhouette_score": 0.68,
    "merge_operations": 2,
    "clusters_final": 4,
    "hungarian_matching": true,
    "clustering_duration_ms": 156.7
}
```

### Temporal Smoothing Events

```python
from conv2d.logging.structured import log_temporal_smoothing

# Log temporal policy application
log_temporal_smoothing(
    policy="median_hysteresis", 
    min_dwell=5,
    transitions_before=87,
    transitions_after=34,
    reduction_percent=60.9,
)

# Example output:
{
    "timestamp": "2024-01-15T10:31:36.123456",
    "level": "INFO",
    "event": "temporal_smoothing",
    "policy": "median_hysteresis",
    "min_dwell": 5,
    "window_size": 7,
    "transitions_before": 87,
    "transitions_after": 34,
    "reduction_percent": 60.9,
    "smoothing_duration_ms": 12.4
}
```

### Quality Assurance Events

```python
from conv2d.logging.structured import log_qa_exception, log_qa_summary

# Log quality exceptions
log_qa_exception(
    check="nan_detection",
    severity="warning",
    affected_rows=5,
    total_rows=1000,
    details="NaN values found in features[245:250]"
)

# Log QA summary
log_qa_summary(
    total_samples=1000,
    interpolated=15,
    all_nan=3,
    mean_fallback=7,
    outliers_detected=2,
)

# Example output:
{
    "timestamp": "2024-01-15T10:31:37.456789",
    "level": "WARNING",
    "event": "qa_exception", 
    "check": "nan_detection",
    "severity": "warning",
    "affected_rows": 5,
    "total_rows": 1000,
    "affected_percent": 0.5,
    "details": "NaN values found in features[245:250]",
    "action_taken": "interpolation"
}
```

## Metrics Collection System

### Counter Tracking

```python
class MetricsCollector:
    """Collect and track metrics throughout pipeline execution."""
    
    def __init__(self):
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
    
    def increment(self, name: str, value: int = 1):
        """Increment a counter."""
        self.counters[name] += value
        
        # Log significant increments
        if value > 100 or self.counters[name] % 1000 == 0:
            logger.info(f"Counter update: {name}", extra={
                "event": "counter_update",
                "counter_name": name,
                "increment": value,
                "total": self.counters[name],
            })
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        self.gauges[name] = value
        
        # Log gauge updates
        logger.debug(f"Gauge update: {name}", extra={
            "event": "gauge_update",
            "gauge_name": name,
            "value": value,
        })
    
    def record_histogram(self, name: str, value: float):
        """Record a histogram value."""
        self.histograms[name].append(value)
        
        # Log histogram stats periodically
        if len(self.histograms[name]) % 100 == 0:
            values = self.histograms[name]
            logger.info(f"Histogram stats: {name}", extra={
                "event": "histogram_stats",
                "histogram_name": name,
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            })
```

### Timer Context Manager

```python
@contextmanager
def timer(self, name: str):
    """Time a code block and record duration."""
    start_time = time.time()
    
    logger.debug(f"Timer start: {name}", extra={
        "event": "timer_start",
        "timer_name": name,
    })
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        self.timers[name].append(duration)
        
        logger.info(f"Timer end: {name}", extra={
            "event": "timer_end",
            "timer_name": name,
            "duration_seconds": duration,
            "duration_ms": duration * 1000,
        })

# Usage example
collector = MetricsCollector()

with collector.timer("full_pipeline"):
    with collector.timer("fsq_encoding"):
        result = encode_fsq(x)
    
    with collector.timer("clustering"):
        labels = clusterer.fit_predict(features, k=4)
    
    with collector.timer("temporal_smoothing"):
        smoothed = policy.smooth(labels)
```

### Metrics Summary

```python
def get_summary(self) -> Dict[str, Any]:
    """Get comprehensive metrics summary."""
    
    summary = {
        "counters": dict(self.counters),
        "gauges": dict(self.gauges),
        "timers": {},
        "histograms": {},
    }
    
    # Timer statistics
    for name, times in self.timers.items():
        if times:
            summary["timers"][name] = {
                "count": len(times),
                "total_seconds": sum(times),
                "mean_seconds": np.mean(times),
                "std_seconds": np.std(times),
                "min_seconds": np.min(times),
                "max_seconds": np.max(times),
                "mean_ms": np.mean(times) * 1000,
            }
    
    # Histogram statistics
    for name, values in self.histograms.items():
        if values:
            summary["histograms"][name] = {
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99),
            }
    
    return summary
```

## Log Analysis Tools

### Log Parser

```python
from conv2d.logging.analysis import LogParser

parser = LogParser()

# Parse log file
events = parser.parse_file("training.log")

# Filter by event type
config_events = parser.filter_events(events, event="config")
error_events = parser.filter_events(events, level="ERROR")

# Time range filtering
recent_events = parser.filter_time_range(
    events, 
    start="2024-01-15T10:00:00",
    end="2024-01-15T11:00:00"
)
```

### Metrics Aggregator

```python
from conv2d.logging.analysis import MetricsAggregator

aggregator = MetricsAggregator()

# Aggregate metrics from log events
metrics = aggregator.aggregate_from_events(events)

print(f"Total samples processed: {metrics['counters']['total_samples']}")
print(f"Average FSQ encoding time: {metrics['timers']['fsq_encoding']['mean_ms']:.1f}ms")
print(f"Clustering accuracy: {metrics['gauges']['accuracy']:.1%}")
```

### Report Generator

```python
from conv2d.logging.analysis import LogReporter

reporter = LogReporter()

# Generate summary report
report = reporter.generate_summary("training.log")

# Generate error report
error_report = reporter.generate_error_report("training.log")

# Generate performance report
perf_report = reporter.generate_performance_report("training.log")

# Save reports
reporter.save_report(report, "summary_report.md")
reporter.save_report(error_report, "error_report.md") 
reporter.save_report(perf_report, "performance_report.md")
```

## CLI Tools

### Log Analysis Commands

```bash
# Basic log analysis
conv2d-logs summary training.log
# Output:
# Log Summary for training.log
# =================================
# Time Range: 2024-01-15T10:30:45 to 2024-01-15T12:45:32
# Total Events: 2,847
# Event Types: config(1), epoch_complete(100), fsq_encoding(1000), clustering(50)
# Errors: 2
# Warnings: 15

conv2d-logs errors training.log --verbose
# Output:
# Errors and Warnings in training.log
# ====================================
# [ERROR] 2024-01-15T11:23:45 - FSQ codebook collapse detected
# [ERROR] 2024-01-15T11:45:12 - Clustering failed to converge
# [WARN]  2024-01-15T10:35:22 - NaN values detected in batch 15
# [WARN]  2024-01-15T10:45:33 - Low perplexity: 23.4 (< 50 threshold)

conv2d-logs grep "clustering" training.log
# Output: All log lines containing "clustering"

conv2d-logs metrics training.log --filter timers
# Output:
# Timer Metrics from training.log
# ===============================
# fsq_encoding: 1000 calls, avg 23.4ms, total 23.4s
# clustering: 50 calls, avg 156.7ms, total 7.8s
# temporal_smoothing: 50 calls, avg 12.4ms, total 0.6s
```

### Log Comparison

```bash
# Compare multiple experiments
conv2d-logs compare experiment1.log experiment2.log experiment3.log

# Output comparison table
conv2d-logs compare logs/ --output comparison.csv --format table

# Generate performance comparison
conv2d-logs compare logs/ --metric performance --output perf_comparison.json
```

### Log Monitoring

```bash
# Real-time log monitoring
conv2d-logs tail training.log --follow

# Monitor for errors
conv2d-logs monitor training.log --alert-on ERROR --alert-on "perplexity.*low"

# Generate periodic reports
conv2d-logs watch logs/ --report-interval 300 --output-dir reports/
```

## Configuration

### Logging Configuration

```yaml
# logging_config.yaml
logging:
  version: 1
  formatters:
    json:
      format: '%(asctime)s %(levelname)s %(name)s %(message)s'
      class: conv2d.logging.json_formatter.JSONFormatter
  handlers:
    file:
      class: logging.FileHandler
      filename: training.log
      formatter: json
      level: INFO
    console:
      class: logging.StreamHandler
      formatter: json
      level: WARNING
  loggers:
    conv2d:
      level: INFO
      handlers: [file, console]
      propagate: false
```

### Environment Variables

```bash
# Control log level
export CONV2D_LOG_LEVEL=DEBUG

# Enable/disable structured logging
export CONV2D_STRUCTURED_LOGGING=true

# Log output directory
export CONV2D_LOG_DIR=/path/to/logs

# Metrics collection interval
export CONV2D_METRICS_INTERVAL=60
```

## Integration Examples

### Training Loop Integration

```python
import torch
from conv2d.logging.structured import setup_logging, MetricsCollector, log_training_epoch

# Setup logging
logger = setup_logging("training", "training.log")
collector = MetricsCollector()

logger.info("Training started", extra={
    "event": "training_start",
    "model": "conv2d_fsq",
    "dataset": "quadruped",
    "config_hash": config_hash,
})

for epoch in range(epochs):
    with collector.timer("epoch"):
        # Training step
        with collector.timer("forward_pass"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        with collector.timer("backward_pass"):
            loss.backward()
            optimizer.step()
        
        # Log epoch results
        accuracy = compute_accuracy(outputs, targets)
        log_training_epoch(
            epoch=epoch,
            loss=loss.item(),
            accuracy=accuracy,
            lr=optimizer.param_groups[0]['lr'],
            duration=collector.timers["epoch"][-1],
        )
        
        # Update metrics
        collector.increment("samples_processed", len(inputs))
        collector.set_gauge("current_accuracy", accuracy)

# Log final summary
summary = collector.get_summary()
logger.info("Training completed", extra={
    "event": "training_complete",
    "final_accuracy": accuracy,
    "total_epochs": epochs,
    "metrics_summary": summary,
})
```

### Pipeline Integration

```python
from conv2d.features.fsq_contract import encode_fsq
from conv2d.clustering.gmm import GMMClusterer
from conv2d.temporal.median import MedianHysteresisPolicy
from conv2d.logging.structured import setup_logging, MetricsCollector

# Setup logging
logger = setup_logging("pipeline", "pipeline.log")
collector = MetricsCollector()

def process_batch(x, config):
    """Process a batch with full logging."""
    
    logger.info("Batch processing started", extra={
        "event": "batch_start",
        "batch_size": x.shape[0],
        "input_shape": list(x.shape),
    })
    
    # FSQ encoding
    with collector.timer("fsq_encoding"):
        result = encode_fsq(x, levels=config['fsq']['levels'])
        collector.increment("fsq_encodings")
        collector.set_gauge("perplexity", float(result.perplexity))
        
        if result.perplexity < 50:
            logger.warning("Low perplexity detected", extra={
                "event": "qa_warning",
                "check": "perplexity",
                "value": float(result.perplexity),
                "threshold": 50,
            })
    
    # Clustering
    with collector.timer("clustering"):
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(result.features.numpy(), k=4)
        collector.increment("clusterings")
    
    # Temporal smoothing
    with collector.timer("temporal_smoothing"):
        policy = MedianHysteresisPolicy(min_dwell=3)
        smoothed_labels = policy.smooth(labels.reshape(1, -1))[0]
        collector.increment("temporal_smoothings")
    
    logger.info("Batch processing completed", extra={
        "event": "batch_complete",
        "batch_size": x.shape[0],
        "unique_labels": len(np.unique(smoothed_labels)),
        "processing_time": collector.timers["fsq_encoding"][-1] + 
                          collector.timers["clustering"][-1] + 
                          collector.timers["temporal_smoothing"][-1],
    })
    
    return smoothed_labels
```

### Error Handling Integration

```python
import traceback
from conv2d.logging.structured import get_logger

logger = get_logger("error_handler")

def safe_process_with_logging(func, *args, **kwargs):
    """Execute function with comprehensive error logging."""
    
    try:
        logger.info("Function execution started", extra={
            "event": "function_start",
            "function": func.__name__,
            "args_count": len(args),
            "kwargs_count": len(kwargs),
        })
        
        result = func(*args, **kwargs)
        
        logger.info("Function execution completed", extra={
            "event": "function_complete",
            "function": func.__name__,
            "success": True,
        })
        
        return result
        
    except Exception as e:
        error_info = {
            "event": "function_error",
            "function": func.__name__,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        
        logger.error(f"Function {func.__name__} failed", extra=error_info)
        
        # Re-raise with context
        raise RuntimeError(f"Function {func.__name__} failed: {e}") from e
```

## Best Practices

1. **Use structured events**: Always include `event` field for easy filtering and analysis
2. **Include context**: Add relevant metadata like batch_size, config_hash, model_name
3. **Monitor key metrics**: Track perplexity, code usage, transition rates, processing times
4. **Log QA exceptions**: Document all data quality issues with severity and actions taken
5. **Use appropriate levels**: INFO for normal events, WARNING for quality issues, ERROR for failures
6. **Include performance data**: Log timing and throughput metrics for optimization
7. **Maintain audit trails**: Log configuration changes, model updates, and parameter changes
8. **Enable log analysis**: Use CLI tools regularly to monitor training and catch issues early

This structured logging system provides comprehensive observability for production behavioral analysis pipelines with complete forensic capabilities and automated analysis tools.