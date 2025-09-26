#!/usr/bin/env python3
"""Demonstration of structured logging and analysis system.

Shows how to:
1. Use structured logging with JSON output
2. Track key events and metrics
3. Parse and analyze log files
4. Generate reports from logs
5. Search and filter log data
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import structured logging (with structlog)
try:
    from conv2d.logging.structured import (
        setup_logging,
        get_logger,
        log_config,
        log_seeds,
        log_fsq_config,
        log_clustering,
        log_qa_issue,
        increment_counter,
        record_event,
        get_metrics,
    )
    HAS_STRUCTLOG = True
except ImportError:
    # Fall back to JSON formatter
    HAS_STRUCTLOG = False
    from conv2d.logging.json_formatter import setup_json_logging as setup_logging
    import logging
    
    # Create mock functions for demo
    class MockLogger:
        def __init__(self, logger):
            self.logger = logger
            
        def log_config(self, config, config_hash):
            self.logger.info("configuration_loaded", extra={
                "event_type": "CONFIG",
                "config_hash": config_hash,
                "config_keys": list(config.keys())
            })
            
        def log_seeds(self, **seeds):
            self.logger.info("random_seeds_set", extra={
                "event_type": "SEEDS",
                "seeds": seeds
            })
            
        def log_fsq_config(self, levels, embedding_dim, codebook_size):
            self.logger.info("fsq_configuration", extra={
                "event_type": "FSQ_CONFIG",
                "levels": levels,
                "embedding_dim": embedding_dim,
                "codebook_size": codebook_size
            })
            
        def log_clustering(self, algorithm, k_selected, metric_used, metric_value):
            self.logger.info("clustering_complete", extra={
                "event_type": "CLUSTERING",
                "algorithm": algorithm,
                "k_selected": k_selected,
                "metric_used": metric_used,
                "metric_value": metric_value
            })
            
        def log_qa_issue(self, issue_type, count, details=None):
            self.logger.warning("qa_issue_detected", extra={
                "event_type": "QA_ISSUE",
                "issue_type": issue_type,
                "count": count,
                "details": details or {}
            })
            
        def log_data_processing(self, **kwargs):
            self.logger.info("data_processing_complete", extra={
                "event_type": "DATA_PROCESSING",
                **kwargs
            })
            
        def log_temporal_smoothing(self, **kwargs):
            self.logger.info("temporal_smoothing_applied", extra={
                "event_type": "SMOOTHING",
                **kwargs
            })
            
        def log_exception(self, exception, context=None):
            self.logger.error("exception_occurred", extra={
                "event_type": "EXCEPTION",
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "context": context or {}
            })
            
        def log_merges(self, merge_table):
            self.logger.info("clusters_merged", extra={
                "event_type": "MERGES",
                "merge_count": len(merge_table),
                "merges": merge_table
            })
            
        def get_metrics_summary(self):
            return {
                "counters": {"samples_loaded": 1000},
                "timers": {"data_loading": 0.1},
                "runtime_seconds": 1.0
            }
            
        def info(self, msg, **kwargs):
            self.logger.info(msg, extra=kwargs)
    
    class MockMetrics:
        def __init__(self):
            self._counters = {}
            self._timers = {}
            
        def increment(self, name, value=1):
            self._counters[name] = self._counters.get(name, 0) + value
            
        def record_time(self, name, duration):
            self._timers[name] = self._timers.get(name, 0) + duration
            
        def timer(self, name):
            import time
            class Timer:
                def __enter__(self_):
                    self_.start = time.time()
                    return self_
                def __exit__(self_, *args):
                    duration = time.time() - self_.start
                    self.record_time(name, duration)
            return Timer()
            
        def get_summary(self):
            return {
                "counters": self._counters,
                "timers": self._timers,
                "runtime_seconds": 1.0
            }
    
    _metrics = MockMetrics()
    
    def get_metrics():
        return _metrics
        
    def increment_counter(name, value=1):
        _metrics.increment(name, value)
        
    def record_event(event, **kwargs):
        pass
        
    def log_config(config, config_hash):
        pass
        
    def log_seeds(**seeds):
        pass
        
    def log_fsq_config(levels, embedding_dim, codebook_size):
        pass
        
    def log_clustering(algorithm, k_selected, metric_used, metric_value):
        pass
        
    def log_qa_issue(issue_type, count, details=None):
        pass
        
    def get_logger(name=None):
        logger = logging.getLogger(name or "conv2d")
        return MockLogger(logger)

from conv2d.logging.analysis import (
    LogParser,
    LogReporter,
    MetricsAggregator,
    analyze_log_file,
    grep_logs,
    find_errors,
)


def example_structured_logging():
    """Example 1: Structured logging with key events."""
    print("=" * 60)
    print("Example 1: Structured Logging")
    print("=" * 60)
    
    # Create temp log file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_file = Path(f.name)
    
    # Setup logging
    if HAS_STRUCTLOG:
        logger = setup_logging(
            name="conv2d_demo",
            level="INFO",
            output_file=log_file,
        )
    else:
        # Use JSON formatter fallback
        base_logger = setup_logging(
            logger_name="conv2d_demo",
            level="INFO",
            output_file=str(log_file),
        )
        logger = MockLogger(base_logger)
    
    # Log configuration
    config = {
        "model": "conv2d_fsq",
        "dataset": "quadruped",
        "batch_size": 32,
        "learning_rate": 0.001,
    }
    config_hash = "abc123def456"
    logger.log_config(config, config_hash)
    
    # Log random seeds
    logger.log_seeds(numpy=42, torch=42, random=42)
    
    # Log FSQ configuration
    logger.log_fsq_config(
        levels=[8, 6, 5],
        embedding_dim=64,
        codebook_size=240,
    )
    
    # Simulate processing with metrics
    with get_metrics().timer("data_loading"):
        time.sleep(0.1)
        logger.info("Data loaded", num_samples=1000)
        increment_counter("samples_loaded", 1000)
    
    # Log clustering
    logger.log_clustering(
        algorithm="gmm",
        k_selected=4,
        metric_used="bic",
        metric_value=-1234.5,
    )
    
    # Log some QA issues
    logger.log_qa_issue("nan", 5, {"locations": [10, 20, 30, 40, 50]})
    logger.log_qa_issue("outlier", 2)
    
    # Log data processing
    logger.log_data_processing(
        rows_processed=1000,
        rows_interpolated=50,
        rows_all_nan=2,
        rows_mean_fallback=10,
    )
    
    # Log temporal smoothing
    logger.log_temporal_smoothing(
        policy="median_hysteresis",
        min_dwell=5,
        transitions_before=150,
        transitions_after=40,
    )
    
    # Simulate an error
    try:
        raise ValueError("Simulated error for demo")
    except ValueError as e:
        logger.log_exception(e, {"step": "demo", "iteration": 1})
    
    # Log merges
    merge_table = [
        {"source": 5, "target": 3, "n_samples": 10},
        {"source": 7, "target": 4, "n_samples": 8},
    ]
    logger.log_merges(merge_table)
    
    # Get final metrics
    metrics_summary = logger.get_metrics_summary()
    logger.info("Training complete", **metrics_summary)
    
    print(f"\nLog file created: {log_file}")
    print(f"Metrics summary: {json.dumps(metrics_summary, indent=2)}")
    
    return log_file


def example_log_parsing(log_file):
    """Example 2: Parsing and filtering logs."""
    print("\n" + "=" * 60)
    print("Example 2: Log Parsing and Filtering")
    print("=" * 60)
    
    parser = LogParser(log_file)
    
    # Parse all entries
    all_entries = list(parser.parse_lines())
    print(f"\nTotal log entries: {len(all_entries)}")
    
    # Filter by level
    info_logs = parser.filter_by_level("INFO")
    warning_logs = parser.filter_by_level("WARNING")
    error_logs = parser.filter_by_level("ERROR")
    
    print(f"INFO entries: {len(info_logs)}")
    print(f"WARNING entries: {len(warning_logs)}")
    print(f"ERROR entries: {len(error_logs)}")
    
    # Filter by event type
    config_events = parser.filter_by_event_type("CONFIG")
    qa_events = parser.filter_by_event_type("QA_ISSUE")
    
    print(f"\nConfig events: {len(config_events)}")
    if config_events:
        print(f"  Config hash: {config_events[0].get('config_hash')}")
    
    print(f"QA issue events: {len(qa_events)}")
    for qa in qa_events:
        print(f"  {qa['issue_type']}: {qa['count']} occurrences")
    
    # Search logs
    results = parser.search("loaded")
    print(f"\nSearch for 'loaded': {len(results)} matches")
    for entry in results:
        print(f"  - {entry['message']}")
    
    # Get exceptions
    exceptions = parser.get_exceptions()
    print(f"\nExceptions found: {len(exceptions)}")
    for exc in exceptions:
        print(f"  {exc['exception_type']}: {exc['exception_message']}")
    
    # Get QA issues summary
    qa_summary = parser.get_qa_issues()
    print(f"\nQA Issues Summary:")
    for issue_type, count in qa_summary.items():
        print(f"  {issue_type}: {count}")


def example_metrics_timeline(log_file):
    """Example 3: Extract metrics timeline."""
    print("\n" + "=" * 60)
    print("Example 3: Metrics Timeline")
    print("=" * 60)
    
    parser = LogParser(log_file)
    df = parser.get_metrics_timeline()
    
    if df.empty:
        print("No metrics found in logs")
        return
    
    print(f"\nMetrics timeline ({len(df)} entries):")
    print("\nColumns available:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Show counters
    counter_cols = [c for c in df.columns if c.startswith("counter_")]
    if counter_cols:
        print("\nCounter totals:")
        for col in counter_cols:
            total = df[col].sum()
            print(f"  {col.replace('counter_', '')}: {total}")
    
    # Show timers
    timer_cols = [c for c in df.columns if c.startswith("timer_")]
    if timer_cols:
        print("\nTimer totals:")
        for col in timer_cols:
            total = df[col].sum()
            print(f"  {col.replace('timer_', '')}: {total:.3f}s")


def example_log_report(log_file):
    """Example 4: Generate log reports."""
    print("\n" + "=" * 60)
    print("Example 4: Log Reports")
    print("=" * 60)
    
    reporter = LogReporter(log_file)
    
    # Generate summary report
    summary = reporter.generate_summary_report()
    print("\n--- Summary Report ---")
    print(summary)
    
    # Generate error report
    error_report = reporter.generate_error_report()
    print("\n--- Error Report ---")
    print(error_report)
    
    # Export to CSV
    csv_file = log_file.with_suffix(".csv")
    reporter.export_to_csv(csv_file)
    print(f"\nLog exported to CSV: {csv_file}")


def example_multi_run_comparison():
    """Example 5: Compare multiple experiment runs."""
    print("\n" + "=" * 60)
    print("Example 5: Multi-Run Comparison")
    print("=" * 60)
    
    aggregator = MetricsAggregator()
    
    # Simulate multiple runs
    runs = []
    for i, (acc, f1, ece) in enumerate([
        (0.85, 0.82, 0.05),
        (0.87, 0.84, 0.04),
        (0.83, 0.80, 0.06),
    ]):
        # Create temp log for each run
        with tempfile.NamedTemporaryFile(mode="w", suffix=f"_run{i}.log", delete=False) as f:
            log_file = Path(f.name)
        
        # Setup logging for this run
        if HAS_STRUCTLOG:
            logger = setup_logging(
                name=f"run_{i}",
                level="INFO",
                output_file=log_file,
                reset_metrics=True,
            )
        else:
            base_logger = setup_logging(
                logger_name=f"run_{i}",
                level="INFO",
                output_file=str(log_file),
            )
            logger = MockLogger(base_logger)
        
        # Log evaluation metrics
        logger.info(
            "evaluation_complete",
            accuracy=acc,
            macro_f1=f1,
            ece=ece,
            coverage=0.95,
            motif_count=4,
            event_type="EVALUATION",
        )
        
        # Add some counters
        increment_counter("samples_processed", 1000 + i * 100)
        increment_counter("epochs", 100 + i * 10)
        
        # Log final metrics
        metrics = get_metrics().get_summary()
        logger.info("run_complete", metrics=metrics)
        
        runs.append(log_file)
        aggregator.add_run(log_file, f"run_{i}")
    
    # Compare runs
    comparison_df = aggregator.compare_runs()
    print("\nRun Comparison:")
    print(comparison_df.to_string())
    
    # Get summary statistics
    summary = aggregator.summarize()
    print("\nSummary Statistics:")
    print(f"  Number of runs: {summary['num_runs']}")
    
    for metric in ["accuracy", "macro_f1", "ece"]:
        if metric in summary:
            stats = summary[metric]
            print(f"\n  {metric}:")
            print(f"    Mean: {stats['mean']:.3f}")
            print(f"    Std:  {stats['std']:.3f}")
            print(f"    Max:  {stats['max']:.3f}")
            print(f"    Min:  {stats['min']:.3f}")
    
    # Clean up
    for log_file in runs:
        log_file.unlink()


def example_json_formatter():
    """Example 6: Alternative JSON formatter (without structlog)."""
    print("\n" + "=" * 60)
    print("Example 6: JSON Formatter (lightweight)")
    print("=" * 60)
    
    import logging
    from conv2d.logging.json_formatter import JSONFormatter, MetricsFilter, setup_json_logging
    
    # Setup JSON logging
    logger = setup_json_logging(
        logger_name="json_demo",
        level="INFO",
    )
    
    # Create metrics filter
    metrics = get_metrics()
    metrics_filter = MetricsFilter(metrics)
    logger.addFilter(metrics_filter)
    
    # Log some events
    logger.info("Starting process")
    
    # Add extra fields
    logger.info("Processing data", 
                extra={"samples": 1000, "batch_size": 32})
    
    # Simulate work with metrics
    increment_counter("processed", 100)
    metrics.record_time("inference", 1.5)
    
    # Log warning (will include metrics)
    logger.warning("High memory usage", 
                   extra={"memory_mb": 2048})
    
    # Log with context
    from conv2d.logging.json_formatter import LogContext
    
    with LogContext(logger, request_id="abc123", user="demo") as ctx_logger:
        ctx_logger.info("Request processing")
        ctx_logger.info("Request complete")
    
    print("\nJSON formatter demonstrated (logs written to console)")


def main():
    """Run all logging examples."""
    # Example 1: Create structured logs
    log_file = example_structured_logging()
    
    # Example 2: Parse and filter logs
    example_log_parsing(log_file)
    
    # Example 3: Extract metrics timeline
    example_metrics_timeline(log_file)
    
    # Example 4: Generate reports
    example_log_report(log_file)
    
    # Example 5: Compare multiple runs
    example_multi_run_comparison()
    
    # Example 6: JSON formatter
    example_json_formatter()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    
    # Clean up
    log_file.unlink()
    log_file.with_suffix(".csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()