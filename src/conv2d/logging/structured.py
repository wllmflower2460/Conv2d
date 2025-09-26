"""Structured logging for Conv2d with JSON output and metrics tracking.

Provides structured logging with:
- JSON formatting for log aggregation
- Key event tracking
- Metrics counters
- Contextual information
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level


class MetricsCollector:
    """Collect and track metrics throughout execution."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self._counters = defaultdict(int)
        self._timers = defaultdict(float)
        self._events = []
        self._start_time = time.time()
        
    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter metric.
        
        Args:
            name: Counter name
            value: Increment value
        """
        self._counters[name] += value
    
    def record_time(self, name: str, duration: float) -> None:
        """Record a timing metric.
        
        Args:
            name: Timer name
            duration: Duration in seconds
        """
        self._timers[name] += duration
    
    def record_event(self, event: str, **kwargs) -> None:
        """Record a key event.
        
        Args:
            event: Event name
            **kwargs: Event attributes
        """
        self._events.append({
            "event": event,
            "timestamp": time.time(),
            **kwargs
        })
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations.
        
        Args:
            name: Timer name
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.record_time(name, duration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.
        
        Returns:
            Dictionary of all collected metrics
        """
        return {
            "counters": dict(self._counters),
            "timers": dict(self._timers),
            "events_count": len(self._events),
            "runtime_seconds": time.time() - self._start_time,
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._timers.clear()
        self._events.clear()
        self._start_time = time.time()


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics


class StructuredLogger:
    """Structured logger with JSON output and context tracking."""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        output_file: Optional[Path] = None,
        include_timestamp: bool = True,
        include_caller: bool = True,
    ):
        """Initialize structured logger.
        
        Args:
            name: Logger name
            level: Log level
            output_file: Optional file to write logs to
            include_timestamp: Whether to include timestamps
            include_caller: Whether to include caller information
        """
        self.name = name
        self.level = level
        self.output_file = output_file
        
        # Configure processors
        processors = []
        
        if include_timestamp:
            processors.append(TimeStamper(fmt="iso"))
        
        processors.extend([
            add_log_level,
            self._add_context,
            self._add_metrics,
        ])
        
        if include_caller:
            processors.append(self._add_caller_info)
        
        # Add JSON renderer last
        processors.append(JSONRenderer())
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.BoundLogger,
            logger_factory=self._logger_factory,
            cache_logger_on_first_use=True,
        )
        
        # Get logger instance
        self.logger = structlog.get_logger(name)
        
        # Setup file handler if needed
        if output_file:
            self._setup_file_handler(output_file)
    
    def _logger_factory(self):
        """Create underlying logger."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level))
        
        # Add console handler with JSON output
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
        
        return logger
    
    def _setup_file_handler(self, output_file: Path):
        """Setup file handler for logging.
        
        Args:
            output_file: Path to log file
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Add to logger
        base_logger = logging.getLogger(self.name)
        base_logger.addHandler(file_handler)
    
    def _add_context(self, logger, method_name, event_dict):
        """Add contextual information to log event."""
        # Add standard context
        event_dict["logger"] = self.name
        
        # Add any bound context
        if hasattr(logger, '_context'):
            event_dict.update(logger._context)
        
        return event_dict
    
    def _add_metrics(self, logger, method_name, event_dict):
        """Add current metrics to important log events."""
        # Only add metrics for certain log levels
        if method_name in ['error', 'warning', 'critical']:
            metrics_summary = _metrics.get_summary()
            if metrics_summary['counters'] or metrics_summary['timers']:
                event_dict['metrics'] = metrics_summary
        
        return event_dict
    
    def _add_caller_info(self, logger, method_name, event_dict):
        """Add caller information to log event."""
        import inspect
        
        # Get caller frame (skip structlog frames)
        frame = inspect.currentframe()
        for _ in range(8):  # Skip through structlog frames
            if frame and frame.f_back:
                frame = frame.f_back
        
        if frame:
            event_dict['caller'] = {
                'filename': frame.f_code.co_filename.split('/')[-1],
                'function': frame.f_code.co_name,
                'line': frame.f_lineno,
            }
        
        return event_dict
    
    def bind(self, **kwargs) -> structlog.BoundLogger:
        """Bind contextual information to logger.
        
        Args:
            **kwargs: Context to bind
            
        Returns:
            Bound logger with context
        """
        return self.logger.bind(**kwargs)
    
    def unbind(self, *keys) -> structlog.BoundLogger:
        """Unbind contextual information.
        
        Args:
            *keys: Keys to unbind
            
        Returns:
            Logger with context removed
        """
        return self.logger.unbind(*keys)
    
    # Logging methods
    def debug(self, msg: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(msg, **kwargs)
    
    # Specific event logging
    def log_config(self, config: Dict[str, Any], config_hash: str) -> None:
        """Log configuration with hash.
        
        Args:
            config: Configuration dictionary
            config_hash: Configuration hash
        """
        self.info(
            "configuration_loaded",
            config_hash=config_hash,
            config_keys=list(config.keys()),
            event_type="CONFIG",
        )
        _metrics.record_event("config_loaded", hash=config_hash)
    
    def log_seeds(self, **seeds) -> None:
        """Log random seeds.
        
        Args:
            **seeds: Seed values by name
        """
        self.info(
            "random_seeds_set",
            seeds=seeds,
            event_type="SEEDS",
        )
        _metrics.record_event("seeds_set", **seeds)
    
    def log_fsq_config(
        self,
        levels: list[int],
        embedding_dim: int,
        codebook_size: int,
    ) -> None:
        """Log FSQ configuration.
        
        Args:
            levels: FSQ levels
            embedding_dim: Embedding dimension
            codebook_size: Total codebook size
        """
        self.info(
            "fsq_configuration",
            levels=levels,
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            event_type="FSQ_CONFIG",
        )
        _metrics.record_event(
            "fsq_configured",
            levels=levels,
            codebook_size=codebook_size
        )
    
    def log_clustering(
        self,
        algorithm: str,
        k_selected: int,
        metric_used: str,
        metric_value: float,
    ) -> None:
        """Log clustering results.
        
        Args:
            algorithm: Clustering algorithm used
            k_selected: Number of clusters selected
            metric_used: Selection metric (BIC, silhouette)
            metric_value: Metric value
        """
        self.info(
            "clustering_complete",
            algorithm=algorithm,
            k_selected=k_selected,
            metric_used=metric_used,
            metric_value=metric_value,
            event_type="CLUSTERING",
        )
        _metrics.record_event(
            "clustering",
            k=k_selected,
            algorithm=algorithm
        )
        _metrics.increment(f"clusters_{algorithm}", k_selected)
    
    def log_merges(self, merge_table: list[Dict[str, Any]]) -> None:
        """Log cluster merge operations.
        
        Args:
            merge_table: List of merge operations
        """
        self.info(
            "clusters_merged",
            merge_count=len(merge_table),
            merges=merge_table,
            event_type="MERGES",
        )
        _metrics.increment("cluster_merges", len(merge_table))
        for merge in merge_table:
            _metrics.record_event(
                "cluster_merge",
                source=merge.get('source'),
                target=merge.get('target'),
                samples=merge.get('n_samples')
            )
    
    def log_temporal_smoothing(
        self,
        policy: str,
        min_dwell: int,
        transitions_before: int,
        transitions_after: int,
    ) -> None:
        """Log temporal smoothing application.
        
        Args:
            policy: Smoothing policy used
            min_dwell: Minimum dwell time
            transitions_before: Transitions before smoothing
            transitions_after: Transitions after smoothing
        """
        reduction = (1 - transitions_after / max(transitions_before, 1)) * 100
        
        self.info(
            "temporal_smoothing_applied",
            policy=policy,
            min_dwell=min_dwell,
            transitions_before=transitions_before,
            transitions_after=transitions_after,
            reduction_percent=reduction,
            event_type="SMOOTHING",
        )
        _metrics.increment("transitions_removed", transitions_before - transitions_after)
        _metrics.record_event(
            "temporal_smoothing",
            policy=policy,
            reduction_percent=reduction
        )
    
    def log_qa_issue(
        self,
        issue_type: str,
        count: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log QA issue.
        
        Args:
            issue_type: Type of issue (nan, inf, outlier)
            count: Number of occurrences
            details: Additional details
        """
        self.warning(
            "qa_issue_detected",
            issue_type=issue_type,
            count=count,
            details=details or {},
            event_type="QA_ISSUE",
        )
        _metrics.increment(f"qa_{issue_type}", count)
    
    def log_data_processing(
        self,
        rows_processed: int,
        rows_interpolated: int,
        rows_all_nan: int,
        rows_mean_fallback: int,
    ) -> None:
        """Log data processing statistics.
        
        Args:
            rows_processed: Total rows processed
            rows_interpolated: Rows with interpolation
            rows_all_nan: Rows that were all NaN
            rows_mean_fallback: Rows using mean fallback
        """
        self.info(
            "data_processing_complete",
            rows_processed=rows_processed,
            rows_interpolated=rows_interpolated,
            rows_all_nan=rows_all_nan,
            rows_mean_fallback=rows_mean_fallback,
            interpolation_rate=rows_interpolated / max(rows_processed, 1),
            event_type="DATA_PROCESSING",
        )
        _metrics.increment("rows_processed", rows_processed)
        _metrics.increment("rows_interpolated", rows_interpolated)
        _metrics.increment("rows_all_nan", rows_all_nan)
        _metrics.increment("rows_mean_fallback", rows_mean_fallback)
    
    def log_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log exception with context.
        
        Args:
            exception: Exception that occurred
            context: Additional context
        """
        import traceback
        
        self.error(
            "exception_occurred",
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            traceback=traceback.format_exc(),
            context=context or {},
            event_type="EXCEPTION",
        )
        _metrics.increment("exceptions")
        _metrics.record_event(
            "exception",
            type=type(exception).__name__,
            message=str(exception)
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary.
        
        Returns:
            Metrics summary dictionary
        """
        return _metrics.get_summary()


# Convenience functions for global logger
_global_logger: Optional[StructuredLogger] = None


def setup_logging(
    name: str = "conv2d",
    level: str = "INFO",
    output_file: Optional[Union[str, Path]] = None,
    reset_metrics: bool = True,
) -> StructuredLogger:
    """Setup global structured logging.
    
    Args:
        name: Logger name
        level: Log level
        output_file: Optional log file path
        reset_metrics: Whether to reset metrics
        
    Returns:
        Configured logger
    """
    global _global_logger
    
    if output_file:
        output_file = Path(output_file)
    
    _global_logger = StructuredLogger(
        name=name,
        level=level,
        output_file=output_file,
    )
    
    if reset_metrics:
        _metrics.reset()
    
    # Log startup
    _global_logger.info(
        "logging_initialized",
        name=name,
        level=level,
        output_file=str(output_file) if output_file else None,
        event_type="STARTUP",
    )
    
    return _global_logger


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """Get logger instance.
    
    Args:
        name: Optional logger name
        
    Returns:
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = setup_logging(name or "conv2d")
    
    if name and name != _global_logger.name:
        # Create child logger
        return StructuredLogger(name=name)
    
    return _global_logger


# Convenience logging functions
def log_config(config: Dict[str, Any], config_hash: str) -> None:
    """Log configuration."""
    get_logger().log_config(config, config_hash)


def log_seeds(**seeds) -> None:
    """Log random seeds."""
    get_logger().log_seeds(**seeds)


def log_fsq_config(levels: list[int], embedding_dim: int, codebook_size: int) -> None:
    """Log FSQ configuration."""
    get_logger().log_fsq_config(levels, embedding_dim, codebook_size)


def log_clustering(algorithm: str, k_selected: int, metric_used: str, metric_value: float) -> None:
    """Log clustering results."""
    get_logger().log_clustering(algorithm, k_selected, metric_used, metric_value)


def log_qa_issue(issue_type: str, count: int, details: Optional[Dict[str, Any]] = None) -> None:
    """Log QA issue."""
    get_logger().log_qa_issue(issue_type, count, details)


def increment_counter(name: str, value: int = 1) -> None:
    """Increment a counter metric."""
    _metrics.increment(name, value)


def record_event(event: str, **kwargs) -> None:
    """Record a key event."""
    _metrics.record_event(event, **kwargs)