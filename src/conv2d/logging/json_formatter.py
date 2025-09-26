"""Simple JSON formatter for Python's standard logging.

Provides a lightweight alternative to structlog that works
with existing logging setup.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for standard Python logging."""
    
    def __init__(
        self,
        include_timestamp: bool = True,
        include_location: bool = True,
        include_process: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        """Initialize JSON formatter.
        
        Args:
            include_timestamp: Include timestamp in logs
            include_location: Include file/line information
            include_process: Include process/thread information
            extra_fields: Extra fields to include in all logs
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_location = include_location
        self.include_process = include_process
        self.extra_fields = extra_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log line
        """
        # Build log entry
        log_entry = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add timestamp
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.fromtimestamp(
                record.created
            ).isoformat()
        
        # Add location information
        if self.include_location:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }
        
        # Add process information
        if self.include_process:
            log_entry["process"] = {
                "pid": record.process,
                "thread": record.thread,
                "thread_name": record.threadName,
            }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName',
                'relativeCreated', 'thread', 'threadName', 'exc_info',
                'exc_text', 'stack_info'
            ]:
                # Ensure JSON serializable
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)
        
        # Add global extra fields
        log_entry.update(self.extra_fields)
        
        # Convert to JSON
        return json.dumps(log_entry, default=str)


class MetricsFilter(logging.Filter):
    """Filter that adds metrics to log records."""
    
    def __init__(self, metrics_collector):
        """Initialize metrics filter.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        super().__init__()
        self.metrics = metrics_collector
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add metrics to log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to pass the record
        """
        # Add metrics summary for important logs
        if record.levelno >= logging.WARNING:
            record.metrics = self.metrics.get_summary()
        
        return True


def setup_json_logging(
    logger_name: Optional[str] = None,
    level: str = "INFO",
    output_file: Optional[str] = None,
    include_console: bool = True,
) -> logging.Logger:
    """Setup JSON logging for a logger.
    
    Args:
        logger_name: Logger name (None for root)
        level: Log level
        output_file: Optional file to log to
        include_console: Whether to include console output
        
    Returns:
        Configured logger
    """
    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create JSON formatter
    formatter = JSONFormatter()
    
    # Add console handler
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if output_file:
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        """Initialize log context.
        
        Args:
            logger: Logger to add context to
            **context: Context fields to add
        """
        self.logger = logger
        self.context = context
        self.adapter = None
    
    def __enter__(self):
        """Enter context."""
        self.adapter = logging.LoggerAdapter(
            self.logger,
            self.context
        )
        return self.adapter
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.adapter = None