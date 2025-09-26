"""Log analysis utilities for parsing and aggregating JSON logs.

Provides tools for:
- Parsing JSON log files
- Filtering and searching logs
- Aggregating metrics across runs
- Generating reports from log data
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class LogParser:
    """Parse and analyze JSON log files."""
    
    def __init__(self, log_path: Union[str, Path]):
        """Initialize parser with log file path.
        
        Args:
            log_path: Path to log file or directory
        """
        self.log_path = Path(log_path)
        self._cache = {}
    
    def parse_lines(self) -> Iterator[Dict[str, Any]]:
        """Parse log file line by line.
        
        Yields:
            Parsed JSON log entries
        """
        if self.log_path.is_file():
            files = [self.log_path]
        else:
            files = sorted(self.log_path.glob("*.log"))
        
        for file_path in files:
            with open(file_path) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        yield json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        # Skip non-JSON lines (could be plain text logs)
                        continue
    
    def filter_by_level(self, level: str) -> List[Dict[str, Any]]:
        """Filter logs by level.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, CRITICAL)
            
        Returns:
            Filtered log entries
        """
        return [
            entry for entry in self.parse_lines()
            if entry.get("level") == level.upper()
        ]
    
    def filter_by_event_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Filter logs by event type.
        
        Args:
            event_type: Event type (CONFIG, SEEDS, FSQ_CONFIG, etc.)
            
        Returns:
            Filtered log entries
        """
        return [
            entry for entry in self.parse_lines()
            if entry.get("event_type") == event_type
        ]
    
    def filter_by_time_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Filter logs by time range.
        
        Args:
            start: Start time (inclusive)
            end: End time (inclusive)
            
        Returns:
            Filtered log entries
        """
        results = []
        for entry in self.parse_lines():
            if "timestamp" not in entry:
                continue
            
            try:
                timestamp = datetime.fromisoformat(entry["timestamp"])
            except (ValueError, TypeError):
                continue
            
            if start and timestamp < start:
                continue
            if end and timestamp > end:
                continue
            
            results.append(entry)
        
        return results
    
    def search(self, pattern: str, field: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search logs with regex pattern.
        
        Args:
            pattern: Regex pattern to search for
            field: Optional field to search in (searches all if None)
            
        Returns:
            Matching log entries
        """
        regex = re.compile(pattern, re.IGNORECASE)
        results = []
        
        for entry in self.parse_lines():
            if field:
                # Search specific field
                value = entry.get(field, "")
                if isinstance(value, str) and regex.search(value):
                    results.append(entry)
            else:
                # Search all string fields
                for value in entry.values():
                    if isinstance(value, str) and regex.search(value):
                        results.append(entry)
                        break
        
        return results
    
    def get_metrics_timeline(self) -> pd.DataFrame:
        """Extract metrics timeline from logs.
        
        Returns:
            DataFrame with metrics over time
        """
        metrics_entries = []
        
        for entry in self.parse_lines():
            if "metrics" in entry:
                timestamp = entry.get("timestamp")
                metrics = entry["metrics"]
                
                # Flatten metrics
                flat_metrics = {"timestamp": timestamp}
                
                # Add counters
                for name, value in metrics.get("counters", {}).items():
                    flat_metrics[f"counter_{name}"] = value
                
                # Add timers
                for name, value in metrics.get("timers", {}).items():
                    flat_metrics[f"timer_{name}"] = value
                
                # Add runtime
                if "runtime_seconds" in metrics:
                    flat_metrics["runtime"] = metrics["runtime_seconds"]
                
                metrics_entries.append(flat_metrics)
        
        if not metrics_entries:
            return pd.DataFrame()
        
        df = pd.DataFrame(metrics_entries)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        
        return df
    
    def get_exceptions(self) -> List[Dict[str, Any]]:
        """Extract all exceptions from logs.
        
        Returns:
            List of exception entries
        """
        return [
            entry for entry in self.parse_lines()
            if entry.get("event_type") == "EXCEPTION" or "exception" in entry
        ]
    
    def get_qa_issues(self) -> Dict[str, int]:
        """Extract and count QA issues.
        
        Returns:
            Dictionary of issue types and counts
        """
        qa_counts = defaultdict(int)
        
        for entry in self.parse_lines():
            if entry.get("event_type") == "QA_ISSUE":
                issue_type = entry.get("issue_type", "unknown")
                count = entry.get("count", 1)
                qa_counts[issue_type] += count
        
        return dict(qa_counts)


class MetricsAggregator:
    """Aggregate metrics across multiple runs."""
    
    def __init__(self):
        """Initialize aggregator."""
        self.runs = []
    
    def add_run(self, log_path: Union[str, Path], run_name: Optional[str] = None):
        """Add a run's logs for aggregation.
        
        Args:
            log_path: Path to log file
            run_name: Optional name for the run
        """
        parser = LogParser(log_path)
        
        if run_name is None:
            run_name = Path(log_path).stem
        
        # Extract key metrics
        run_data = {
            "name": run_name,
            "path": str(log_path),
            "metrics": self._extract_run_metrics(parser),
        }
        
        self.runs.append(run_data)
    
    def _extract_run_metrics(self, parser: LogParser) -> Dict[str, Any]:
        """Extract metrics from a single run.
        
        Args:
            parser: Log parser for the run
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {
            "accuracy": None,
            "macro_f1": None,
            "ece": None,
            "mce": None,
            "coverage": None,
            "motif_count": None,
            "runtime": None,
            "exceptions": 0,
            "warnings": 0,
            "qa_issues": {},
        }
        
        # Find evaluation metrics
        for entry in parser.filter_by_event_type("EVALUATION"):
            if "accuracy" in entry:
                metrics["accuracy"] = entry["accuracy"]
            if "macro_f1" in entry:
                metrics["macro_f1"] = entry["macro_f1"]
            if "ece" in entry:
                metrics["ece"] = entry["ece"]
            if "mce" in entry:
                metrics["mce"] = entry["mce"]
            if "coverage" in entry:
                metrics["coverage"] = entry["coverage"]
            if "motif_count" in entry:
                metrics["motif_count"] = entry["motif_count"]
        
        # Count issues
        metrics["exceptions"] = len(parser.get_exceptions())
        metrics["warnings"] = len(parser.filter_by_level("WARNING"))
        metrics["qa_issues"] = parser.get_qa_issues()
        
        # Get runtime
        timeline = parser.get_metrics_timeline()
        if not timeline.empty and "runtime" in timeline.columns:
            metrics["runtime"] = timeline["runtime"].iloc[-1]
        
        return metrics
    
    def compare_runs(self) -> pd.DataFrame:
        """Compare metrics across all runs.
        
        Returns:
            DataFrame with run comparison
        """
        if not self.runs:
            return pd.DataFrame()
        
        # Create comparison table
        rows = []
        for run in self.runs:
            row = {"run": run["name"]}
            row.update(run["metrics"])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by accuracy if available
        if "accuracy" in df.columns:
            df = df.sort_values("accuracy", ascending=False)
        
        return df
    
    def summarize(self) -> Dict[str, Any]:
        """Generate summary statistics across runs.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.runs:
            return {}
        
        # Collect metrics
        accuracies = []
        f1_scores = []
        ece_scores = []
        runtimes = []
        
        for run in self.runs:
            metrics = run["metrics"]
            if metrics["accuracy"] is not None:
                accuracies.append(metrics["accuracy"])
            if metrics["macro_f1"] is not None:
                f1_scores.append(metrics["macro_f1"])
            if metrics["ece"] is not None:
                ece_scores.append(metrics["ece"])
            if metrics["runtime"] is not None:
                runtimes.append(metrics["runtime"])
        
        summary = {
            "num_runs": len(self.runs),
            "accuracy": {
                "mean": np.mean(accuracies) if accuracies else None,
                "std": np.std(accuracies) if accuracies else None,
                "max": max(accuracies) if accuracies else None,
                "min": min(accuracies) if accuracies else None,
            },
            "macro_f1": {
                "mean": np.mean(f1_scores) if f1_scores else None,
                "std": np.std(f1_scores) if f1_scores else None,
                "max": max(f1_scores) if f1_scores else None,
                "min": min(f1_scores) if f1_scores else None,
            },
            "ece": {
                "mean": np.mean(ece_scores) if ece_scores else None,
                "std": np.std(ece_scores) if ece_scores else None,
                "max": max(ece_scores) if ece_scores else None,
                "min": min(ece_scores) if ece_scores else None,
            },
            "runtime": {
                "mean": np.mean(runtimes) if runtimes else None,
                "std": np.std(runtimes) if runtimes else None,
                "total": sum(runtimes) if runtimes else None,
            },
        }
        
        return summary


class LogReporter:
    """Generate reports from log data."""
    
    def __init__(self, log_path: Union[str, Path]):
        """Initialize reporter.
        
        Args:
            log_path: Path to log file
        """
        self.parser = LogParser(log_path)
    
    def generate_summary_report(self) -> str:
        """Generate a summary report.
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("LOG ANALYSIS REPORT")
        lines.append("=" * 60)
        
        # Count log levels
        level_counts = defaultdict(int)
        event_counts = defaultdict(int)
        total_entries = 0
        
        for entry in self.parser.parse_lines():
            total_entries += 1
            level = entry.get("level", "UNKNOWN")
            level_counts[level] += 1
            
            event_type = entry.get("event_type")
            if event_type:
                event_counts[event_type] += 1
        
        # Log level summary
        lines.append("\nLog Level Summary:")
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            count = level_counts.get(level, 0)
            pct = (count / max(total_entries, 1)) * 100
            lines.append(f"  {level:8s}: {count:6d} ({pct:5.1f}%)")
        
        # Event type summary
        if event_counts:
            lines.append("\nEvent Type Summary:")
            for event_type, count in sorted(event_counts.items()):
                lines.append(f"  {event_type:20s}: {count:6d}")
        
        # QA issues
        qa_issues = self.parser.get_qa_issues()
        if qa_issues:
            lines.append("\nQA Issues:")
            total_issues = sum(qa_issues.values())
            for issue_type, count in sorted(qa_issues.items()):
                lines.append(f"  {issue_type:20s}: {count:6d}")
            lines.append(f"  {'TOTAL':20s}: {total_issues:6d}")
        
        # Exceptions
        exceptions = self.parser.get_exceptions()
        if exceptions:
            lines.append(f"\nExceptions: {len(exceptions)}")
            # Show unique exception types
            exception_types = defaultdict(int)
            for exc in exceptions:
                exc_type = exc.get("exception_type", "Unknown")
                exception_types[exc_type] += 1
            
            for exc_type, count in sorted(exception_types.items()):
                lines.append(f"  {exc_type:30s}: {count:3d}")
        
        # Metrics timeline
        timeline = self.parser.get_metrics_timeline()
        if not timeline.empty:
            lines.append("\nMetrics Summary:")
            
            # Show counter totals
            counter_cols = [c for c in timeline.columns if c.startswith("counter_")]
            if counter_cols:
                lines.append("  Counters:")
                for col in counter_cols:
                    name = col.replace("counter_", "")
                    total = timeline[col].sum()
                    lines.append(f"    {name:30s}: {total:10.0f}")
            
            # Show timer totals
            timer_cols = [c for c in timeline.columns if c.startswith("timer_")]
            if timer_cols:
                lines.append("  Timers:")
                for col in timer_cols:
                    name = col.replace("timer_", "")
                    total = timeline[col].sum()
                    lines.append(f"    {name:30s}: {total:10.2f}s")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def generate_error_report(self) -> str:
        """Generate a report of errors and exceptions.
        
        Returns:
            Formatted error report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ERROR REPORT")
        lines.append("=" * 60)
        
        # Get errors and exceptions
        errors = self.parser.filter_by_level("ERROR")
        exceptions = self.parser.get_exceptions()
        
        if not errors and not exceptions:
            lines.append("\nNo errors or exceptions found.")
            return "\n".join(lines)
        
        # Show errors
        if errors:
            lines.append(f"\nErrors: {len(errors)}")
            for i, entry in enumerate(errors[:10], 1):  # Show first 10
                timestamp = entry.get("timestamp", "")
                message = entry.get("message", "")
                lines.append(f"\n  {i}. [{timestamp}]")
                lines.append(f"     {message}")
                
                # Show additional context if available
                for key in ["caller", "context"]:
                    if key in entry:
                        lines.append(f"     {key}: {entry[key]}")
        
        # Show exceptions
        if exceptions:
            lines.append(f"\n\nExceptions: {len(exceptions)}")
            for i, entry in enumerate(exceptions[:10], 1):  # Show first 10
                timestamp = entry.get("timestamp", "")
                exc_type = entry.get("exception_type", "Unknown")
                exc_msg = entry.get("exception_message", "")
                
                lines.append(f"\n  {i}. [{timestamp}] {exc_type}")
                lines.append(f"     {exc_msg}")
                
                # Show context if available
                if "context" in entry:
                    lines.append(f"     Context: {entry['context']}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def export_to_csv(self, output_path: Union[str, Path], filter_func=None):
        """Export logs to CSV format.
        
        Args:
            output_path: Output CSV file path
            filter_func: Optional filter function for entries
        """
        output_path = Path(output_path)
        
        # Collect entries
        entries = []
        for entry in self.parser.parse_lines():
            if filter_func and not filter_func(entry):
                continue
            
            # Flatten nested structures
            flat_entry = {}
            for key, value in entry.items():
                if isinstance(value, (dict, list)):
                    flat_entry[key] = json.dumps(value)
                else:
                    flat_entry[key] = value
            
            entries.append(flat_entry)
        
        # Convert to DataFrame and save
        if entries:
            df = pd.DataFrame(entries)
            df.to_csv(output_path, index=False)
            print(f"Exported {len(entries)} entries to {output_path}")
        else:
            print("No entries to export")


# Convenience functions
def analyze_log_file(log_path: Union[str, Path]) -> None:
    """Quick analysis of a log file.
    
    Args:
        log_path: Path to log file
    """
    reporter = LogReporter(log_path)
    print(reporter.generate_summary_report())


def compare_experiments(log_dir: Union[str, Path]) -> pd.DataFrame:
    """Compare multiple experiment runs.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Comparison DataFrame
    """
    aggregator = MetricsAggregator()
    
    log_dir = Path(log_dir)
    for log_file in sorted(log_dir.glob("*.log")):
        aggregator.add_run(log_file)
    
    return aggregator.compare_runs()


def find_errors(log_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Find all errors in a log file.
    
    Args:
        log_path: Path to log file
        
    Returns:
        List of error entries
    """
    parser = LogParser(log_path)
    return parser.filter_by_level("ERROR") + parser.get_exceptions()


def grep_logs(
    log_path: Union[str, Path],
    pattern: str,
    field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Grep-like search through logs.
    
    Args:
        log_path: Path to log file
        pattern: Search pattern (regex)
        field: Optional field to search in
        
    Returns:
        Matching entries
    """
    parser = LogParser(log_path)
    return parser.search(pattern, field)