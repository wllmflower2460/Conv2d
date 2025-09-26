"""Tests for log analysis utilities."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from conv2d.logging.analysis import (
    LogParser,
    LogReporter,
    MetricsAggregator,
    analyze_log_file,
    compare_experiments,
    find_errors,
    grep_logs,
)


@pytest.fixture
def sample_log_file():
    """Create a sample log file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        # Write sample log entries
        entries = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "level": "INFO",
                "message": "Starting experiment",
                "event_type": "STARTUP",
            },
            {
                "timestamp": "2024-01-01T10:00:01",
                "level": "INFO",
                "message": "Configuration loaded",
                "event_type": "CONFIG",
                "config_hash": "abc123",
            },
            {
                "timestamp": "2024-01-01T10:00:02",
                "level": "INFO",
                "message": "Random seeds set",
                "event_type": "SEEDS",
                "seeds": {"numpy": 42, "torch": 42},
            },
            {
                "timestamp": "2024-01-01T10:00:03",
                "level": "INFO",
                "message": "FSQ configuration",
                "event_type": "FSQ_CONFIG",
                "levels": [8, 6, 5],
                "embedding_dim": 64,
                "codebook_size": 240,
            },
            {
                "timestamp": "2024-01-01T10:00:10",
                "level": "INFO",
                "message": "Clustering complete",
                "event_type": "CLUSTERING",
                "algorithm": "gmm",
                "k_selected": 4,
                "metric_used": "bic",
                "metric_value": -1234.5,
            },
            {
                "timestamp": "2024-01-01T10:00:15",
                "level": "WARNING",
                "message": "QA issue detected",
                "event_type": "QA_ISSUE",
                "issue_type": "nan",
                "count": 5,
            },
            {
                "timestamp": "2024-01-01T10:00:20",
                "level": "ERROR",
                "message": "Error during processing",
                "exception_type": "ValueError",
                "exception_message": "Invalid value",
                "event_type": "EXCEPTION",
            },
            {
                "timestamp": "2024-01-01T10:00:30",
                "level": "INFO",
                "message": "Evaluation complete",
                "event_type": "EVALUATION",
                "accuracy": 0.85,
                "macro_f1": 0.82,
                "ece": 0.05,
                "coverage": 0.95,
                "motif_count": 4,
            },
            {
                "timestamp": "2024-01-01T10:00:35",
                "level": "INFO",
                "message": "Metrics update",
                "metrics": {
                    "counters": {
                        "rows_processed": 1000,
                        "rows_interpolated": 50,
                    },
                    "timers": {
                        "training": 120.5,
                        "evaluation": 15.3,
                    },
                    "runtime_seconds": 135.8,
                },
            },
        ]
        
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
        
        return Path(f.name)


@pytest.fixture
def multiple_log_files():
    """Create multiple log files for comparison."""
    files = []
    
    for i, (acc, f1, ece) in enumerate([(0.85, 0.82, 0.05), (0.87, 0.84, 0.04), (0.83, 0.80, 0.06)]):
        with tempfile.NamedTemporaryFile(mode="w", suffix=f"_run{i}.log", delete=False) as f:
            entries = [
                {
                    "timestamp": f"2024-01-0{i+1}T10:00:00",
                    "level": "INFO",
                    "message": "Evaluation complete",
                    "event_type": "EVALUATION",
                    "accuracy": acc,
                    "macro_f1": f1,
                    "ece": ece,
                    "coverage": 0.95,
                    "motif_count": 4,
                },
                {
                    "timestamp": f"2024-01-0{i+1}T10:00:05",
                    "level": "INFO",
                    "message": "Metrics",
                    "metrics": {"runtime_seconds": 100 + i * 10},
                },
            ]
            
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
            
            files.append(Path(f.name))
    
    return files


def test_log_parser_parse_lines(sample_log_file):
    """Test parsing log lines."""
    parser = LogParser(sample_log_file)
    
    entries = list(parser.parse_lines())
    assert len(entries) == 9
    
    # Check first entry
    assert entries[0]["level"] == "INFO"
    assert entries[0]["message"] == "Starting experiment"
    assert entries[0]["event_type"] == "STARTUP"


def test_log_parser_filter_by_level(sample_log_file):
    """Test filtering by log level."""
    parser = LogParser(sample_log_file)
    
    # Filter INFO
    info_entries = parser.filter_by_level("INFO")
    assert len(info_entries) == 7
    
    # Filter WARNING
    warning_entries = parser.filter_by_level("WARNING")
    assert len(warning_entries) == 1
    assert warning_entries[0]["message"] == "QA issue detected"
    
    # Filter ERROR
    error_entries = parser.filter_by_level("ERROR")
    assert len(error_entries) == 1
    assert error_entries[0]["exception_type"] == "ValueError"


def test_log_parser_filter_by_event_type(sample_log_file):
    """Test filtering by event type."""
    parser = LogParser(sample_log_file)
    
    # Filter CONFIG
    config_entries = parser.filter_by_event_type("CONFIG")
    assert len(config_entries) == 1
    assert config_entries[0]["config_hash"] == "abc123"
    
    # Filter QA_ISSUE
    qa_entries = parser.filter_by_event_type("QA_ISSUE")
    assert len(qa_entries) == 1
    assert qa_entries[0]["issue_type"] == "nan"


def test_log_parser_filter_by_time_range(sample_log_file):
    """Test filtering by time range."""
    parser = LogParser(sample_log_file)
    
    # Filter first 10 seconds
    start = datetime(2024, 1, 1, 10, 0, 0)
    end = datetime(2024, 1, 1, 10, 0, 10)
    
    entries = parser.filter_by_time_range(start, end)
    assert len(entries) == 5  # Should get first 5 entries


def test_log_parser_search(sample_log_file):
    """Test regex search."""
    parser = LogParser(sample_log_file)
    
    # Search for "complete"
    results = parser.search("complete")
    assert len(results) == 2  # "Clustering complete" and "Evaluation complete"
    
    # Search in specific field
    results = parser.search("gmm", field="algorithm")
    assert len(results) == 1
    assert results[0]["k_selected"] == 4


def test_log_parser_get_metrics_timeline(sample_log_file):
    """Test extracting metrics timeline."""
    parser = LogParser(sample_log_file)
    
    df = parser.get_metrics_timeline()
    assert not df.empty
    assert len(df) == 1  # One metrics entry
    
    # Check columns
    assert "counter_rows_processed" in df.columns
    assert "counter_rows_interpolated" in df.columns
    assert "timer_training" in df.columns
    assert "runtime" in df.columns
    
    # Check values
    assert df["counter_rows_processed"].iloc[0] == 1000
    assert df["timer_training"].iloc[0] == 120.5


def test_log_parser_get_exceptions(sample_log_file):
    """Test extracting exceptions."""
    parser = LogParser(sample_log_file)
    
    exceptions = parser.get_exceptions()
    assert len(exceptions) == 1
    assert exceptions[0]["exception_type"] == "ValueError"
    assert exceptions[0]["exception_message"] == "Invalid value"


def test_log_parser_get_qa_issues(sample_log_file):
    """Test extracting QA issues."""
    parser = LogParser(sample_log_file)
    
    qa_issues = parser.get_qa_issues()
    assert qa_issues == {"nan": 5}


def test_metrics_aggregator(multiple_log_files):
    """Test metrics aggregation across runs."""
    aggregator = MetricsAggregator()
    
    # Add all runs
    for i, log_file in enumerate(multiple_log_files):
        aggregator.add_run(log_file, f"run_{i}")
    
    # Compare runs
    df = aggregator.compare_runs()
    assert len(df) == 3
    assert "accuracy" in df.columns
    assert "macro_f1" in df.columns
    
    # Check sorting (should be by accuracy)
    assert df.iloc[0]["accuracy"] == 0.87  # Highest accuracy first
    
    # Get summary
    summary = aggregator.summarize()
    assert summary["num_runs"] == 3
    assert summary["accuracy"]["mean"] == pytest.approx(0.85, rel=0.01)
    assert summary["accuracy"]["max"] == 0.87
    assert summary["accuracy"]["min"] == 0.83


def test_log_reporter_summary(sample_log_file):
    """Test generating summary report."""
    reporter = LogReporter(sample_log_file)
    
    report = reporter.generate_summary_report()
    
    # Check report contains expected sections
    assert "Log Level Summary:" in report
    assert "Event Type Summary:" in report
    assert "QA Issues:" in report
    assert "Exceptions:" in report
    assert "Metrics Summary:" in report
    
    # Check counts
    assert "INFO" in report
    assert "WARNING" in report
    assert "ERROR" in report


def test_log_reporter_error_report(sample_log_file):
    """Test generating error report."""
    reporter = LogReporter(sample_log_file)
    
    report = reporter.generate_error_report()
    
    # Check report contains errors and exceptions
    assert "Errors: 1" in report
    assert "Exceptions: 1" in report
    assert "ValueError" in report
    assert "Invalid value" in report


def test_log_reporter_export_csv(sample_log_file):
    """Test exporting to CSV."""
    reporter = LogReporter(sample_log_file)
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = Path(f.name)
    
    # Export all entries
    reporter.export_to_csv(output_path)
    assert output_path.exists()
    
    # Read and verify
    import pandas as pd
    df = pd.read_csv(output_path)
    assert len(df) == 9
    assert "timestamp" in df.columns
    assert "level" in df.columns
    assert "message" in df.columns
    
    # Clean up
    output_path.unlink()


def test_convenience_functions(sample_log_file, capsys):
    """Test convenience functions."""
    # Test analyze_log_file
    analyze_log_file(sample_log_file)
    captured = capsys.readouterr()
    assert "LOG ANALYSIS REPORT" in captured.out
    
    # Test find_errors
    errors = find_errors(sample_log_file)
    assert len(errors) == 1
    assert errors[0]["level"] == "ERROR"
    
    # Test grep_logs
    results = grep_logs(sample_log_file, "complete")
    assert len(results) == 2
    
    results = grep_logs(sample_log_file, "gmm", field="algorithm")
    assert len(results) == 1


def test_compare_experiments(multiple_log_files):
    """Test comparing experiments."""
    # Create temp directory with log files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Move files to temp directory
        for log_file in multiple_log_files:
            target = tmpdir / log_file.name
            log_file.rename(target)
        
        # Compare experiments
        df = compare_experiments(tmpdir)
        assert len(df) == 3
        assert "accuracy" in df.columns
        assert df["accuracy"].max() == 0.87


def test_log_parser_handles_malformed_json(tmp_path):
    """Test parser handles malformed JSON gracefully."""
    log_file = tmp_path / "malformed.log"
    
    with open(log_file, "w") as f:
        f.write('{"level": "INFO", "message": "Valid JSON"}\n')
        f.write('This is not JSON\n')
        f.write('{"level": "ERROR", "message": "Another valid entry"}\n')
    
    parser = LogParser(log_file)
    entries = list(parser.parse_lines())
    
    # Should only get valid JSON entries
    assert len(entries) == 2
    assert entries[0]["level"] == "INFO"
    assert entries[1]["level"] == "ERROR"


def test_log_parser_directory_parsing(tmp_path):
    """Test parsing all log files in a directory."""
    # Create multiple log files
    for i in range(3):
        log_file = tmp_path / f"test_{i}.log"
        with open(log_file, "w") as f:
            entry = {"level": "INFO", "message": f"Log {i}"}
            f.write(json.dumps(entry) + "\n")
    
    # Parse directory
    parser = LogParser(tmp_path)
    entries = list(parser.parse_lines())
    
    assert len(entries) == 3
    messages = [e["message"] for e in entries]
    assert "Log 0" in messages
    assert "Log 1" in messages
    assert "Log 2" in messages