#!/usr/bin/env python3
"""Command-line interface for log analysis.

Provides CLI commands for parsing, searching, and analyzing JSON logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from conv2d.logging.analysis import (
    LogParser,
    LogReporter,
    MetricsAggregator,
    analyze_log_file,
    compare_experiments,
    find_errors,
    grep_logs,
)


class LogCLI:
    """CLI for log analysis."""
    
    def __init__(self):
        """Initialize CLI."""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with subcommands."""
        parser = argparse.ArgumentParser(
            prog="conv2d-logs",
            description="Conv2d log analysis tools",
        )
        
        subparsers = parser.add_subparsers(
            dest="command",
            help="Available commands",
        )
        
        # Summary command
        summary_parser = subparsers.add_parser(
            "summary",
            help="Generate log summary report",
        )
        summary_parser.add_argument(
            "log_path",
            help="Path to log file",
        )
        
        # Errors command
        errors_parser = subparsers.add_parser(
            "errors",
            help="Find and display errors",
        )
        errors_parser.add_argument(
            "log_path",
            help="Path to log file",
        )
        errors_parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Show full error details",
        )
        
        # Grep command
        grep_parser = subparsers.add_parser(
            "grep",
            help="Search logs with regex",
        )
        grep_parser.add_argument(
            "pattern",
            help="Search pattern (regex)",
        )
        grep_parser.add_argument(
            "log_path",
            help="Path to log file",
        )
        grep_parser.add_argument(
            "--field", "-f",
            help="Field to search in",
        )
        grep_parser.add_argument(
            "--json", "-j",
            action="store_true",
            help="Output as JSON",
        )
        
        # Filter command
        filter_parser = subparsers.add_parser(
            "filter",
            help="Filter logs by criteria",
        )
        filter_parser.add_argument(
            "log_path",
            help="Path to log file",
        )
        filter_parser.add_argument(
            "--level", "-l",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Filter by log level",
        )
        filter_parser.add_argument(
            "--event", "-e",
            help="Filter by event type",
        )
        filter_parser.add_argument(
            "--start",
            help="Start time (ISO format)",
        )
        filter_parser.add_argument(
            "--end",
            help="End time (ISO format)",
        )
        filter_parser.add_argument(
            "--json", "-j",
            action="store_true",
            help="Output as JSON",
        )
        
        # Compare command
        compare_parser = subparsers.add_parser(
            "compare",
            help="Compare multiple experiment runs",
        )
        compare_parser.add_argument(
            "log_dir",
            help="Directory containing log files",
        )
        compare_parser.add_argument(
            "--output", "-o",
            help="Output CSV file",
        )
        
        # Metrics command
        metrics_parser = subparsers.add_parser(
            "metrics",
            help="Extract metrics timeline",
        )
        metrics_parser.add_argument(
            "log_path",
            help="Path to log file",
        )
        metrics_parser.add_argument(
            "--output", "-o",
            help="Output CSV file",
        )
        
        # Export command
        export_parser = subparsers.add_parser(
            "export",
            help="Export logs to CSV",
        )
        export_parser.add_argument(
            "log_path",
            help="Path to log file",
        )
        export_parser.add_argument(
            "output",
            help="Output CSV file",
        )
        export_parser.add_argument(
            "--level", "-l",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Filter by log level",
        )
        
        # QA command
        qa_parser = subparsers.add_parser(
            "qa",
            help="Show QA issues",
        )
        qa_parser.add_argument(
            "log_path",
            help="Path to log file",
        )
        
        # Tail command
        tail_parser = subparsers.add_parser(
            "tail",
            help="Follow log file (like tail -f)",
        )
        tail_parser.add_argument(
            "log_path",
            help="Path to log file",
        )
        tail_parser.add_argument(
            "--lines", "-n",
            type=int,
            default=10,
            help="Number of lines to show initially",
        )
        
        return parser
    
    def run(self, args: Optional[list] = None) -> int:
        """Run CLI with given arguments.
        
        Args:
            args: Command-line arguments (None for sys.argv)
            
        Returns:
            Exit code
        """
        parsed_args = self.parser.parse_args(args)
        
        if parsed_args.command == "summary":
            return self._run_summary(parsed_args)
        elif parsed_args.command == "errors":
            return self._run_errors(parsed_args)
        elif parsed_args.command == "grep":
            return self._run_grep(parsed_args)
        elif parsed_args.command == "filter":
            return self._run_filter(parsed_args)
        elif parsed_args.command == "compare":
            return self._run_compare(parsed_args)
        elif parsed_args.command == "metrics":
            return self._run_metrics(parsed_args)
        elif parsed_args.command == "export":
            return self._run_export(parsed_args)
        elif parsed_args.command == "qa":
            return self._run_qa(parsed_args)
        elif parsed_args.command == "tail":
            return self._run_tail(parsed_args)
        else:
            self.parser.print_help()
            return 1
    
    def _run_summary(self, args: argparse.Namespace) -> int:
        """Run summary command."""
        try:
            reporter = LogReporter(args.log_path)
            print(reporter.generate_summary_report())
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _run_errors(self, args: argparse.Namespace) -> int:
        """Run errors command."""
        try:
            if args.verbose:
                reporter = LogReporter(args.log_path)
                print(reporter.generate_error_report())
            else:
                errors = find_errors(args.log_path)
                print(f"Found {len(errors)} errors/exceptions")
                
                for entry in errors[:20]:  # Show first 20
                    timestamp = entry.get("timestamp", "")
                    level = entry.get("level", "")
                    message = entry.get("message", entry.get("exception_message", ""))
                    print(f"[{timestamp}] {level}: {message}")
            
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _run_grep(self, args: argparse.Namespace) -> int:
        """Run grep command."""
        try:
            results = grep_logs(args.log_path, args.pattern, args.field)
            
            if args.json:
                for entry in results:
                    print(json.dumps(entry))
            else:
                print(f"Found {len(results)} matches")
                for entry in results:
                    timestamp = entry.get("timestamp", "")
                    level = entry.get("level", "")
                    message = entry.get("message", "")
                    print(f"[{timestamp}] {level}: {message}")
            
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _run_filter(self, args: argparse.Namespace) -> int:
        """Run filter command."""
        try:
            parser = LogParser(args.log_path)
            
            # Apply filters
            if args.level:
                results = parser.filter_by_level(args.level)
            elif args.event:
                results = parser.filter_by_event_type(args.event)
            elif args.start or args.end:
                start = datetime.fromisoformat(args.start) if args.start else None
                end = datetime.fromisoformat(args.end) if args.end else None
                results = parser.filter_by_time_range(start, end)
            else:
                print("No filter specified", file=sys.stderr)
                return 1
            
            # Output results
            if args.json:
                for entry in results:
                    print(json.dumps(entry))
            else:
                print(f"Found {len(results)} entries")
                for entry in results[:100]:  # Limit output
                    timestamp = entry.get("timestamp", "")
                    level = entry.get("level", "")
                    message = entry.get("message", "")
                    print(f"[{timestamp}] {level}: {message}")
            
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _run_compare(self, args: argparse.Namespace) -> int:
        """Run compare command."""
        try:
            df = compare_experiments(args.log_dir)
            
            if df.empty:
                print("No experiments found")
                return 1
            
            if args.output:
                df.to_csv(args.output, index=False)
                print(f"Comparison saved to {args.output}")
            else:
                print(df.to_string())
            
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _run_metrics(self, args: argparse.Namespace) -> int:
        """Run metrics command."""
        try:
            parser = LogParser(args.log_path)
            df = parser.get_metrics_timeline()
            
            if df.empty:
                print("No metrics found")
                return 1
            
            if args.output:
                df.to_csv(args.output, index=False)
                print(f"Metrics saved to {args.output}")
            else:
                print(df.to_string())
            
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _run_export(self, args: argparse.Namespace) -> int:
        """Run export command."""
        try:
            reporter = LogReporter(args.log_path)
            
            # Create filter function if needed
            filter_func = None
            if args.level:
                level = args.level
                filter_func = lambda e: e.get("level") == level
            
            reporter.export_to_csv(args.output, filter_func)
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _run_qa(self, args: argparse.Namespace) -> int:
        """Run QA command."""
        try:
            parser = LogParser(args.log_path)
            qa_issues = parser.get_qa_issues()
            
            if not qa_issues:
                print("No QA issues found")
                return 0
            
            print("QA Issues Summary:")
            print("-" * 40)
            total = 0
            for issue_type, count in sorted(qa_issues.items()):
                print(f"  {issue_type:20s}: {count:6d}")
                total += count
            print("-" * 40)
            print(f"  {'TOTAL':20s}: {total:6d}")
            
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _run_tail(self, args: argparse.Namespace) -> int:
        """Run tail command (follow log file)."""
        import time
        
        try:
            log_path = Path(args.log_path)
            
            # Show initial lines
            parser = LogParser(log_path)
            entries = list(parser.parse_lines())
            
            # Show last N entries
            for entry in entries[-args.lines:]:
                self._print_entry(entry)
            
            # Follow file
            with open(log_path) as f:
                # Move to end
                f.seek(0, 2)
                
                print(f"\n--- Following {log_path} ---")
                
                while True:
                    line = f.readline()
                    if line:
                        try:
                            entry = json.loads(line.strip())
                            self._print_entry(entry)
                        except json.JSONDecodeError:
                            # Print raw line if not JSON
                            print(line.strip())
                    else:
                        time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n--- Stopped ---")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _print_entry(self, entry: dict) -> None:
        """Print a log entry in readable format."""
        timestamp = entry.get("timestamp", "")
        level = entry.get("level", "UNKNOWN")
        message = entry.get("message", "")
        
        # Color coding for levels
        colors = {
            "DEBUG": "\033[36m",    # Cyan
            "INFO": "\033[0m",       # Default
            "WARNING": "\033[33m",   # Yellow
            "ERROR": "\033[31m",     # Red
            "CRITICAL": "\033[91m",  # Bright Red
        }
        
        color = colors.get(level, "\033[0m")
        reset = "\033[0m"
        
        # Format output
        output = f"{color}[{timestamp}] {level:8s}{reset}: {message}"
        
        # Add event type if present
        if "event_type" in entry:
            output += f" [{entry['event_type']}]"
        
        # Add key metrics if present
        if "metrics" in entry:
            metrics = entry["metrics"]
            if "counters" in metrics and metrics["counters"]:
                counts = ", ".join(f"{k}={v}" for k, v in metrics["counters"].items())
                output += f" (counters: {counts})"
        
        print(output)


def main():
    """Main entry point for CLI."""
    cli = LogCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()