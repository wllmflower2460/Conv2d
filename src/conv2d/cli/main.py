#!/usr/bin/env python3
"""Main CLI entry point for Conv2d-FSQ-HSMM."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from conv2d import __version__
from conv2d.cli.evaluate import evaluate_command
from conv2d.cli.export import export_command
from conv2d.cli.train import train_command
from conv2d.utils import setup_logger


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="conv2d",
        description="Conv2d-FSQ-HSMM: Behavioral synchrony analysis framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  conv2d train --config configs/training_config.yaml
  conv2d evaluate --checkpoint models/best_model.pth
  conv2d export --checkpoint models/best_model.pth --format onnx
  conv2d --version
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated: -v, -vv, -vvv)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        help="Command to run",
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a Conv2d-FSQ-HSMM model",
        description="Train a behavioral synchrony model",
    )
    train_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to training configuration file",
    )
    train_parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume training from",
    )
    train_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("outputs"),
        help="Output directory for checkpoints and logs",
    )
    train_parser.set_defaults(func=train_command)

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model",
        description="Evaluate model performance on test data",
    )
    eval_parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--data",
        "-d",
        type=Path,
        help="Path to test data directory",
    )
    eval_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for evaluation results",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    eval_parser.set_defaults(func=evaluate_command)

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export model to different formats",
        description="Export trained model for deployment",
    )
    export_parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["onnx", "coreml", "hailo", "torchscript"],
        default="onnx",
        help="Export format",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (auto-generated if not specified)",
    )
    export_parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply optimizations for deployment",
    )
    export_parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization (format-specific)",
    )
    export_parser.set_defaults(func=export_command)

    return parser


def setup_logging(verbose: int, quiet: bool, log_file: Optional[Path]) -> None:
    """Set up logging configuration."""
    if quiet:
        level = logging.ERROR
    elif verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    setup_logger(level=level, log_file=log_file)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Set up logging
    setup_logging(args.verbose, args.quiet, args.log_file)

    # If no command specified, print help
    if not args.command:
        parser.print_help()
        return 0

    # Execute the command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logging.error("Interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=args.verbose > 1)
        return 1


if __name__ == "__main__":
    sys.exit(main())