#!/usr/bin/env python3
"""Command-line interface for Conv2d evaluation and training.

Provides CLI commands for model evaluation, training, and analysis
with configuration-based experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from conv2d.metrics import BundleGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Conv2dCLI:
    """Main CLI for Conv2d framework."""
    
    def __init__(self):
        """Initialize CLI."""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with subcommands."""
        parser = argparse.ArgumentParser(
            prog="conv2d",
            description="Conv2d behavioral analysis framework",
        )
        
        subparsers = parser.add_subparsers(
            dest="command",
            help="Available commands",
        )
        
        # Evaluation command
        eval_parser = subparsers.add_parser(
            "eval",
            help="Evaluate a trained model",
        )
        eval_parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to experiment config (e.g., conf/exp/dogs.yaml)",
        )
        eval_parser.add_argument(
            "--split",
            type=str,
            default="val",
            choices=["train", "val", "test"],
            help="Dataset split to evaluate",
        )
        eval_parser.add_argument(
            "--checkpoint",
            type=str,
            help="Path to model checkpoint",
        )
        eval_parser.add_argument(
            "--output",
            type=str,
            default="reports",
            help="Output directory for evaluation bundle",
        )
        eval_parser.add_argument(
            "--save-predictions",
            action="store_true",
            help="Save raw predictions to bundle",
        )
        
        # Training command
        train_parser = subparsers.add_parser(
            "train",
            help="Train a model",
        )
        train_parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to training config",
        )
        train_parser.add_argument(
            "--resume",
            type=str,
            help="Resume from checkpoint",
        )
        
        # Analysis command
        analyze_parser = subparsers.add_parser(
            "analyze",
            help="Analyze evaluation results",
        )
        analyze_parser.add_argument(
            "bundle_dir",
            type=str,
            help="Path to evaluation bundle directory",
        )
        analyze_parser.add_argument(
            "--compare",
            type=str,
            nargs="+",
            help="Additional bundles to compare",
        )
        
        # Add packaging commands
        from .cli_pack import add_pack_commands
        add_pack_commands(subparsers)
        
        return parser
    
    def run(self, args: Optional[list] = None) -> int:
        """Run CLI with given arguments.
        
        Args:
            args: Command-line arguments (None for sys.argv)
            
        Returns:
            Exit code (0 for success)
        """
        parsed_args = self.parser.parse_args(args)
        
        if parsed_args.command == "eval":
            return self._run_evaluation(parsed_args)
        elif parsed_args.command == "train":
            return self._run_training(parsed_args)
        elif parsed_args.command == "analyze":
            return self._run_analysis(parsed_args)
        elif parsed_args.command == "pack":
            from .cli_pack import run_pack_command
            return run_pack_command(parsed_args)
        else:
            self.parser.print_help()
            return 1
    
    def _run_evaluation(self, args: argparse.Namespace) -> int:
        """Run model evaluation.
        
        Args:
            args: Parsed arguments
            
        Returns:
            Exit code
        """
        logger.info(f"Running evaluation with config: {args.config}")
        
        try:
            # Load configuration
            config = self._load_config(args.config)
            
            # Load model and data
            model, dataloader = self._load_model_and_data(
                config, args.checkpoint, args.split
            )
            
            # Run evaluation
            y_true, y_pred, y_prob, codes, X_data = self._evaluate_model(
                model, dataloader, config
            )
            
            # Generate evaluation bundle
            generator = BundleGenerator(
                output_base=Path(args.output),
                save_raw_predictions=args.save_predictions,
            )
            
            bundle = generator.generate(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                codes=codes,
                config=config,
                exp_name=Path(args.config).stem,
                X_data=X_data,
            )
            
            # Print summary
            print("\n" + "="*60)
            print(bundle.summary())
            print("="*60)
            
            return 0
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return 1
    
    def _run_training(self, args: argparse.Namespace) -> int:
        """Run model training.
        
        Args:
            args: Parsed arguments
            
        Returns:
            Exit code
        """
        logger.info(f"Running training with config: {args.config}")
        
        try:
            # Load configuration
            config = self._load_config(args.config)
            
            # TODO: Implement actual training logic
            logger.warning("Training command not fully implemented yet")
            
            # Placeholder for training
            print(f"Would train with config: {config.get('model', {}).get('name', 'unknown')}")
            print(f"Epochs: {config.get('training', {}).get('epochs', 100)}")
            print(f"Batch size: {config.get('training', {}).get('batch_size', 32)}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return 1
    
    def _run_analysis(self, args: argparse.Namespace) -> int:
        """Run results analysis.
        
        Args:
            args: Parsed arguments
            
        Returns:
            Exit code
        """
        logger.info(f"Analyzing bundle: {args.bundle_dir}")
        
        try:
            bundle_dir = Path(args.bundle_dir)
            
            # Load metrics
            metrics_path = bundle_dir / "metrics.json"
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            # Load calibration if available
            calib_path = bundle_dir / "calibration.json"
            calibration = {}
            if calib_path.exists():
                with open(calib_path) as f:
                    calibration = json.load(f)
            
            # Print analysis
            print("\n" + "="*60)
            print(f"Analysis of: {bundle_dir.name}")
            print("="*60)
            
            print("\nKey Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Macro-F1: {metrics['macro_f1']:.3f}")
            print(f"  ECE: {metrics.get('ece', 0):.3f}")
            print(f"  Coverage: {metrics['coverage']:.2f}")
            print(f"  Motifs: {metrics['motif_count']}")
            
            # Compare with other bundles if requested
            if args.compare:
                print("\nComparison with other bundles:")
                self._compare_bundles(bundle_dir, args.compare)
            
            return 0
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return 1
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(path) as f:
            if path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return config
    
    def _load_model_and_data(
        self,
        config: Dict[str, Any],
        checkpoint_path: Optional[str],
        split: str,
    ):
        """Load model and dataset.
        
        This is a placeholder - actual implementation would load
        the specific model and dataset based on config.
        """
        # Simulate loading model
        logger.info(f"Loading model: {config.get('model', {}).get('name', 'unknown')}")
        
        if checkpoint_path:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Simulate data loading
        logger.info(f"Loading {split} split of {config.get('data', {}).get('name', 'unknown')}")
        
        # Return dummy objects for now
        class DummyModel:
            def eval(self):
                return self
        
        class DummyDataLoader:
            def __iter__(self):
                # Generate some dummy data
                for _ in range(10):
                    X = torch.randn(32, 9, 2, 100)
                    y = torch.randint(0, 4, (32,))
                    yield X, y
        
        return DummyModel(), DummyDataLoader()
    
    def _evaluate_model(self, model, dataloader, config):
        """Run model evaluation.
        
        This is a simplified version - actual implementation
        would properly run the model.
        """
        logger.info("Running model evaluation...")
        
        # Collect predictions (dummy for now)
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        all_codes = []
        all_X = []
        
        for X, y in dataloader:
            # Simulate predictions
            batch_size = y.shape[0]
            n_classes = 4
            
            y_prob = torch.softmax(torch.randn(batch_size, n_classes), dim=1)
            y_pred = y_prob.argmax(dim=1)
            codes = torch.randint(0, 240, (batch_size, 100))
            
            all_y_true.append(y.numpy())
            all_y_pred.append(y_pred.numpy())
            all_y_prob.append(y_prob.numpy())
            all_codes.append(codes.numpy())
            all_X.append(X.numpy())
        
        # Concatenate
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        y_prob = np.concatenate(all_y_prob)
        codes = np.concatenate(all_codes)
        X_data = np.concatenate(all_X)
        
        return y_true, y_pred, y_prob, codes, X_data
    
    def _compare_bundles(self, main_bundle: Path, other_bundles: list) -> None:
        """Compare multiple evaluation bundles."""
        # Load all metrics
        all_metrics = {}
        
        # Load main bundle
        with open(main_bundle / "metrics.json") as f:
            all_metrics[main_bundle.name] = json.load(f)
        
        # Load comparison bundles
        for bundle_path in other_bundles:
            bundle_dir = Path(bundle_path)
            with open(bundle_dir / "metrics.json") as f:
                all_metrics[bundle_dir.name] = json.load(f)
        
        # Create comparison table
        metrics_to_compare = ["accuracy", "macro_f1", "ece", "coverage"]
        
        print("\n| Bundle | " + " | ".join(metrics_to_compare) + " |")
        print("|" + "-"*8 + "|" + "|".join(["-"*10]*len(metrics_to_compare)) + "|")
        
        for name, metrics in all_metrics.items():
            values = []
            for metric in metrics_to_compare:
                value = metrics.get(metric, 0)
                if isinstance(value, float):
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            
            # Truncate name if too long
            if len(name) > 30:
                name = name[:27] + "..."
            
            print(f"| {name[:30]:30s} | " + " | ".join(f"{v:8s}" for v in values) + " |")


def main():
    """Main entry point for CLI."""
    cli = Conv2dCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()