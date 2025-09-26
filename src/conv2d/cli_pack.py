#!/usr/bin/env python3
"""CLI extension for artifact packaging commands.

Adds packaging commands to the main CLI:
- conv2d pack: Create deployment bundle
- conv2d pack --verify: Validate bundle
- conv2d pack --list: List bundles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from conv2d.packaging import ArtifactBundler, BundleValidator, DeploymentBundle


def add_pack_commands(subparsers):
    """Add packaging commands to CLI parser.
    
    Args:
        subparsers: Subparser object from main CLI
    """
    # Pack command
    pack_parser = subparsers.add_parser(
        "pack",
        help="Create and manage deployment packages",
    )
    
    pack_subparsers = pack_parser.add_subparsers(
        dest="pack_command",
        help="Packaging operations",
    )
    
    # Create bundle
    create_parser = pack_subparsers.add_parser(
        "create",
        help="Create deployment bundle",
    )
    create_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config",
    )
    create_parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model (PyTorch checkpoint)",
    )
    create_parser.add_argument(
        "--metrics",
        type=str,
        help="Path to metrics JSON file",
    )
    create_parser.add_argument(
        "--labels",
        type=str,
        help="Path to label mapping JSON file",
    )
    create_parser.add_argument(
        "--name",
        type=str,
        default="experiment",
        help="Experiment name",
    )
    create_parser.add_argument(
        "--output",
        type=str,
        default="artifacts",
        help="Output directory for bundles",
    )
    
    # Verify bundle
    verify_parser = pack_subparsers.add_parser(
        "verify",
        help="Validate deployment bundle",
    )
    verify_parser.add_argument(
        "bundle_path",
        help="Path to bundle directory or bundle hash",
    )
    verify_parser.add_argument(
        "--target",
        choices=["ios", "hailo", "edge", "onnx"],
        help="Validate for specific target platform",
    )
    verify_parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts directory (if using bundle hash)",
    )
    
    # List bundles
    list_parser = pack_subparsers.add_parser(
        "list",
        help="List deployment bundles",
    )
    list_parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts directory",
    )
    list_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed bundle information",
    )
    
    # Clean bundles
    clean_parser = pack_subparsers.add_parser(
        "clean",
        help="Clean old deployment bundles",
    )
    clean_parser.add_argument(
        "--keep",
        type=int,
        default=10,
        help="Number of recent bundles to keep",
    )
    clean_parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts directory",
    )
    
    # Archive bundle
    archive_parser = pack_subparsers.add_parser(
        "archive",
        help="Archive deployment bundle",
    )
    archive_parser.add_argument(
        "bundle_path",
        help="Path to bundle directory or bundle hash",
    )
    archive_parser.add_argument(
        "--format",
        choices=["zip", "tar", "tar.gz"],
        default="zip",
        help="Archive format",
    )
    archive_parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts directory (if using bundle hash)",
    )


def run_pack_command(args: argparse.Namespace) -> int:
    """Run packaging command.
    
    Args:
        args: Parsed command arguments
        
    Returns:
        Exit code
    """
    if args.pack_command == "create":
        return _run_create(args)
    elif args.pack_command == "verify":
        return _run_verify(args)
    elif args.pack_command == "list":
        return _run_list(args)
    elif args.pack_command == "clean":
        return _run_clean(args)
    elif args.pack_command == "archive":
        return _run_archive(args)
    else:
        print("No pack subcommand specified. Use --help for available commands.")
        return 1


def _run_create(args: argparse.Namespace) -> int:
    """Create deployment bundle."""
    try:
        import yaml
        import json
        
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            return 1
        
        with open(config_path) as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # Load optional files
        model_path = Path(args.model) if args.model else None
        if model_path and not model_path.exists():
            print(f"Warning: Model file not found: {model_path}")
            model_path = None
        
        metrics = None
        if args.metrics:
            metrics_path = Path(args.metrics)
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
            else:
                print(f"Warning: Metrics file not found: {metrics_path}")
        
        label_mapping = None
        if args.labels:
            labels_path = Path(args.labels)
            if labels_path.exists():
                with open(labels_path) as f:
                    label_data = json.load(f)
                    if "mapping" in label_data:
                        # Convert string keys back to int
                        label_mapping = {int(k): v for k, v in label_data["mapping"].items()}
                    else:
                        label_mapping = {int(k): v for k, v in label_data.items()}
            else:
                print(f"Warning: Labels file not found: {labels_path}")
        
        # Create bundle
        bundler = ArtifactBundler(output_base=args.output)
        
        print(f"Creating deployment bundle...")
        print(f"  Config: {config_path}")
        if model_path:
            print(f"  Model: {model_path}")
        if metrics:
            print(f"  Metrics: {len(metrics)} entries")
        if label_mapping:
            print(f"  Labels: {len(label_mapping)} behaviors")
        
        bundle = bundler.create_bundle(
            config=config,
            model_path=model_path,
            metrics=metrics,
            label_mapping=label_mapping,
            exp_name=args.name,
        )
        
        print(f"\n✅ Bundle created successfully!")
        print(f"Hash: {bundle.exp_hash}")
        print(f"Location: {bundle.bundle_dir}")
        
        # Show summary
        print(f"\n{bundle.summary()}")
        
        return 0
        
    except Exception as e:
        print(f"Error creating bundle: {e}")
        return 1


def _run_verify(args: argparse.Namespace) -> int:
    """Verify deployment bundle."""
    try:
        # Resolve bundle path
        bundle = _resolve_bundle_path(args.bundle_path, args.artifacts_dir)
        if not bundle:
            return 1
        
        print(f"Validating bundle: {bundle.exp_hash}")
        print(f"Location: {bundle.bundle_dir}")
        
        # Validate bundle
        validator = BundleValidator()
        
        if args.target:
            result = validator.validate_for_target(bundle, args.target)
            print(f"Target: {args.target.upper()}")
        else:
            result = validator.validate(bundle)
        
        # Print results
        print(f"\n{result.summary()}")
        
        return 0 if result.passed else 1
        
    except Exception as e:
        print(f"Error validating bundle: {e}")
        return 1


def _run_list(args: argparse.Namespace) -> int:
    """List deployment bundles."""
    try:
        bundler = ArtifactBundler(output_base=args.artifacts_dir)
        bundles = bundler.list_bundles()
        
        if not bundles:
            print(f"No bundles found in {args.artifacts_dir}")
            return 0
        
        print(f"Deployment Bundles ({len(bundles)}):")
        print("=" * 80)
        
        for bundle in bundles:
            # Get creation time
            try:
                created = bundle.bundle_dir.stat().st_mtime
                from datetime import datetime
                created_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
            except Exception:
                created_str = "Unknown"
            
            print(f"Hash: {bundle.exp_hash}")
            print(f"Created: {created_str}")
            print(f"Location: {bundle.bundle_dir}")
            
            if args.verbose:
                manifest = bundle.get_manifest()
                
                # Show file summary
                files_exist = sum(1 for f in manifest["files"].values() if f.get("exists", False))
                total_files = len(manifest["files"])
                print(f"Files: {files_exist}/{total_files}")
                
                # Show total size
                total_size = sum(
                    f.get("size_bytes", 0) 
                    for f in manifest["files"].values() 
                    if f.get("exists", False)
                )
                size_mb = total_size / (1024 * 1024)
                print(f"Size: {size_mb:.1f} MB")
                
                # Show available models
                models = []
                if bundle.model_onnx.exists():
                    models.append("ONNX")
                if bundle.model_coreml.exists():
                    models.append("CoreML")
                if bundle.model_hailo.exists():
                    models.append("Hailo")
                print(f"Models: {', '.join(models) if models else 'None'}")
            
            print("-" * 80)
        
        return 0
        
    except Exception as e:
        print(f"Error listing bundles: {e}")
        return 1


def _run_clean(args: argparse.Namespace) -> int:
    """Clean old deployment bundles."""
    try:
        bundler = ArtifactBundler(output_base=args.artifacts_dir)
        
        print(f"Cleaning old bundles (keeping {args.keep} most recent)...")
        
        removed = bundler.clean_old_bundles(keep_recent=args.keep)
        
        if removed > 0:
            print(f"✅ Removed {removed} old bundles")
        else:
            print("No bundles to remove")
        
        return 0
        
    except Exception as e:
        print(f"Error cleaning bundles: {e}")
        return 1


def _run_archive(args: argparse.Namespace) -> int:
    """Archive deployment bundle."""
    try:
        # Resolve bundle path
        bundle = _resolve_bundle_path(args.bundle_path, args.artifacts_dir)
        if not bundle:
            return 1
        
        print(f"Archiving bundle: {bundle.exp_hash}")
        print(f"Format: {args.format}")
        
        bundler = ArtifactBundler()
        archive_path = bundler.archive_bundle(bundle, args.format)
        
        print(f"✅ Archive created: {archive_path}")
        
        # Show archive size
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB")
        
        return 0
        
    except Exception as e:
        print(f"Error archiving bundle: {e}")
        return 1


def _resolve_bundle_path(
    bundle_path: str,
    artifacts_dir: str,
) -> Optional[DeploymentBundle]:
    """Resolve bundle path from directory or hash.
    
    Args:
        bundle_path: Bundle directory or hash
        artifacts_dir: Artifacts directory
        
    Returns:
        DeploymentBundle if found, None otherwise
    """
    path = Path(bundle_path)
    
    # Check if it's a directory path
    if path.exists() and path.is_dir():
        exp_hash = path.name
        return DeploymentBundle(path, exp_hash)
    
    # Check if it's a hash in artifacts directory
    artifacts_path = Path(artifacts_dir)
    if artifacts_path.exists():
        bundle_dir = artifacts_path / bundle_path
        if bundle_dir.exists() and bundle_dir.is_dir():
            return DeploymentBundle(bundle_dir, bundle_path)
    
    # Try as full hash search
    if artifacts_path.exists():
        for candidate in artifacts_path.iterdir():
            if candidate.is_dir() and candidate.name.startswith(bundle_path):
                return DeploymentBundle(candidate, candidate.name)
    
    print(f"Error: Bundle not found: {bundle_path}")
    print(f"Searched in: {artifacts_path}")
    return None


def main():
    """Standalone entry point for packaging CLI."""
    parser = argparse.ArgumentParser(
        prog="conv2d-pack",
        description="Conv2d deployment packaging tools",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    add_pack_commands(subparsers)
    
    args = parser.parse_args()
    
    if args.command == "pack":
        sys.exit(run_pack_command(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()