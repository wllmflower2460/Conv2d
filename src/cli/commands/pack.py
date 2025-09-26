"""Deployment packaging command implementation with standardized artifact bundles."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import os
import yaml

from ...conv2d.packaging import ArtifactBundle


def create_bundle(
    config_path: Path,
    metrics_path: Path,
    model_path: Path,
    model_format: str,
    output_dir: Path = Path("artifacts"),
    bundle_name: Optional[str] = None
) -> ArtifactBundle:
    """Create standardized artifact bundle."""
    
    # Load config and metrics
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create bundle
    bundle = ArtifactBundle.create(
        config=config,
        metrics=metrics,
        model_path=model_path,
        model_format=model_format,
        output_dir=output_dir,
        bundle_name=bundle_name
    )
    
    return bundle


def verify_bundle(bundle_hash: str, artifacts_dir: Path = Path("artifacts")) -> Dict[str, Any]:
    """Verify artifact bundle structure and contents."""
    bundle_dir = artifacts_dir / bundle_hash
    
    if not bundle_dir.exists():
        return {
            "valid": False,
            "errors": [f"Bundle not found: {bundle_hash}"]
        }
    
    bundle = ArtifactBundle(bundle_dir)
    return bundle.verify()


def list_bundles(artifacts_dir: Path = Path("artifacts")) -> List[Dict[str, Any]]:
    """List all available artifact bundles."""
    return ArtifactBundle.list_bundles(artifacts_dir)


def archive_bundle(
    bundle_hash: str,
    format: str = "tar.gz",
    artifacts_dir: Path = Path("artifacts")
) -> Path:
    """Archive bundle to compressed file."""
    bundle_dir = artifacts_dir / bundle_hash
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_hash}")
    
    bundle = ArtifactBundle(bundle_dir)
    return bundle.archive(format)


def get_bundle_info(bundle_hash: str, artifacts_dir: Path = Path("artifacts")) -> Dict[str, Any]:
    """Get detailed bundle information."""
    bundle_dir = artifacts_dir / bundle_hash
    
    if not bundle_dir.exists():
        return {
            "exists": False,
            "error": f"Bundle not found: {bundle_hash}"
        }
    
    bundle = ArtifactBundle(bundle_dir)
    return bundle.get_info()


def format_size(n_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f}TB"