"""Utility functions for Conv2d CLI."""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys


def setup_logging(level: str = "INFO"):
    """Configure logging for CLI."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("conv2d_cli.log")
        ]
    )


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    if not config_path:
        return {}
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_config(config: Dict[str, Any], output_path: Path):
    """Save configuration to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if output_path.suffix in ['.yaml', '.yml']:
            yaml.safe_dump(config, f, default_flow_style=False)
        elif output_path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {output_path.suffix}")


def format_bytes(n_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f}TB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"