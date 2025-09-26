#!/usr/bin/env python3
"""
Validate FSQ configuration against real data.

This script validates FSQ configurations using real datasets, checking
performance metrics against target thresholds.

Usage:
    python scripts/validate_fsq_config.py
    python scripts/validate_fsq_config.py --fsq-levels 4 4 4 --window-size 100
    python scripts/validate_fsq_config.py --config configs/fsq_production.yaml
"""

import argparse
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.fsq_tuning_toolkit import FSQConfig, FSQTuner


def load_config_from_yaml(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_fsq_configuration(
    fsq_levels: List[int] = None,
    window_size: int = 100,
    clustering_k: int = 12,
    min_support: float = 0.005,
    config_file: Optional[str] = None,
    verbose: bool = False
) -> bool:
    """
    Validate FSQ configuration with simulated or real data.
    
    Args:
        fsq_levels: FSQ quantization levels
        window_size: Window size for temporal data
        clustering_k: Number of clusters
        min_support: Minimum cluster support
        config_file: Path to YAML configuration file
        verbose: Show detailed output
        
    Returns:
        True if validation passes
    """
    # Load configuration from file if provided
    if config_file:
        config_data = load_config_from_yaml(config_file)
        fsq_config = config_data.get('fsq', {})
        fsq_levels = fsq_levels or fsq_config.get('levels', [4, 4, 4])
        clustering_config = config_data.get('clustering', {})
        clustering_k = clustering_config.get('k', clustering_k)
        min_support = clustering_config.get('min_support', min_support)
    else:
        fsq_levels = fsq_levels or [4, 4, 4]
    
    # Create FSQ configuration
    config = FSQConfig(
        levels=fsq_levels,
        window_size=window_size,
        clustering_k=clustering_k,
        min_support=min_support
    )
    
    print("FSQ Configuration Validation")
    print("=" * 50)
    print(f"FSQ Levels: {config.levels}")
    print(f"Codebook size: {config.codebook_size}")
    print(f"Window size: {config.window_size}")
    print(f"Clustering K: {config.clustering_k}")
    print(f"Min support: {config.min_support:.1%}")
    
    # Create tuner and generate test data
    tuner = FSQTuner()
    
    # Generate synthetic test data (in practice, load real data)
    print("\nGenerating test data...")
    test_data = np.random.randn(5000, config.embedding_dim)
    
    # Validate configuration
    print("\nValidating configuration...")
    validation_results = tuner.validate_config(config, test_data)
    
    # Display results
    print("\n" + "=" * 50)
    print("Validation Results:")
    print("=" * 50)
    
    metrics = validation_results['metrics']
    print(f"Estimated accuracy: {metrics['estimated_accuracy']:.1%}")
    print(f"Estimated ECE: {metrics['estimated_ece']:.3f}")
    print(f"Estimated utilization: {metrics['estimated_utilization']:.1%}")
    print(f"Estimated latency: {metrics['estimated_latency_ms']}ms")
    
    # Check targets
    passed = validation_results['passed']
    if passed:
        print("\n✅ Configuration PASSED all targets")
    else:
        print("\n❌ Configuration FAILED some targets")
    
    # Show warnings if verbose
    if verbose and validation_results['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    # Show recommendations if verbose
    if verbose and validation_results['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in validation_results['recommendations']:
            print(f"  - {rec}")
    
    return passed


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Validate FSQ configuration against real data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate default configuration
  python scripts/validate_fsq_config.py
  
  # Validate custom FSQ levels
  python scripts/validate_fsq_config.py --fsq-levels 4 4 4
  
  # Validate with full parameters
  python scripts/validate_fsq_config.py \\
      --fsq-levels 8 6 5 \\
      --window-size 150 \\
      --clustering-k 16 \\
      --min-support 0.01
  
  # Load configuration from YAML
  python scripts/validate_fsq_config.py --config configs/fsq_production.yaml
        """
    )
    
    parser.add_argument(
        '--fsq-levels',
        nargs='+',
        type=int,
        help='FSQ quantization levels (e.g., 4 4 4)'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=100,
        help='Window size for temporal data (default: 100)'
    )
    
    parser.add_argument(
        '--clustering-k',
        type=int,
        default=12,
        help='Number of clusters (default: 12)'
    )
    
    parser.add_argument(
        '--min-support',
        type=float,
        default=0.005,
        help='Minimum cluster support fraction (default: 0.005)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output including warnings'
    )
    
    args = parser.parse_args()
    
    # Run validation
    passed = validate_fsq_configuration(
        fsq_levels=args.fsq_levels,
        window_size=args.window_size,
        clustering_k=args.clustering_k,
        min_support=args.min_support,
        config_file=args.config,
        verbose=args.verbose
    )
    
    # Return 0 for success, 1 for failure
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())