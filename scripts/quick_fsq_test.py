#!/usr/bin/env python3
"""
Quick FSQ configuration test script.

A simple utility to quickly test FSQ model configurations and display
key metrics like codebook size and parameter count.

Usage:
    python scripts/quick_fsq_test.py
    python scripts/quick_fsq_test.py --levels 8 6 5
    python scripts/quick_fsq_test.py --levels 4 4 --verbose
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.conv2d_fsq_optimized import Conv2dFSQOptimized


def test_fsq_config(levels=None, verbose=False):
    """
    Test FSQ configuration and display key metrics.
    
    Args:
        levels: List of FSQ levels (default: [4, 4, 4])
        verbose: Show additional details
    """
    if levels is None:
        levels = [4, 4, 4]
    
    # Create model with specified configuration
    model = Conv2dFSQOptimized(fsq_levels=levels)
    
    # Calculate metrics
    codebook_size = model.fsq.num_codes
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024  # Assuming float32
    
    # Display results
    print(f"FSQ Configuration Test")
    print("=" * 40)
    print(f"FSQ Levels: {levels}")
    print(f"Codebook size: {codebook_size}")
    print(f"Model params: {total_params:,}")
    
    if verbose:
        print(f"\nDetailed Information:")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Non-trainable params: {total_params - trainable_params:,}")
        print(f"  Model size: ~{model_size_mb:.2f} MB (float32)")
        print(f"  Embedding dimension: {model.fsq.dim}")
        
        # Calculate expected utilization
        if codebook_size <= 64:
            expected_utilization = ">80%"
        elif codebook_size <= 256:
            expected_utilization = "60-80%"
        else:
            expected_utilization = "40-60%"
        print(f"  Expected utilization: {expected_utilization}")
        
        # Show layer breakdown
        print(f"\nLayer Breakdown:")
        print(f"  Encoder params: {sum(p.numel() for n, p in model.named_parameters() if 'encoder' in n):,}")
        print(f"  FSQ params: {sum(p.numel() for n, p in model.named_parameters() if 'fsq' in n):,}")
        print(f"  Classifier params: {sum(p.numel() for n, p in model.named_parameters() if 'classifier' in n):,}")
    
    return {
        'levels': levels,
        'codebook_size': codebook_size,
        'total_params': total_params,
        'model_size_mb': model_size_mb
    }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Quick FSQ configuration test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default configuration [4, 4, 4]
  python scripts/quick_fsq_test.py
  
  # Test balanced configuration [8, 6, 5]
  python scripts/quick_fsq_test.py --levels 8 6 5
  
  # Test minimal configuration with verbose output
  python scripts/quick_fsq_test.py --levels 4 4 --verbose
  
  # Test extended configuration
  python scripts/quick_fsq_test.py --levels 8 6 5 5 4 -v
        """
    )
    
    parser.add_argument(
        '--levels',
        nargs='+',
        type=int,
        default=[4, 4, 4],
        help='FSQ levels (default: 4 4 4)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information'
    )
    
    args = parser.parse_args()
    
    # Run test
    results = test_fsq_config(levels=args.levels, verbose=args.verbose)
    
    # Return 0 for success
    return 0


if __name__ == "__main__":
    sys.exit(main())