#!/usr/bin/env python3
"""Examples demonstrating FSQ encoding contract usage.

This module shows how to:
1. Use the single-function interface
2. Integrate with data pipelines
3. Track code usage statistics
4. Verify encoding invariants
5. Deploy in production settings
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from conv2d.features import encode_fsq, verify_fsq_invariants, FSQEncoder
from conv2d.data import Compose, Standardize, ToTensor, Window


def example_basic_encoding():
    """Example 1: Basic FSQ encoding with single function interface."""
    print("=" * 60)
    print("Example 1: Basic FSQ Encoding")
    print("=" * 60)
    
    # Create IMU data: (batch, channels=9, sensors=2, timesteps=100)
    batch_size = 32
    x = torch.randn(batch_size, 9, 2, 100, dtype=torch.float32)
    
    # Encode using single function interface
    result = encode_fsq(x)
    
    # Access outputs
    print(f"Input shape: {x.shape}")
    print(f"Codes shape: {result.codes.shape}")
    print(f"Codes dtype: {result.codes.dtype}")
    print(f"Quantized shape: {result.quantized.shape}")
    print(f"Quantized dtype: {result.quantized.dtype}")
    print(f"Codebook size: {result.codebook_size}")
    print(f"Perplexity: {result.perplexity:.2f}")
    print(f"Unique codes used: {len(torch.unique(result.codes))}")
    print()


def example_custom_levels():
    """Example 2: Custom quantization levels for different granularity."""
    print("=" * 60)
    print("Example 2: Custom Quantization Levels")
    print("=" * 60)
    
    x = torch.randn(16, 9, 2, 100, dtype=torch.float32)
    
    # Test different level configurations
    configs = [
        ([4, 4, 4], "Coarse: 64 codes"),
        ([8, 6, 5], "Default: 240 codes"),
        ([16, 12, 10], "Fine: 1920 codes"),
    ]
    
    for levels, description in configs:
        result = encode_fsq(x, levels=levels, reset_stats=True)
        print(f"{description}")
        print(f"  Levels: {levels}")
        print(f"  Codebook size: {result.codebook_size}")
        print(f"  Codes used: {len(torch.unique(result.codes))}")
        print(f"  Perplexity: {result.perplexity:.2f}")
    print()


def example_data_pipeline_integration():
    """Example 3: Integration with preprocessing pipeline."""
    print("=" * 60)
    print("Example 3: Data Pipeline Integration")
    print("=" * 60)
    
    # Create raw data
    raw_data = np.random.randn(100, 9, 200).astype(np.float32)
    raw_data[0, 0, 50:60] = np.nan  # Add some NaNs
    
    # Create preprocessing pipeline
    from conv2d.data import InterpolateNaN
    
    pipeline = Compose([
        InterpolateNaN(method="linear"),
        Standardize(),
        Window(window_size=100, step_size=50),
        ToTensor(),
    ])
    
    # Preprocess data
    processed = pipeline.fit_transform(raw_data)
    # After windowing, shape is (B, C, n_windows, window_size)
    # We need to reshape to (B*n_windows, C, window_size)
    B, C = raw_data.shape[0], raw_data.shape[1]
    
    # Processed is actually (n_total_windows, C, window_size) after Window transform
    # because Window flattens across all samples
    n_windows, C, T = processed.shape
    print(f"After windowing: {processed.shape}")
    
    # Add sensor dimension for FSQ
    processed_fsq = processed.unsqueeze(2).expand(-1, -1, 2, -1)  # Add sensors
    
    # Encode
    result = encode_fsq(processed_fsq)
    print(f"Encoded {n_windows} windows")
    print(f"Codes shape: {result.codes.shape}")
    print(f"Perplexity: {result.perplexity:.2f}")
    print()


def example_statistics_tracking():
    """Example 4: Tracking code usage statistics across batches."""
    print("=" * 60)
    print("Example 4: Statistics Tracking")
    print("=" * 60)
    
    # Process multiple batches
    n_batches = 5
    batch_size = 32
    
    print("Processing batches:")
    for i in range(n_batches):
        x = torch.randn(batch_size, 9, 2, 100)
        
        # Don't reset stats to accumulate across batches
        result = encode_fsq(x, reset_stats=(i == 0))
        
        print(f"  Batch {i+1}: Perplexity = {result.perplexity:.2f}, "
              f"Unique codes = {len(torch.unique(result.codes))}")
    
    # Final statistics
    print(f"\nFinal accumulated statistics:")
    print(f"  Total perplexity: {result.perplexity:.2f}")
    print(f"  Code usage distribution (top 10):")
    
    hist = result.usage_histogram
    top_indices = np.argsort(hist)[-10:][::-1]
    for idx in top_indices[:5]:
        print(f"    Code {idx}: {hist[idx]*100:.2f}%")
    print()


def example_determinism_verification():
    """Example 5: Verify deterministic encoding guarantee."""
    print("=" * 60)
    print("Example 5: Determinism Verification")
    print("=" * 60)
    
    # Create test input
    torch.manual_seed(42)
    x = torch.randn(16, 9, 2, 100)
    
    # Encode multiple times
    results = []
    for i in range(3):
        result = encode_fsq(x, reset_stats=True)
        results.append(result)
        print(f"Encoding {i+1}: First 10 codes = {result.codes[0, :10].tolist()}")
    
    # Verify identical codes
    codes_match = all(
        torch.equal(results[0].codes, r.codes) for r in results[1:]
    )
    features_match = all(
        torch.allclose(results[0].quantized, r.quantized, atol=1e-7) 
        for r in results[1:]
    )
    
    print(f"\nDeterminism check:")
    print(f"  Codes identical: {codes_match} ✓" if codes_match else f"  Codes differ: ✗")
    print(f"  Features identical: {features_match} ✓" if features_match else f"  Features differ: ✗")
    print()


def example_invariant_checking():
    """Example 6: Use invariant checking for production validation."""
    print("=" * 60)
    print("Example 6: Invariant Checking")
    print("=" * 60)
    
    # Create encoder
    encoder = FSQEncoder(levels=[8, 6, 5], seed=42)
    
    # Create valid input
    x_valid = torch.randn(16, 9, 2, 100)
    
    # Verify invariants
    is_valid = verify_fsq_invariants(encoder, x_valid)
    print(f"Valid input invariants: {is_valid} ✓" if is_valid else f"Valid input invariants: Failed ✗")
    
    # Test with invalid input (wrong shape)
    x_invalid = torch.randn(16, 8, 2, 100)  # Wrong channels
    
    try:
        is_valid = verify_fsq_invariants(encoder, x_invalid)
        if is_valid:
            print(f"Invalid input caught: No ✗")
        else:
            print(f"Invalid input caught: Yes ✓ (returned False)")
    except Exception as e:
        print(f"Invalid input caught: Yes ✓ (raised exception)")
        print(f"  Error: {str(e)[:50]}...")
    print()


def example_batch_processing():
    """Example 7: Efficient batch processing with DataLoader."""
    print("=" * 60)
    print("Example 7: Batch Processing with DataLoader")
    print("=" * 60)
    
    # Create dataset
    n_samples = 1000
    X = torch.randn(n_samples, 9, 2, 100)
    y = torch.randint(0, 4, (n_samples,))  # 4 classes
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Process batches
    all_codes = []
    all_labels = []
    
    print(f"Processing {len(dataloader)} batches...")
    for i, (batch_x, batch_y) in enumerate(dataloader):
        if i >= 3:  # Just show first 3
            break
            
        result = encode_fsq(batch_x, reset_stats=(i == 0))
        all_codes.append(result.codes)
        all_labels.append(batch_y)
        
        print(f"  Batch {i+1}: Shape {result.codes.shape}, "
              f"Perplexity {result.perplexity:.2f}")
    
    # Combine results
    all_codes = torch.cat(all_codes, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"\nProcessed {all_codes.shape[0]} samples")
    print(f"Code tensor shape: {all_codes.shape}")
    print(f"Ready for downstream classification!")
    print()


def example_embedding_dimensions():
    """Example 8: Different embedding dimensions for various use cases."""
    print("=" * 60)
    print("Example 8: Embedding Dimensions")
    print("=" * 60)
    
    x = torch.randn(8, 9, 2, 100)
    
    dimensions = [
        (32, "Compact: Memory-efficient"),
        (64, "Standard: Balanced"),
        (128, "Rich: More expressive"),
        (256, "Large: Maximum capacity"),
    ]
    
    for dim, description in dimensions:
        result = encode_fsq(x, embedding_dim=dim, reset_stats=True)
        print(f"{description} (dim={dim})")
        print(f"  Output shape: {result.quantized.shape}")
        print(f"  Memory: ~{result.quantized.numel() * 4 / 1024:.1f} KB")
    print()


def example_production_deployment():
    """Example 9: Production deployment pattern."""
    print("=" * 60)
    print("Example 9: Production Deployment")
    print("=" * 60)
    
    class FSQProcessor:
        """Production-ready FSQ processor with error handling."""
        
        def __init__(self, levels=[8, 6, 5], embedding_dim=64):
            self.levels = levels
            self.embedding_dim = embedding_dim
            self.stats = {"processed": 0, "errors": 0}
            
        def process_batch(self, x):
            """Process a batch with error handling."""
            try:
                # Validate input
                assert x.shape[1:] == (9, 2, 100), f"Invalid shape: {x.shape}"
                assert x.dtype == torch.float32, f"Invalid dtype: {x.dtype}"
                
                # Encode
                result = encode_fsq(
                    x, 
                    levels=self.levels,
                    embedding_dim=self.embedding_dim,
                    reset_stats=False
                )
                
                # Update stats
                self.stats["processed"] += x.shape[0]
                
                return {
                    "success": True,
                    "codes": result.codes,
                    "features": result.quantized,
                    "perplexity": result.perplexity,
                }
                
            except Exception as e:
                self.stats["errors"] += 1
                return {
                    "success": False,
                    "error": str(e),
                }
        
        def get_stats(self):
            """Get processing statistics."""
            return self.stats
    
    # Use in production
    processor = FSQProcessor()
    
    # Process valid batch
    x_valid = torch.randn(16, 9, 2, 100)
    result = processor.process_batch(x_valid)
    print(f"Valid batch: success={result['success']}")
    
    # Process invalid batch
    x_invalid = torch.randn(16, 9, 3, 100)  # Wrong sensors
    result = processor.process_batch(x_invalid)
    print(f"Invalid batch: success={result['success']}, error={result.get('error', '')[:30]}...")
    
    # Check stats
    stats = processor.get_stats()
    print(f"\nProcessing stats:")
    print(f"  Processed: {stats['processed']} samples")
    print(f"  Errors: {stats['errors']}")
    print()


def main():
    """Run all examples."""
    examples = [
        example_basic_encoding,
        example_custom_levels,
        example_data_pipeline_integration,
        example_statistics_tracking,
        example_determinism_verification,
        example_invariant_checking,
        example_batch_processing,
        example_embedding_dimensions,
        example_production_deployment,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example {example_func.__name__} failed: {e}")
            print()


if __name__ == "__main__":
    main()