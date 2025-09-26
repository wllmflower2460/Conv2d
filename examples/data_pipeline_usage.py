#!/usr/bin/env python3
"""Example usage of the refactored data layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from conv2d.data import (
    BandpassFilter,
    Clip,
    Compose,
    InterpolateNaN,
    IterableTimeSeriesDataset,
    MemoryMappedDataset,
    Pipeline,
    QuantizeFSQ,
    Standardize,
    ToTensor,
    TransformPipeline,
    Window,
)


def example_basic_pipeline():
    """Example: Basic preprocessing pipeline."""
    print("=" * 60)
    print("Example: Basic Preprocessing Pipeline")
    print("=" * 60)
    
    # Create sample data with some issues
    X = np.random.randn(100, 9, 1000).astype(np.float64)  # Wrong dtype
    X[0, 0, 100:110] = np.nan  # Add some NaNs
    X[1, :, 200:210] = 100  # Add outliers
    
    # Create pipeline
    pipeline = Compose([
        InterpolateNaN(method="linear"),  # Fix NaNs
        Standardize(),  # Zero mean, unit variance
        Clip(min_val=-3, max_val=3),  # Remove outliers
        Window(window_size=100, step_size=50),  # Extract windows
        ToTensor(device="cpu"),  # Convert to tensor
    ])
    
    # Fit on training data
    pipeline.fit(X[:80])  # Use first 80 samples for training
    
    # Transform training data
    X_train = pipeline.transform(X[:80])
    print(f"Training shape: {X_train.shape}")
    print(f"Training dtype: {X_train.dtype}")
    print(f"No NaNs: {not torch.isnan(X_train).any()}")
    print(f"Clipped range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    
    # Transform test data (uses fitted parameters)
    X_test = pipeline.transform(X[80:])
    print(f"Test shape: {X_test.shape}")
    print()


def example_fsq_pipeline():
    """Example: Pipeline with FSQ quantization."""
    print("=" * 60)
    print("Example: FSQ Quantization Pipeline")
    print("=" * 60)
    
    # Create continuous data
    X = np.random.randn(100, 3, 500).astype(np.float32)
    
    # Create pipeline with FSQ
    pipeline = Pipeline([
        ("standardize", Standardize()),
        ("quantize", QuantizeFSQ(levels=[8, 6, 5], method="percentile")),
    ])
    
    # Fit and transform
    X_quantized = pipeline.fit_transform(X)
    
    # Check quantization
    for dim in range(3):
        unique_vals = np.unique(X_quantized[:, dim, :])
        print(f"Dimension {dim}: {len(unique_vals)} unique values")
    
    print(f"Codebook size: {8 * 6 * 5} = 240 codes")
    print()


def example_standard_pipeline():
    """Example: Using pre-configured standard pipeline."""
    print("=" * 60)
    print("Example: Standard Pipeline Factory")
    print("=" * 60)
    
    # Create standard pipeline
    pipeline = TransformPipeline.create_standard_pipeline(
        sampling_rate=100.0,
        window_size=100,
        step_size=25,
        standardize=True,
        interpolate_nan=True,
        bandpass=(0.5, 20.0),  # Filter 0.5-20 Hz
        clip_range=(-5, 5),
        to_tensor=True,
    )
    
    # Create data
    X = np.random.randn(50, 9, 1000).astype(np.float32)
    
    # Fit and transform
    X_processed = pipeline.fit_transform(X)
    
    print(f"Output shape: {X_processed.shape}")
    print(f"Output type: {type(X_processed)}")
    print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
    print()


def example_memory_mapped_dataset():
    """Example: Memory-mapped dataset for large files."""
    print("=" * 60)
    print("Example: Memory-Mapped Dataset")
    print("=" * 60)
    
    # Create large dataset and save to disk
    data_path = Path("example_data.npy")
    shape = (1000, 9, 10000)  # 1000 samples, 9 channels, 10k timesteps
    
    # Create memory-mapped file
    print(f"Creating memory-mapped file: {shape}")
    if not data_path.exists():
        # Create fake data
        data = np.random.randn(*shape).astype(np.float32)
        MemoryMappedDataset.create_memmap_file(data, data_path)
        del data  # Free memory
    
    # Create preprocessing pipeline
    transform = Compose([
        Standardize(),
        Window(window_size=100, step_size=50),
        ToTensor(),
    ])
    
    # Create dataset (data not loaded into memory)
    dataset = MemoryMappedDataset(
        data_path=data_path,
        shape=shape,
        dtype=np.float32,
        transform=transform,
        window_size=100,
        step_size=50,
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample (only loads what's needed)
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    
    # Clean up
    dataset.close()
    data_path.unlink(missing_ok=True)
    print()


def example_iterable_dataset():
    """Example: Iterable dataset for streaming."""
    print("=" * 60)
    print("Example: Iterable Streaming Dataset")
    print("=" * 60)
    
    # Create data file
    data_path = Path("streaming_data.bin")
    channels = 9
    n_timesteps = 100000  # Long recording
    
    if not data_path.exists():
        # Create fake streaming data
        data = np.random.randn(channels, n_timesteps).astype(np.float32)
        data.tofile(data_path)
        del data
    
    # Create transform
    transform = Compose([
        Standardize(),
        ToTensor(),
    ])
    
    # Create iterable dataset
    dataset = IterableTimeSeriesDataset(
        data_path=data_path,
        window_size=100,
        step_size=50,
        channels=channels,
        transform=transform,
        chunk_size=1000,  # Read 1000 timesteps at a time
    )
    
    # Iterate over windows (streaming, no full load)
    print("Streaming windows:")
    for i, window in enumerate(dataset):
        if i >= 5:  # Just show first 5
            break
        print(f"  Window {i}: shape {window.shape}")
    
    # Clean up
    data_path.unlink(missing_ok=True)
    print()


def example_dataloader_integration():
    """Example: Integration with PyTorch DataLoader."""
    print("=" * 60)
    print("Example: PyTorch DataLoader Integration")
    print("=" * 60)
    
    # Create data
    X = np.random.randn(1000, 9, 200).astype(np.float32)
    y = np.random.randint(0, 4, size=(1000,))
    
    # Save as memory-mapped files
    X_path = Path("X_data.npy")
    y_path = Path("y_data.npy")
    
    MemoryMappedDataset.create_memmap_file(X, X_path)
    MemoryMappedDataset.create_memmap_file(y, y_path)
    
    # Create transform pipeline
    transform = Compose([
        Standardize(),
        ToTensor(),
    ])
    
    # Create dataset
    dataset = MemoryMappedDataset(
        data_path=X_path,
        shape=X.shape,
        transform=transform,
        labels_path=y_path,
        labels_shape=y.shape,
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    # Iterate over batches
    print("DataLoader batches:")
    for i, (batch_X, batch_y) in enumerate(dataloader):
        if i >= 3:  # Just show first 3 batches
            break
        print(f"  Batch {i}: X={batch_X.shape}, y={batch_y.shape}")
    
    # Clean up
    dataset.close()
    X_path.unlink(missing_ok=True)
    y_path.unlink(missing_ok=True)
    print()


def example_dtype_safety():
    """Example: Dtype enforcement throughout pipeline."""
    print("=" * 60)
    print("Example: Dtype Safety")
    print("=" * 60)
    
    # Start with various dtypes
    data_f64 = np.random.randn(10, 3, 100).astype(np.float64)
    data_f16 = np.random.randn(10, 3, 100).astype(np.float16)
    data_i32 = np.random.randint(-100, 100, (10, 3, 100), dtype=np.int32)
    
    # Pipeline enforces float32 at boundaries
    pipeline = Compose([
        Standardize(),
        Window(20, 10),
        ToTensor(),
    ])
    
    for name, data in [
        ("float64", data_f64),
        ("float16", data_f16),
        ("int32", data_i32),
    ]:
        result = pipeline.fit_transform(data)
        print(f"{name:8} -> {result.dtype}")
    
    print("\nAll outputs are float32 for consistency!")
    print()


def main():
    """Run all examples."""
    examples = [
        example_basic_pipeline,
        example_fsq_pipeline,
        example_standard_pipeline,
        example_memory_mapped_dataset,
        example_iterable_dataset,
        example_dataloader_integration,
        example_dtype_safety,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example failed: {e}")
            print()


if __name__ == "__main__":
    main()