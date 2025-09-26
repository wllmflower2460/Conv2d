#!/usr/bin/env python3
"""Example usage of the configuration system."""

from __future__ import annotations

from pathlib import Path

from conv2d.config import (
    Config,
    ConfigLoader,
    compute_config_hash,
    get_config_diff,
    load_config,
)


def example_basic_loading():
    """Example: Basic configuration loading."""
    print("=" * 60)
    print("Example: Basic Configuration Loading")
    print("=" * 60)
    
    # Load base configuration
    config = load_config(config_dir="../conf", config_name="base")
    
    print(f"Project: {config.project_name} v{config.project_version}")
    print(f"Seed: {config.seed}")
    print(f"Device: {config.hardware.device}")
    print(f"Dataset: {config.data.dataset_name}")
    print()


def example_with_overrides():
    """Example: Loading with overrides."""
    print("=" * 60)
    print("Example: Configuration with Overrides")
    print("=" * 60)
    
    # Load with overrides (Hydra syntax)
    overrides = [
        "env=gpu",  # Use GPU environment
        "exp=quadruped",  # Quadruped experiment
        "training.epochs=50",  # Override epochs
        "data.batch_size=64",  # Override batch size
    ]
    
    loader = ConfigLoader(config_dir="../conf")
    config = loader.load(overrides=overrides)
    
    print(f"Environment: {config.hardware.device}")
    print(f"Experiment: {config.experiment.name}")
    print(f"Training epochs: {config.training.epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Config hash: {config.experiment.config_hash}")
    print()


def example_config_hashing():
    """Example: Configuration hashing for reproducibility."""
    print("=" * 60)
    print("Example: Configuration Hashing")
    print("=" * 60)
    
    # Load two configurations
    config1 = load_config(
        config_dir="../conf",
        overrides=["training.epochs=100"]
    )
    
    config2 = load_config(
        config_dir="../conf",
        overrides=["training.epochs=200"]
    )
    
    # Compute hashes
    hash1 = compute_config_hash(config1.model_dump())
    hash2 = compute_config_hash(config2.model_dump())
    
    print(f"Config 1 hash (100 epochs): {hash1}")
    print(f"Config 2 hash (200 epochs): {hash2}")
    print(f"Hashes match: {hash1 == hash2}")
    
    # Get differences
    diff = get_config_diff(config1.model_dump(), config2.model_dump())
    print(f"Differences: {diff}")
    print()


def example_validation():
    """Example: Configuration validation with Pydantic."""
    print("=" * 60)
    print("Example: Configuration Validation")
    print("=" * 60)
    
    # Try to load with invalid configuration
    try:
        config = load_config(
            config_dir="../conf",
            overrides=["data.batch_size=-1"]  # Invalid: negative batch size
        )
    except Exception as e:
        print(f"Validation failed (expected): {e}")
    
    # Try with valid configuration
    config = load_config(
        config_dir="../conf",
        overrides=["data.batch_size=32"]
    )
    print(f"Valid batch size: {config.data.batch_size}")
    print()


def example_experiment_configs():
    """Example: Using experiment configurations."""
    print("=" * 60)
    print("Example: Experiment Configurations")
    print("=" * 60)
    
    experiments = ["quadruped", "wisdm", "pamap2"]
    
    for exp_name in experiments:
        config = load_config(
            config_dir="../conf",
            overrides=[f"exp={exp_name}"]
        )
        
        print(f"\n{exp_name.upper()} Experiment:")
        print(f"  Dataset: {config.data.dataset_name}")
        print(f"  FSQ levels: {config.model.fsq.levels}")
        print(f"  FSQ codebook size: {config.model.fsq.codebook_size}")
        print(f"  HSMM states: {config.model.hsmm.num_states}")
        print(f"  Training epochs: {config.training.epochs}")
        print(f"  Config hash: {config.experiment.config_hash[:8]}")


def example_environment_configs():
    """Example: Using environment configurations."""
    print("=" * 60)
    print("Example: Environment Configurations")
    print("=" * 60)
    
    environments = ["local", "gpu", "pi", "hailo"]
    
    for env_name in environments:
        try:
            config = load_config(
                config_dir="../conf",
                overrides=[f"env={env_name}"]
            )
            
            print(f"\n{env_name.upper()} Environment:")
            print(f"  Device: {config.hardware.device}")
            print(f"  Mixed precision: {config.hardware.mixed_precision}")
            print(f"  Compile model: {config.hardware.compile_model}")
            
            if env_name == "hailo":
                print(f"  Batch size adjusted: {config.data.batch_size}")
                
        except Exception as e:
            print(f"\n{env_name.upper()} Environment: Not available on this system")


def example_save_config():
    """Example: Saving configuration with hash."""
    print("=" * 60)
    print("Example: Saving Configuration")
    print("=" * 60)
    
    # Load configuration
    loader = ConfigLoader(config_dir="../conf")
    config = loader.load(overrides=["exp=quadruped", "training.epochs=42"])
    
    # Save with hash
    output_dir = Path("example_output")
    saved_path = loader.save(output_dir, include_hash=True)
    
    print(f"Configuration saved to: {saved_path}")
    print(f"Config hash in filename: {config.experiment.config_hash}")
    
    # Clean up
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    print()


def main():
    """Run all examples."""
    examples = [
        example_basic_loading,
        example_with_overrides,
        example_config_hashing,
        example_validation,
        example_experiment_configs,
        example_environment_configs,
        example_save_config,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example failed: {e}")
            print()


if __name__ == "__main__":
    main()