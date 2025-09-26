"""
Conway2d Pipeline Contracts - Runtime Implementation

This module implements the FROZEN contracts defined in docs/contracts.md.
These contracts ensure pipeline stability and prevent interface drift.

Version: 1.0.0 (FROZEN)
"""

import json
import os
from functools import wraps
from pathlib import Path
from typing import Dict, Tuple, Callable, Any, Union

import torch
import numpy as np


# Contract Configuration
ENABLE_CONTRACT_VALIDATION = os.getenv("CONV2D_VALIDATE_CONTRACTS", "true").lower() == "true"


# =============================================================================
# INPUT CONTRACT - IMMUTABLE
# =============================================================================

class InputContract:
    """Frozen input contract for pipeline stability."""
    
    # IMMUTABLE CONSTANTS
    SHAPE: Tuple[int, int, int, int] = (None, 9, 2, 100)  # (B, channels, height, width)
    DTYPE: torch.dtype = torch.float32
    RANGE: Tuple[float, float] = (-10.0, 10.0)
    
    CHANNEL_LAYOUT = {
        0: "accel_x",    # m/s²
        1: "accel_y",    # m/s²
        2: "accel_z",    # m/s²
        3: "gyro_x",     # rad/s
        4: "gyro_y",     # rad/s
        5: "gyro_z",     # rad/s
        6: "mag_x",      # μT
        7: "mag_y",      # μT
        8: "mag_z"       # μT
    }
    
    @staticmethod
    def validate(x: torch.Tensor) -> None:
        """Validate input tensor meets pipeline contract."""
        # Shape validation
        assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}D"
        assert x.shape[1:] == (9, 2, 100), f"Expected (B,9,2,100), got {x.shape}"
        
        # Dtype validation
        assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
        
        # Value validation
        assert torch.isfinite(x).all(), "Input contains NaN or Inf values"
        max_val = x.abs().max()
        assert max_val <= 10.0, f"Input range violation: {max_val:.2f} > 10.0"


# =============================================================================
# FSQ CONTRACT - IMMUTABLE
# =============================================================================

class FSQContract:
    """Frozen FSQ encoding contract."""
    
    # IMMUTABLE PARAMETERS
    LEVELS: Tuple[int, ...] = (4, 4, 4)
    CODEBOOK_SIZE: int = 64
    FEATURE_DIM: int = 32
    TEMPORAL_DIM: int = 50
    
    # Output shapes (batch dimension is None for flexibility)
    OUTPUT_SHAPES = {
        "codes": (None, 50),
        "quantized": (None, 32, 50),
        "indices": (None, 50, 3),
        "commitment_loss": ()
    }
    
    OUTPUT_DTYPES = {
        "codes": torch.int32,
        "quantized": torch.float32,
        "indices": torch.int32,
        "commitment_loss": torch.float32
    }
    
    REQUIRED_KEYS = {"codes", "quantized", "indices", "commitment_loss"}
    
    @staticmethod
    def validate_output(output: Dict[str, torch.Tensor], batch_size: int) -> None:
        """Validate FSQ output meets contract."""
        # Key validation
        missing_keys = FSQContract.REQUIRED_KEYS - set(output.keys())
        assert not missing_keys, f"Missing required keys: {missing_keys}"
        
        # Shape validation
        expected_shapes = {
            "codes": (batch_size, 50),
            "quantized": (batch_size, 32, 50),
            "indices": (batch_size, 50, 3),
            "commitment_loss": ()
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = output[key].shape
            assert actual_shape == expected_shape, f"{key} shape mismatch: {actual_shape} != {expected_shape}"
        
        # Dtype validation
        for key, expected_dtype in FSQContract.OUTPUT_DTYPES.items():
            actual_dtype = output[key].dtype
            assert actual_dtype == expected_dtype, f"{key} dtype mismatch: {actual_dtype} != {expected_dtype}"
        
        # Range validation
        codes = output["codes"]
        assert codes.min() >= 0, f"codes minimum {codes.min()} < 0"
        assert codes.max() < FSQContract.CODEBOOK_SIZE, f"codes maximum {codes.max()} >= {FSQContract.CODEBOOK_SIZE}"
        
        # Finite validation
        assert torch.isfinite(output["quantized"]).all(), "quantized contains NaN/Inf"
        assert torch.isfinite(output["commitment_loss"]).all(), "commitment_loss contains NaN/Inf"
        
        indices = output["indices"]
        assert indices.min() >= 0, "indices must be non-negative"


# =============================================================================
# CLUSTERING CONTRACT - IMMUTABLE
# =============================================================================

class ClusteringContract:
    """Frozen clustering contract."""
    
    # IMMUTABLE PARAMETERS
    DEFAULT_N_CLUSTERS: int = 12
    MIN_CLUSTER_SUPPORT: float = 0.005
    CLUSTER_METHODS: Tuple[str, ...] = ("kmeans", "gmm")
    
    OUTPUT_SHAPES = {
        "motifs_raw": (None, 50),
        "cluster_centers": (None, None),  # (n_clusters, feature_dim)
        "assignment_confidence": (None, 50),
        "hungarian_mapping": (None,)  # (n_clusters,)
    }
    
    OUTPUT_DTYPES = {
        "motifs_raw": torch.int32,
        "cluster_centers": torch.float32,
        "assignment_confidence": torch.float32,
        "hungarian_mapping": torch.int32
    }
    
    REQUIRED_KEYS = {"motifs_raw", "cluster_centers", "assignment_confidence", "hungarian_mapping"}
    
    @staticmethod
    def validate_output(
        output: Dict[str, torch.Tensor], 
        batch_size: int, 
        n_clusters: int
    ) -> None:
        """Validate clustering output meets contract."""
        # Key validation
        missing_keys = ClusteringContract.REQUIRED_KEYS - set(output.keys())
        assert not missing_keys, f"Missing required keys: {missing_keys}"
        
        # Shape validation
        expected_shapes = {
            "motifs_raw": (batch_size, 50),
            "assignment_confidence": (batch_size, 50),
            "hungarian_mapping": (n_clusters,)
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = output[key].shape
            assert actual_shape == expected_shape, f"{key} shape mismatch: {actual_shape} != {expected_shape}"
        
        # Cluster centers shape (variable feature dimension)
        centers_shape = output["cluster_centers"].shape
        assert centers_shape[0] == n_clusters, f"cluster_centers clusters: {centers_shape[0]} != {n_clusters}"
        assert len(centers_shape) == 2, f"cluster_centers must be 2D, got {len(centers_shape)}D"
        
        # Dtype validation
        for key, expected_dtype in ClusteringContract.OUTPUT_DTYPES.items():
            actual_dtype = output[key].dtype
            assert actual_dtype == expected_dtype, f"{key} dtype mismatch: {actual_dtype} != {expected_dtype}"
        
        # Range validation
        motifs = output["motifs_raw"]
        assert motifs.min() >= 0, f"motifs_raw minimum {motifs.min()} < 0"
        assert motifs.max() < n_clusters, f"motifs_raw maximum {motifs.max()} >= {n_clusters}"
        
        confidence = output["assignment_confidence"]
        assert confidence.min() >= 0.0, f"confidence minimum {confidence.min()} < 0.0"
        assert confidence.max() <= 1.0, f"confidence maximum {confidence.max()} > 1.0"
        
        # Finite validation
        assert torch.isfinite(output["cluster_centers"]).all(), "cluster_centers contains NaN/Inf"
        assert torch.isfinite(output["assignment_confidence"]).all(), "assignment_confidence contains NaN/Inf"


# =============================================================================
# SMOOTHING CONTRACT - IMMUTABLE  
# =============================================================================

class SmoothingContract:
    """Frozen temporal smoothing contract."""
    
    # IMMUTABLE PARAMETERS
    DEFAULT_WINDOW_SIZE: int = 7
    DEFAULT_MIN_DURATION: int = 3
    SMOOTHING_METHODS: Tuple[str, ...] = ("median", "mode", "hysteresis")
    
    OUTPUT_SHAPES = {
        "motifs": (None, 50),        # Same as input
        "transitions": (None,),      # (batch_size,)
        "duration_stats": (None, 3), # (n_clusters, 3)
        "smoothing_mask": (None, 50) # Same as input
    }
    
    OUTPUT_DTYPES = {
        "motifs": torch.int32,
        "transitions": torch.int32,
        "duration_stats": torch.float32,
        "smoothing_mask": torch.bool
    }
    
    REQUIRED_KEYS = {"motifs", "transitions", "duration_stats", "smoothing_mask"}
    
    @staticmethod
    def validate_output(
        output: Dict[str, torch.Tensor],
        batch_size: int,
        n_clusters: int,
        input_motifs: torch.Tensor
    ) -> None:
        """Validate smoothing output meets contract."""
        # Key validation
        missing_keys = SmoothingContract.REQUIRED_KEYS - set(output.keys())
        assert not missing_keys, f"Missing required keys: {missing_keys}"
        
        # Shape validation - output shapes must match input
        input_shape = input_motifs.shape
        
        expected_shapes = {
            "motifs": input_shape,
            "transitions": (batch_size,),
            "duration_stats": (n_clusters, 3),
            "smoothing_mask": input_shape
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = output[key].shape
            assert actual_shape == expected_shape, f"{key} shape mismatch: {actual_shape} != {expected_shape}"
        
        # Dtype validation
        for key, expected_dtype in SmoothingContract.OUTPUT_DTYPES.items():
            actual_dtype = output[key].dtype
            assert actual_dtype == expected_dtype, f"{key} dtype mismatch: {actual_dtype} != {expected_dtype}"
        
        # Range validation
        motifs = output["motifs"]
        assert motifs.min() >= 0, f"motifs minimum {motifs.min()} < 0"
        assert motifs.max() < n_clusters, f"motifs maximum {motifs.max()} >= {n_clusters}"
        
        transitions = output["transitions"]
        assert transitions.min() >= 0, f"transitions must be non-negative, got {transitions.min()}"
        
        # Finite validation
        assert torch.isfinite(output["duration_stats"]).all(), "duration_stats contains NaN/Inf"
        
        # Smoothing invariants - should preserve more than it changes
        preserved = (output["motifs"] == input_motifs).sum()
        changed = (output["motifs"] != input_motifs).sum()
        assert preserved >= changed, f"Over-smoothing: {changed} changes > {preserved} preserved"


# =============================================================================
# LABEL MAPPING CONTRACT - IMMUTABLE
# =============================================================================

class LabelContract:
    """Frozen label mapping contract."""
    
    # IMMUTABLE LABEL MAP
    FROZEN_LABELS = {
        0: "rest",
        1: "walk_slow", 
        2: "walk_normal",
        3: "walk_fast",
        4: "trot",
        5: "run",
        6: "turn_left",
        7: "turn_right", 
        8: "sit",
        9: "lie_down",
        10: "play",
        11: "other"
    }
    
    CARDINALITY: int = 12
    RESERVED_IDS = [11]  # "other" category
    
    @staticmethod
    def validate_consistency(motifs: torch.Tensor, label_map_path: Union[str, Path] = None) -> None:
        """Validate motif IDs match frozen label map."""
        if label_map_path and Path(label_map_path).exists():
            # Load external label map for validation
            with open(label_map_path) as f:
                external_map = json.load(f)
            
            assert external_map.get("frozen", False), "External label map must be frozen for production"
            assert external_map.get("cardinality") == LabelContract.CARDINALITY, "Cardinality mismatch with frozen labels"
        
        # Validate against frozen labels
        unique_motifs = torch.unique(motifs)
        max_id = max(LabelContract.FROZEN_LABELS.keys())
        
        assert unique_motifs.max() <= max_id, f"Motif ID {unique_motifs.max()} exceeds max label {max_id}"
        assert unique_motifs.min() >= 0, "Negative motif IDs not allowed"
        
        # Ensure all used IDs have labels
        for motif_id in unique_motifs:
            motif_int = motif_id.item()
            assert motif_int in LabelContract.FROZEN_LABELS, f"No frozen label for motif ID {motif_int}"
    
    @staticmethod
    def get_label(motif_id: int) -> str:
        """Get human-readable label for motif ID."""
        return LabelContract.FROZEN_LABELS.get(motif_id, "unknown")
    
    @staticmethod
    def get_all_labels() -> Dict[int, str]:
        """Get complete frozen label mapping."""
        return LabelContract.FROZEN_LABELS.copy()


# =============================================================================
# CONTRACT ENFORCEMENT DECORATOR
# =============================================================================

def enforce_contract(
    contract_class: Any = None,
    validate_input: bool = True,
    validate_output: bool = True
):
    """
    Decorator to enforce pipeline contracts with runtime validation.
    
    Args:
        contract_class: Contract class with validate methods
        validate_input: Whether to validate inputs
        validate_output: Whether to validate outputs  
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_CONTRACT_VALIDATION:
                return func(*args, **kwargs)
            
            # Pre-condition validation
            if validate_input and contract_class and hasattr(contract_class, 'validate'):
                if args:  # Validate first argument as input
                    contract_class.validate(args[0])
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Post-condition validation
            if validate_output and contract_class and hasattr(contract_class, 'validate_output'):
                # Extract batch size for validation
                batch_size = args[0].shape[0] if args and hasattr(args[0], 'shape') else 1
                
                # Call appropriate validation method
                if contract_class == FSQContract:
                    contract_class.validate_output(result, batch_size)
                elif contract_class == ClusteringContract:
                    n_clusters = kwargs.get('n_clusters', ClusteringContract.DEFAULT_N_CLUSTERS)
                    contract_class.validate_output(result, batch_size, n_clusters)
                elif contract_class == SmoothingContract:
                    n_clusters = 12  # Default, could be inferred from input
                    input_motifs = args[0]
                    contract_class.validate_output(result, batch_size, n_clusters, input_motifs)
            
            return result
        
        return wrapper
    
    if contract_class is None:
        # Used as @enforce_contract() - return decorator
        return decorator
    else:
        # Used as @enforce_contract(ContractClass) - apply decorator
        return decorator


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_pipeline_compatibility(*tensors) -> None:
    """Validate tensor compatibility across pipeline stages."""
    if not ENABLE_CONTRACT_VALIDATION:
        return
    
    batch_sizes = [t.shape[0] for t in tensors if hasattr(t, 'shape') and len(t.shape) > 0]
    
    if batch_sizes:
        assert all(b == batch_sizes[0] for b in batch_sizes), f"Inconsistent batch sizes: {batch_sizes}"


def get_contract_info() -> Dict[str, Any]:
    """Get information about all frozen contracts."""
    return {
        "version": "1.0.0",
        "frozen": True,
        "validation_enabled": ENABLE_CONTRACT_VALIDATION,
        "contracts": {
            "input": {
                "shape": InputContract.SHAPE,
                "dtype": str(InputContract.DTYPE),
                "range": InputContract.RANGE
            },
            "fsq": {
                "levels": FSQContract.LEVELS,
                "codebook_size": FSQContract.CODEBOOK_SIZE,
                "output_shapes": FSQContract.OUTPUT_SHAPES
            },
            "clustering": {
                "default_clusters": ClusteringContract.DEFAULT_N_CLUSTERS,
                "methods": ClusteringContract.CLUSTER_METHODS,
                "min_support": ClusteringContract.MIN_CLUSTER_SUPPORT
            },
            "smoothing": {
                "default_window": SmoothingContract.DEFAULT_WINDOW_SIZE,
                "methods": SmoothingContract.SMOOTHING_METHODS,
                "min_duration": SmoothingContract.DEFAULT_MIN_DURATION
            },
            "labels": {
                "cardinality": LabelContract.CARDINALITY,
                "frozen_labels": LabelContract.FROZEN_LABELS,
                "reserved_ids": LabelContract.RESERVED_IDS
            }
        }
    }


def save_frozen_label_map(output_path: Union[str, Path]) -> None:
    """Save frozen label map to JSON file."""
    label_data = {
        "version": "1.0.0",
        "frozen": True,
        "last_updated": "2025-09-26",
        "label_map": {str(k): v for k, v in LabelContract.FROZEN_LABELS.items()},
        "cardinality": LabelContract.CARDINALITY,
        "reserved_ids": LabelContract.RESERVED_IDS,
        "description": "Frozen behavioral motif labels for production deployment"
    }
    
    with open(output_path, 'w') as f:
        json.dump(label_data, f, indent=2)


# =============================================================================
# CLI CONTRACT VALIDATION
# =============================================================================

def validate_contracts_cli(input_path: str, strict: bool = True) -> bool:
    """CLI function to validate contracts against test data."""
    try:
        # Load test data
        if input_path.endswith('.pt'):
            data = torch.load(input_path)
        elif input_path.endswith('.npy'):
            data = torch.from_numpy(np.load(input_path))
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        # Validate input contract
        InputContract.validate(data)
        print("✅ Input contract validation passed")
        
        # TODO: Add FSQ, clustering, smoothing validation when test outputs available
        
        return True
        
    except Exception as e:
        if strict:
            print(f"❌ Contract validation failed: {e}")
            return False
        else:
            print(f"⚠️  Contract validation warning: {e}")
            return True


if __name__ == "__main__":
    # Print contract information
    import pprint
    pprint.pprint(get_contract_info())