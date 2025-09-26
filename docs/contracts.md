# Conv2d Pipeline Contracts Specification

**Version**: 1.0.0  
**Status**: FROZEN - Breaking changes require major version bump  
**Purpose**: Formal interface contracts for production stability

---

## 📋 Overview

This document defines the **frozen contracts** for all major pipeline components. These contracts are immutable and any breaking changes require a major version increment and migration guide.

## 🎯 Core Principles

1. **Shape Stability**: All tensor shapes are fixed and validated
2. **Dtype Consistency**: Data types are enforced at runtime
3. **Interface Immutability**: Function signatures cannot change
4. **Backward Compatibility**: New features must maintain existing contracts
5. **Test Coverage**: All contracts have corresponding test cases

---

## 🏗️ Pipeline Input Contract

### **Input Data**
```python
# INPUT CONTRACT - IMMUTABLE
input_shape: Tuple[int, int, int, int] = (B, 9, 2, 100)
input_dtype: torch.dtype = torch.float32
input_range: Tuple[float, float] = (-10.0, 10.0)  # Normalized IMU range

# Channel Layout (FROZEN)
channels = {
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

# Spatial Layout (FROZEN)
spatial_dims = {
    "height": 2,     # [raw, filtered] or [left, right] sensor
    "width": 100     # Temporal window (10Hz × 10s)
}
```

### **Validation Contract**
```python
def validate_input(x: torch.Tensor) -> None:
    """Validate input tensor meets pipeline contract."""
    assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}D"
    assert x.shape[1:] == (9, 2, 100), f"Expected (B,9,2,100), got {x.shape}"
    assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
    assert torch.isfinite(x).all(), "Input contains NaN or Inf values"
    assert x.abs().max() <= 10.0, f"Input range violation: {x.abs().max():.2f} > 10.0"
```

---

## 🔢 FSQ Encoding Contract

### **FSQ Function Signature**
```python
def encode_fsq(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    FSQ encoding with FROZEN contract.
    
    Args:
        x: Input tensor [B, 9, 2, 100] float32
        
    Returns:
        Dict with IMMUTABLE keys and shapes:
        {
            "codes": torch.Tensor,      # [B, T] int32, range [0, num_codes)
            "quantized": torch.Tensor,  # [B, D, T] float32, quantized features
            "indices": torch.Tensor,    # [B, T, L] int32, per-level indices  
            "commitment_loss": torch.Tensor,  # [] float32, scalar loss
        }
    """
    pass

# FSQ CONTRACT PARAMETERS - FROZEN
FSQ_LEVELS: Tuple[int, ...] = (4, 4, 4)  # 64 total codes
FSQ_CODEBOOK_SIZE: int = 64
FSQ_FEATURE_DIM: int = 32
FSQ_TEMPORAL_DIM: int = 50  # Downsampled from 100

# Output Shapes - IMMUTABLE
fsq_output_shapes = {
    "codes": (None, 50),        # [B, T] 
    "quantized": (None, 32, 50), # [B, D, T]
    "indices": (None, 50, 3),    # [B, T, L] where L=len(levels)
    "commitment_loss": ()        # scalar
}

# Output Dtypes - IMMUTABLE  
fsq_output_dtypes = {
    "codes": torch.int32,
    "quantized": torch.float32,
    "indices": torch.int32,
    "commitment_loss": torch.float32
}
```

### **FSQ Validation Contract**
```python
def validate_fsq_output(output: Dict[str, torch.Tensor], batch_size: int) -> None:
    """Validate FSQ output meets contract."""
    required_keys = {"codes", "quantized", "indices", "commitment_loss"}
    assert set(output.keys()) >= required_keys, f"Missing keys: {required_keys - set(output.keys())}"
    
    # Shape validation
    assert output["codes"].shape == (batch_size, 50), f"codes shape mismatch: {output['codes'].shape}"
    assert output["quantized"].shape == (batch_size, 32, 50), f"quantized shape mismatch: {output['quantized'].shape}"
    assert output["indices"].shape == (batch_size, 50, 3), f"indices shape mismatch: {output['indices'].shape}"
    assert output["commitment_loss"].shape == (), f"commitment_loss must be scalar: {output['commitment_loss'].shape}"
    
    # Dtype validation
    assert output["codes"].dtype == torch.int32, f"codes dtype: {output['codes'].dtype}"
    assert output["quantized"].dtype == torch.float32, f"quantized dtype: {output['quantized'].dtype}"
    assert output["indices"].dtype == torch.int32, f"indices dtype: {output['indices'].dtype}"
    assert output["commitment_loss"].dtype == torch.float32, f"commitment_loss dtype: {output['commitment_loss'].dtype}"
    
    # Range validation
    assert output["codes"].min() >= 0 and output["codes"].max() < 64, "codes out of range [0, 64)"
    assert output["indices"].min() >= 0, "indices must be non-negative"
    assert torch.isfinite(output["quantized"]).all(), "quantized contains NaN/Inf"
    assert torch.isfinite(output["commitment_loss"]).all(), "commitment_loss contains NaN/Inf"
```

---

## 🎯 Clustering Contract

### **Clustering Function Signature**
```python
def cluster_codes(
    codes: torch.Tensor, 
    method: str = "kmeans",
    n_clusters: int = 12
) -> Dict[str, torch.Tensor]:
    """
    Post-hoc clustering with FROZEN contract.
    
    Args:
        codes: FSQ codes [B, T] int32
        method: Clustering method ("kmeans", "gmm") 
        n_clusters: Number of behavioral motifs
        
    Returns:
        Dict with IMMUTABLE structure:
        {
            "motifs_raw": torch.Tensor,     # [B, T] int32, range [0, n_clusters)
            "cluster_centers": torch.Tensor, # [n_clusters, D] float32
            "assignment_confidence": torch.Tensor, # [B, T] float32, range [0, 1]
            "hungarian_mapping": torch.Tensor, # [n_clusters] int32, optimal label assignment
        }
    """
    pass

# CLUSTERING CONTRACT PARAMETERS - FROZEN
DEFAULT_N_CLUSTERS: int = 12
MIN_CLUSTER_SUPPORT: float = 0.005  # 0.5% minimum samples per cluster
CLUSTER_METHODS: Tuple[str, ...] = ("kmeans", "gmm")

# Output Shapes - IMMUTABLE
clustering_output_shapes = {
    "motifs_raw": (None, 50),           # [B, T]
    "cluster_centers": (12, None),      # [n_clusters, feature_dim] 
    "assignment_confidence": (None, 50), # [B, T]
    "hungarian_mapping": (12,)          # [n_clusters]
}

# Output Dtypes - IMMUTABLE
clustering_output_dtypes = {
    "motifs_raw": torch.int32,
    "cluster_centers": torch.float32, 
    "assignment_confidence": torch.float32,
    "hungarian_mapping": torch.int32
}
```

### **Clustering Validation Contract**
```python
def validate_clustering_output(output: Dict[str, torch.Tensor], batch_size: int, n_clusters: int) -> None:
    """Validate clustering output meets contract."""
    required_keys = {"motifs_raw", "cluster_centers", "assignment_confidence", "hungarian_mapping"}
    assert set(output.keys()) >= required_keys, f"Missing keys: {required_keys - set(output.keys())}"
    
    # Shape validation
    assert output["motifs_raw"].shape == (batch_size, 50), f"motifs_raw shape: {output['motifs_raw'].shape}"
    assert output["cluster_centers"].shape[0] == n_clusters, f"cluster_centers clusters: {output['cluster_centers'].shape[0]}"
    assert output["assignment_confidence"].shape == (batch_size, 50), f"confidence shape: {output['assignment_confidence'].shape}"
    assert output["hungarian_mapping"].shape == (n_clusters,), f"hungarian_mapping shape: {output['hungarian_mapping'].shape}"
    
    # Dtype validation
    for key, expected_dtype in clustering_output_dtypes.items():
        assert output[key].dtype == expected_dtype, f"{key} dtype: {output[key].dtype} != {expected_dtype}"
    
    # Range validation
    assert output["motifs_raw"].min() >= 0 and output["motifs_raw"].max() < n_clusters, f"motifs_raw range: [{output['motifs_raw'].min()}, {output['motifs_raw'].max()}]"
    assert (output["assignment_confidence"] >= 0).all() and (output["assignment_confidence"] <= 1).all(), "confidence not in [0,1]"
    assert torch.isfinite(output["cluster_centers"]).all(), "cluster_centers contains NaN/Inf"
```

---

## 🌊 Temporal Smoothing Contract

### **Smoothing Function Signature**
```python
def smooth_motifs(
    motifs_raw: torch.Tensor,
    method: str = "median", 
    window_size: int = 7,
    min_duration: int = 3
) -> Dict[str, torch.Tensor]:
    """
    Temporal smoothing with FROZEN contract.
    
    Args:
        motifs_raw: Raw cluster assignments [B, T] int32
        method: Smoothing method ("median", "mode", "hysteresis")
        window_size: Filter window size (frames)
        min_duration: Minimum motif duration (frames)
        
    Returns:
        Dict with IMMUTABLE structure:
        {
            "motifs": torch.Tensor,         # [B, T] int32, final smoothed motifs
            "transitions": torch.Tensor,    # [B] int32, number of transitions
            "duration_stats": torch.Tensor, # [n_clusters, 3] float32, [mean, std, count]
            "smoothing_mask": torch.Tensor, # [B, T] bool, which frames were modified
        }
    """
    pass

# SMOOTHING CONTRACT PARAMETERS - FROZEN
DEFAULT_WINDOW_SIZE: int = 7
DEFAULT_MIN_DURATION: int = 3 
SMOOTHING_METHODS: Tuple[str, ...] = ("median", "mode", "hysteresis")

# Output Shapes - IMMUTABLE
smoothing_output_shapes = {
    "motifs": (None, 50),        # [B, T] - same as motifs_raw
    "transitions": (None,),      # [B] - per-sequence transition count
    "duration_stats": (None, 3), # [n_clusters, 3] - [mean, std, count]
    "smoothing_mask": (None, 50) # [B, T] - modification mask
}

# Output Dtypes - IMMUTABLE
smoothing_output_dtypes = {
    "motifs": torch.int32,
    "transitions": torch.int32,
    "duration_stats": torch.float32,
    "smoothing_mask": torch.bool
}
```

### **Smoothing Validation Contract**
```python
def validate_smoothing_output(
    output: Dict[str, torch.Tensor], 
    batch_size: int, 
    n_clusters: int,
    input_motifs: torch.Tensor
) -> None:
    """Validate smoothing output meets contract."""
    required_keys = {"motifs", "transitions", "duration_stats", "smoothing_mask"}
    assert set(output.keys()) >= required_keys, f"Missing keys: {required_keys - set(output.keys())}"
    
    # Shape validation
    assert output["motifs"].shape == input_motifs.shape, f"motifs shape changed: {output['motifs'].shape} != {input_motifs.shape}"
    assert output["transitions"].shape == (batch_size,), f"transitions shape: {output['transitions'].shape}"
    assert output["duration_stats"].shape == (n_clusters, 3), f"duration_stats shape: {output['duration_stats'].shape}"
    assert output["smoothing_mask"].shape == input_motifs.shape, f"smoothing_mask shape: {output['smoothing_mask'].shape}"
    
    # Dtype validation
    for key, expected_dtype in smoothing_output_dtypes.items():
        assert output[key].dtype == expected_dtype, f"{key} dtype: {output[key].dtype} != {expected_dtype}"
    
    # Range validation
    assert output["motifs"].min() >= 0 and output["motifs"].max() < n_clusters, "motifs range violation"
    assert (output["transitions"] >= 0).all(), "negative transitions"
    assert torch.isfinite(output["duration_stats"]).all(), "duration_stats contains NaN/Inf"
    
    # Smoothing invariants
    assert (output["motifs"] == input_motifs).sum() >= (output["motifs"] != input_motifs).sum(), "Over-smoothing: more changes than preserved"
```

---

## 📊 Label Mapping Contract

### **Frozen Label Map**
```json
{
    "version": "1.0.0",
    "frozen": true,
    "last_updated": "2025-09-26",
    "label_map": {
        "0": "rest",
        "1": "walk_slow", 
        "2": "walk_normal",
        "3": "walk_fast",
        "4": "trot",
        "5": "run",
        "6": "turn_left",
        "7": "turn_right", 
        "8": "sit",
        "9": "lie_down",
        "10": "play",
        "11": "other"
    },
    "cardinality": 12,
    "reserved_ids": [11],
    "description": "Frozen behavioral motif labels for production deployment"
}
```

### **Label Contract Validation**
```python
def validate_label_consistency(motifs: torch.Tensor, label_map_path: str) -> None:
    """Validate motif IDs match frozen label map."""
    with open(label_map_path) as f:
        label_map = json.load(f)
    
    assert label_map["frozen"], "Label map must be frozen for production"
    assert label_map["cardinality"] == len(label_map["label_map"]), "Cardinality mismatch"
    
    unique_motifs = torch.unique(motifs)
    max_id = max(int(k) for k in label_map["label_map"].keys())
    
    assert unique_motifs.max() <= max_id, f"Motif ID {unique_motifs.max()} exceeds max label {max_id}"
    assert unique_motifs.min() >= 0, "Negative motif IDs not allowed"
    
    # Ensure all used IDs have labels
    for motif_id in unique_motifs:
        assert str(motif_id.item()) in label_map["label_map"], f"No label for motif ID {motif_id}"
```

---

## 🧪 Contract Testing Framework

### **Test Structure**
```python
class TestPipelineContracts:
    """Comprehensive contract testing for production stability."""
    
    def test_input_contract(self):
        """Test input validation contract."""
        # Valid input
        valid_input = torch.randn(2, 9, 2, 100, dtype=torch.float32)
        validate_input(valid_input)  # Should not raise
        
        # Invalid shapes
        with pytest.raises(AssertionError, match="Expected 4D tensor"):
            validate_input(torch.randn(9, 2, 100))
            
        # Invalid dtype
        with pytest.raises(AssertionError, match="Expected float32"):
            validate_input(torch.randn(2, 9, 2, 100, dtype=torch.float64))
            
        # Invalid range
        with pytest.raises(AssertionError, match="Input range violation"):
            invalid_input = torch.randn(2, 9, 2, 100) * 20  # > 10.0
            validate_input(invalid_input)
    
    def test_fsq_contract(self):
        """Test FSQ encoding contract."""
        x = torch.randn(4, 9, 2, 100, dtype=torch.float32)
        output = encode_fsq(x)
        validate_fsq_output(output, batch_size=4)
    
    def test_clustering_contract(self):
        """Test clustering contract.""" 
        codes = torch.randint(0, 64, (4, 50), dtype=torch.int32)
        output = cluster_codes(codes, n_clusters=12)
        validate_clustering_output(output, batch_size=4, n_clusters=12)
    
    def test_smoothing_contract(self):
        """Test temporal smoothing contract."""
        motifs_raw = torch.randint(0, 12, (4, 50), dtype=torch.int32)
        output = smooth_motifs(motifs_raw)
        validate_smoothing_output(output, batch_size=4, n_clusters=12, input_motifs=motifs_raw)
    
    def test_end_to_end_contract(self):
        """Test full pipeline contract compatibility."""
        # Input
        x = torch.randn(2, 9, 2, 100, dtype=torch.float32)
        validate_input(x)
        
        # FSQ encoding
        fsq_out = encode_fsq(x)
        validate_fsq_output(fsq_out, batch_size=2)
        
        # Clustering
        cluster_out = cluster_codes(fsq_out["codes"])
        validate_clustering_output(cluster_out, batch_size=2, n_clusters=12)
        
        # Smoothing
        smooth_out = smooth_motifs(cluster_out["motifs_raw"])
        validate_smoothing_output(smooth_out, batch_size=2, n_clusters=12, input_motifs=cluster_out["motifs_raw"])
        
        # Label consistency
        validate_label_consistency(smooth_out["motifs"], "label_map.json")
```

---

## 🚀 Contract Enforcement

### **Runtime Validation**
```python
# Enable contract validation in production
ENABLE_CONTRACT_VALIDATION = os.getenv("CONV2D_VALIDATE_CONTRACTS", "true").lower() == "true"

def enforce_contract(func: Callable) -> Callable:
    """Decorator to enforce pipeline contracts."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if ENABLE_CONTRACT_VALIDATION:
            # Pre-condition validation
            validate_inputs(func, args, kwargs)
            
        result = func(*args, **kwargs) 
        
        if ENABLE_CONTRACT_VALIDATION:
            # Post-condition validation
            validate_outputs(func, result)
            
        return result
    return wrapper

# Apply to all pipeline functions
@enforce_contract
def encode_fsq(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    # Implementation here
    pass
```

### **CLI Contract Validation**
```bash
# Validate contracts during CLI operations
conv2d validate-contracts --input data/test_batch.pt --strict

# Test contract compliance
conv2d test --contracts-only --coverage

# Generate contract documentation  
conv2d docs --contracts --output docs/contracts_generated.md
```

---

## 📝 Contract Change Process

### **Version Compatibility Matrix**

| Change Type | Version Bump | Backward Compatible | Migration Required |
|-------------|--------------|--------------------|--------------------|
| Add optional field | Patch | ✅ Yes | ❌ No |
| Change default value | Minor | ✅ Yes | ❌ No |  
| Add new method | Minor | ✅ Yes | ❌ No |
| Change shape/dtype | Major | ❌ No | ✅ Yes |
| Remove field/method | Major | ❌ No | ✅ Yes |
| Rename interface | Major | ❌ No | ✅ Yes |

### **Breaking Change Protocol**

1. **Deprecation Warning**: Add warnings 1 version before removal
2. **Migration Guide**: Provide automated migration scripts
3. **Testing**: Validate backward compatibility with contract tests  
4. **Documentation**: Update all references and examples
5. **Rollout**: Staged deployment with rollback capability

---

## ✅ Contract Compliance Checklist

- [ ] All shapes and dtypes are explicitly validated
- [ ] Function signatures are frozen and documented  
- [ ] Test cases cover all contract violations
- [ ] Label map is frozen and version-controlled
- [ ] Runtime validation can be enabled/disabled
- [ ] Contract documentation is auto-generated
- [ ] Breaking changes follow deprecation protocol

---

**🔒 CONTRACT FREEZE NOTICE**: This specification is FROZEN for production stability. Any modifications require approval and version increment following the change process above.

*Last Updated: 2025-09-26 | Version: 1.0.0*