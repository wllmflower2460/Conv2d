"""
Comprehensive contract testing for production stability.

These tests enforce the FROZEN contracts defined in docs/contracts.md
and implemented in src/conv2d/contracts.py.
"""

import json
import pytest
import torch
import numpy as np
from pathlib import Path

from src.conv2d.contracts import (
    InputContract,
    FSQContract, 
    ClusteringContract,
    SmoothingContract,
    LabelContract,
    enforce_contract,
    validate_pipeline_compatibility,
    get_contract_info,
    save_frozen_label_map
)


class TestInputContract:
    """Test frozen input contract validation."""
    
    def test_valid_input(self):
        """Test valid input passes validation."""
        valid_input = torch.randn(4, 9, 2, 100, dtype=torch.float32)
        InputContract.validate(valid_input)  # Should not raise
    
    def test_invalid_ndim(self):
        """Test invalid number of dimensions."""
        with pytest.raises(AssertionError, match="Expected 4D tensor"):
            InputContract.validate(torch.randn(9, 2, 100))
    
    def test_invalid_shape(self):
        """Test invalid tensor shape.""" 
        with pytest.raises(AssertionError, match="Expected \\(B,9,2,100\\)"):
            InputContract.validate(torch.randn(4, 8, 2, 100))
        
        with pytest.raises(AssertionError, match="Expected \\(B,9,2,100\\)"):
            InputContract.validate(torch.randn(4, 9, 3, 100))
            
        with pytest.raises(AssertionError, match="Expected \\(B,9,2,100\\)"):
            InputContract.validate(torch.randn(4, 9, 2, 50))
    
    def test_invalid_dtype(self):
        """Test invalid data type."""
        with pytest.raises(AssertionError, match="Expected float32"):
            InputContract.validate(torch.randn(4, 9, 2, 100, dtype=torch.float64))
        
        with pytest.raises(AssertionError, match="Expected float32"):
            InputContract.validate(torch.randint(0, 10, (4, 9, 2, 100)))
    
    def test_invalid_values(self):
        """Test invalid value ranges."""
        # NaN values
        invalid_input = torch.randn(4, 9, 2, 100)
        invalid_input[0, 0, 0, 0] = float('nan')
        with pytest.raises(AssertionError, match="contains NaN or Inf"):
            InputContract.validate(invalid_input)
        
        # Inf values  
        invalid_input = torch.randn(4, 9, 2, 100)
        invalid_input[0, 0, 0, 0] = float('inf')
        with pytest.raises(AssertionError, match="contains NaN or Inf"):
            InputContract.validate(invalid_input)
        
        # Out of range values
        invalid_input = torch.randn(4, 9, 2, 100) * 20  # Scale to > 10.0
        with pytest.raises(AssertionError, match="Input range violation"):
            InputContract.validate(invalid_input)
    
    def test_boundary_values(self):
        """Test boundary value handling."""
        # Exactly at boundary should pass
        boundary_input = torch.full((4, 9, 2, 100), 10.0, dtype=torch.float32)
        InputContract.validate(boundary_input)
        
        # Just over boundary should fail
        over_boundary = torch.full((4, 9, 2, 100), 10.1, dtype=torch.float32)
        with pytest.raises(AssertionError, match="Input range violation"):
            InputContract.validate(over_boundary)


class TestFSQContract:
    """Test frozen FSQ encoding contract."""
    
    def create_valid_fsq_output(self, batch_size: int = 4):
        """Create valid FSQ output for testing."""
        return {
            "codes": torch.randint(0, 64, (batch_size, 50), dtype=torch.int32),
            "quantized": torch.randn(batch_size, 32, 50, dtype=torch.float32),
            "indices": torch.randint(0, 4, (batch_size, 50, 3), dtype=torch.int32),
            "commitment_loss": torch.tensor(0.5, dtype=torch.float32)
        }
    
    def test_valid_output(self):
        """Test valid FSQ output passes validation."""
        output = self.create_valid_fsq_output(batch_size=4)
        FSQContract.validate_output(output, batch_size=4)
    
    def test_missing_keys(self):
        """Test missing required keys."""
        output = self.create_valid_fsq_output()
        del output["codes"]
        
        with pytest.raises(AssertionError, match="Missing required keys"):
            FSQContract.validate_output(output, batch_size=4)
    
    def test_invalid_shapes(self):
        """Test invalid output shapes."""
        # Wrong codes shape
        output = self.create_valid_fsq_output()
        output["codes"] = torch.randint(0, 64, (4, 25), dtype=torch.int32)  # Wrong temporal dim
        
        with pytest.raises(AssertionError, match="codes shape mismatch"):
            FSQContract.validate_output(output, batch_size=4)
        
        # Wrong quantized shape
        output = self.create_valid_fsq_output()
        output["quantized"] = torch.randn(4, 16, 50, dtype=torch.float32)  # Wrong feature dim
        
        with pytest.raises(AssertionError, match="quantized shape mismatch"):
            FSQContract.validate_output(output, batch_size=4)
        
        # Wrong commitment loss shape
        output = self.create_valid_fsq_output()
        output["commitment_loss"] = torch.tensor([0.5, 0.3], dtype=torch.float32)  # Should be scalar
        
        with pytest.raises(AssertionError, match="commitment_loss must be scalar"):
            FSQContract.validate_output(output, batch_size=4)
    
    def test_invalid_dtypes(self):
        """Test invalid data types."""
        output = self.create_valid_fsq_output()
        output["codes"] = output["codes"].float()  # Should be int32
        
        with pytest.raises(AssertionError, match="codes dtype mismatch"):
            FSQContract.validate_output(output, batch_size=4)
    
    def test_invalid_ranges(self):
        """Test invalid value ranges."""
        # Codes out of range
        output = self.create_valid_fsq_output()
        output["codes"][0, 0] = 64  # >= codebook size
        
        with pytest.raises(AssertionError, match="codes maximum"):
            FSQContract.validate_output(output, batch_size=4)
        
        output = self.create_valid_fsq_output()
        output["codes"][0, 0] = -1  # < 0
        
        with pytest.raises(AssertionError, match="codes minimum"):
            FSQContract.validate_output(output, batch_size=4)
        
        # NaN in quantized
        output = self.create_valid_fsq_output()
        output["quantized"][0, 0, 0] = float('nan')
        
        with pytest.raises(AssertionError, match="quantized contains NaN/Inf"):
            FSQContract.validate_output(output, batch_size=4)


class TestClusteringContract:
    """Test frozen clustering contract."""
    
    def create_valid_clustering_output(self, batch_size: int = 4, n_clusters: int = 12):
        """Create valid clustering output for testing."""
        return {
            "motifs_raw": torch.randint(0, n_clusters, (batch_size, 50), dtype=torch.int32),
            "cluster_centers": torch.randn(n_clusters, 32, dtype=torch.float32),
            "assignment_confidence": torch.rand(batch_size, 50, dtype=torch.float32),
            "hungarian_mapping": torch.arange(n_clusters, dtype=torch.int32)
        }
    
    def test_valid_output(self):
        """Test valid clustering output passes validation."""
        output = self.create_valid_clustering_output()
        ClusteringContract.validate_output(output, batch_size=4, n_clusters=12)
    
    def test_invalid_motif_ranges(self):
        """Test invalid motif ID ranges."""
        output = self.create_valid_clustering_output(n_clusters=12)
        output["motifs_raw"][0, 0] = 12  # >= n_clusters
        
        with pytest.raises(AssertionError, match="motifs_raw maximum"):
            ClusteringContract.validate_output(output, batch_size=4, n_clusters=12)
        
        output = self.create_valid_clustering_output()
        output["motifs_raw"][0, 0] = -1  # < 0
        
        with pytest.raises(AssertionError, match="motifs_raw minimum"):
            ClusteringContract.validate_output(output, batch_size=4, n_clusters=12)
    
    def test_invalid_confidence_ranges(self):
        """Test invalid confidence ranges.""" 
        output = self.create_valid_clustering_output()
        output["assignment_confidence"][0, 0] = 1.5  # > 1.0
        
        with pytest.raises(AssertionError, match="confidence maximum"):
            ClusteringContract.validate_output(output, batch_size=4, n_clusters=12)
        
        output = self.create_valid_clustering_output()
        output["assignment_confidence"][0, 0] = -0.1  # < 0.0
        
        with pytest.raises(AssertionError, match="confidence minimum"):
            ClusteringContract.validate_output(output, batch_size=4, n_clusters=12)


class TestSmoothingContract:
    """Test frozen temporal smoothing contract."""
    
    def create_valid_smoothing_output(
        self, 
        batch_size: int = 4, 
        n_clusters: int = 12,
        input_motifs: torch.Tensor = None
    ):
        """Create valid smoothing output for testing.""" 
        if input_motifs is None:
            input_motifs = torch.randint(0, n_clusters, (batch_size, 50), dtype=torch.int32)
        
        # Create smoothed version (small changes)
        smoothed = input_motifs.clone()
        smoothed[0, 0] = (smoothed[0, 0] + 1) % n_clusters  # Small change
        
        return {
            "motifs": smoothed,
            "transitions": torch.randint(5, 15, (batch_size,), dtype=torch.int32),
            "duration_stats": torch.rand(n_clusters, 3, dtype=torch.float32) * 10 + 1,
            "smoothing_mask": torch.randint(0, 2, (batch_size, 50), dtype=torch.bool)
        }
    
    def test_valid_output(self):
        """Test valid smoothing output passes validation."""
        input_motifs = torch.randint(0, 12, (4, 50), dtype=torch.int32)
        output = self.create_valid_smoothing_output(input_motifs=input_motifs)
        
        SmoothingContract.validate_output(
            output, batch_size=4, n_clusters=12, input_motifs=input_motifs
        )
    
    def test_shape_preservation(self):
        """Test that smoothing preserves input shapes."""
        input_motifs = torch.randint(0, 12, (4, 50), dtype=torch.int32)
        output = self.create_valid_smoothing_output(input_motifs=input_motifs)
        
        # Change shape should fail
        output["motifs"] = torch.randint(0, 12, (4, 25), dtype=torch.int32)
        
        with pytest.raises(AssertionError, match="motifs shape mismatch"):
            SmoothingContract.validate_output(
                output, batch_size=4, n_clusters=12, input_motifs=input_motifs
            )
    
    def test_over_smoothing_protection(self):
        """Test protection against over-smoothing."""
        input_motifs = torch.randint(0, 12, (4, 50), dtype=torch.int32)
        
        # Create output where everything is changed (over-smoothing)
        output = self.create_valid_smoothing_output(input_motifs=input_motifs)
        output["motifs"] = (input_motifs + 1) % 12  # Change everything
        
        with pytest.raises(AssertionError, match="Over-smoothing"):
            SmoothingContract.validate_output(
                output, batch_size=4, n_clusters=12, input_motifs=input_motifs
            )


class TestLabelContract:
    """Test frozen label mapping contract."""
    
    def test_valid_motifs(self):
        """Test valid motif IDs pass validation."""
        valid_motifs = torch.tensor([0, 1, 2, 5, 8, 11], dtype=torch.int32)
        LabelContract.validate_consistency(valid_motifs)
    
    def test_invalid_motif_ranges(self):
        """Test invalid motif ID ranges."""
        # Negative IDs
        with pytest.raises(AssertionError, match="Negative motif IDs"):
            invalid_motifs = torch.tensor([-1, 0, 1], dtype=torch.int32)
            LabelContract.validate_consistency(invalid_motifs)
        
        # IDs exceeding max label
        with pytest.raises(AssertionError, match="exceeds max label"):
            invalid_motifs = torch.tensor([0, 1, 15], dtype=torch.int32)  # 15 > 11
            LabelContract.validate_consistency(invalid_motifs)
    
    def test_unknown_motif_ids(self):
        """Test that all motif IDs have corresponding labels.""" 
        # This should pass since all IDs 0-11 are defined
        valid_motifs = torch.arange(12, dtype=torch.int32)
        LabelContract.validate_consistency(valid_motifs)
    
    def test_label_lookup(self):
        """Test label lookup functionality."""
        assert LabelContract.get_label(0) == "rest"
        assert LabelContract.get_label(5) == "run"
        assert LabelContract.get_label(11) == "other"
        assert LabelContract.get_label(99) == "unknown"  # Not in frozen map
    
    def test_frozen_labels_immutable(self):
        """Test that frozen labels are complete and immutable."""
        labels = LabelContract.get_all_labels()
        
        assert len(labels) == LabelContract.CARDINALITY
        assert all(isinstance(k, int) for k in labels.keys())
        assert all(isinstance(v, str) for v in labels.values())
        
        # Verify all expected labels exist
        expected_ids = set(range(12))
        actual_ids = set(labels.keys())
        assert expected_ids == actual_ids
    
    def test_external_label_map_validation(self, tmp_path):
        """Test validation against external label map file."""
        # Create valid external label map
        valid_map = {
            "version": "1.0.0",
            "frozen": True,
            "cardinality": 12,
            "label_map": {str(k): v for k, v in LabelContract.FROZEN_LABELS.items()}
        }
        
        map_file = tmp_path / "labels.json"
        with open(map_file, 'w') as f:
            json.dump(valid_map, f)
        
        valid_motifs = torch.tensor([0, 1, 2], dtype=torch.int32)
        LabelContract.validate_consistency(valid_motifs, str(map_file))
        
        # Test unfrozen external map should fail
        valid_map["frozen"] = False
        with open(map_file, 'w') as f:
            json.dump(valid_map, f)
        
        with pytest.raises(AssertionError, match="must be frozen"):
            LabelContract.validate_consistency(valid_motifs, str(map_file))


class TestContractEnforcement:
    """Test contract enforcement decorator."""
    
    def test_decorator_with_validation_enabled(self, monkeypatch):
        """Test decorator enforces contracts when validation is enabled."""
        monkeypatch.setenv("CONV2D_VALIDATE_CONTRACTS", "true")
        
        @enforce_contract(InputContract, validate_input=True)
        def dummy_function(x):
            return x * 2
        
        # Valid input should pass
        valid_input = torch.randn(4, 9, 2, 100, dtype=torch.float32)
        result = dummy_function(valid_input)
        assert torch.allclose(result, valid_input * 2)
        
        # Invalid input should fail
        with pytest.raises(AssertionError):
            invalid_input = torch.randn(4, 8, 2, 100, dtype=torch.float32)  # Wrong shape
            dummy_function(invalid_input)
    
    def test_decorator_with_validation_disabled(self, monkeypatch):
        """Test decorator bypasses validation when disabled."""
        monkeypatch.setenv("CONV2D_VALIDATE_CONTRACTS", "false")
        
        @enforce_contract(InputContract, validate_input=True)
        def dummy_function(x):
            return x * 2
        
        # Invalid input should pass when validation disabled
        invalid_input = torch.randn(4, 8, 2, 100, dtype=torch.float32)  # Wrong shape
        result = dummy_function(invalid_input)
        assert torch.allclose(result, invalid_input * 2)


class TestPipelineCompatibility:
    """Test pipeline compatibility validation."""
    
    def test_compatible_batch_sizes(self):
        """Test compatible batch sizes pass validation."""
        tensor1 = torch.randn(4, 9, 2, 100)
        tensor2 = torch.randn(4, 50)
        tensor3 = torch.randn(4, 12, 3)
        
        validate_pipeline_compatibility(tensor1, tensor2, tensor3)
    
    def test_incompatible_batch_sizes(self):
        """Test incompatible batch sizes fail validation."""
        tensor1 = torch.randn(4, 9, 2, 100)
        tensor2 = torch.randn(3, 50)  # Different batch size
        
        with pytest.raises(AssertionError, match="Inconsistent batch sizes"):
            validate_pipeline_compatibility(tensor1, tensor2)


class TestContractUtilities:
    """Test contract utility functions."""
    
    def test_contract_info(self):
        """Test contract information retrieval."""
        info = get_contract_info()
        
        assert info["version"] == "1.0.0"
        assert info["frozen"] is True
        assert "contracts" in info
        assert "input" in info["contracts"]
        assert "fsq" in info["contracts"]
        assert "clustering" in info["contracts"]
        assert "smoothing" in info["contracts"]
        assert "labels" in info["contracts"]
    
    def test_save_frozen_label_map(self, tmp_path):
        """Test saving frozen label map."""
        output_file = tmp_path / "test_labels.json"
        save_frozen_label_map(output_file)
        
        assert output_file.exists()
        
        with open(output_file) as f:
            saved_map = json.load(f)
        
        assert saved_map["frozen"] is True
        assert saved_map["cardinality"] == 12
        assert len(saved_map["label_map"]) == 12
        assert saved_map["label_map"]["0"] == "rest"
        assert saved_map["label_map"]["11"] == "other"


class TestContractVersioning:
    """Test contract versioning and compatibility."""
    
    def test_contract_constants_immutable(self):
        """Test that contract constants match documentation."""
        # Input contract
        assert InputContract.SHAPE == (None, 9, 2, 100)
        assert InputContract.DTYPE == torch.float32
        assert InputContract.RANGE == (-10.0, 10.0)
        
        # FSQ contract
        assert FSQContract.LEVELS == (4, 4, 4)
        assert FSQContract.CODEBOOK_SIZE == 64
        assert FSQContract.FEATURE_DIM == 32
        assert FSQContract.TEMPORAL_DIM == 50
        
        # Clustering contract
        assert ClusteringContract.DEFAULT_N_CLUSTERS == 12
        assert ClusteringContract.MIN_CLUSTER_SUPPORT == 0.005
        
        # Smoothing contract
        assert SmoothingContract.DEFAULT_WINDOW_SIZE == 7
        assert SmoothingContract.DEFAULT_MIN_DURATION == 3
        
        # Label contract
        assert LabelContract.CARDINALITY == 12
        assert LabelContract.RESERVED_IDS == [11]
    
    def test_frozen_label_consistency(self):
        """Test frozen labels match expected behavioral categories."""
        labels = LabelContract.FROZEN_LABELS
        
        # Verify key behavioral categories are present
        assert labels[0] == "rest"
        assert labels[5] == "run"
        assert labels[8] == "sit"
        assert labels[11] == "other"  # Reserved category
        
        # Verify locomotion progression
        locomotion_sequence = [labels[1], labels[2], labels[3], labels[4], labels[5]]
        expected_sequence = ["walk_slow", "walk_normal", "walk_fast", "trot", "run"]
        assert locomotion_sequence == expected_sequence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])