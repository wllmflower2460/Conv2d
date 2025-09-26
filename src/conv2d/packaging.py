"""
Conv2d Artifact Packaging System
================================

Standardized artifact bundling for deployment with:
- Single directory per experiment (artifacts/EXP_HASH/)
- Mandatory files: config.yaml, metrics.json, label_map.json, VERSION, COMMIT_SHA
- Format-specific models: model.onnx, coreml.mlpackage, hailo.hef
- Bundle verification and validation

Architecture:
    Bundle → Validation → Packaging → Verification → Deploy
"""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml

from ..contracts import LabelContract


class ArtifactBundle:
    """Standardized artifact bundle for Conv2d deployments."""
    
    REQUIRED_FILES = [
        "config.yaml",
        "metrics.json", 
        "label_map.json",
        "VERSION",
        "COMMIT_SHA"
    ]
    
    MODEL_FORMATS = {
        "onnx": "model.onnx",
        "coreml": "coreml.mlpackage",
        "hailo": "hailo.hef"
    }
    
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir)
        self.bundle_hash = self.bundle_dir.name
        
    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        metrics: Dict[str, Any], 
        model_path: Path,
        model_format: str,
        output_dir: Path = Path("artifacts"),
        bundle_name: Optional[str] = None
    ) -> "ArtifactBundle":
        """Create new artifact bundle."""
        
        # Generate experiment hash from config
        config_str = json.dumps(config, sort_keys=True)
        exp_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
        
        # Create bundle directory
        bundle_dir = output_dir / exp_hash
        bundle_dir.mkdir(parents=True, exist_ok=True)
        
        bundle = cls(bundle_dir)
        
        # Write required files
        bundle._write_config(config)
        bundle._write_metrics(metrics)
        bundle._write_label_map()
        bundle._write_version()
        bundle._write_commit_sha()
        
        # Copy model file
        bundle._copy_model(model_path, model_format)
        
        # Validate bundle
        bundle.verify()
        
        return bundle
    
    def _write_config(self, config: Dict[str, Any]):
        """Write config.yaml."""
        config_path = self.bundle_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
    
    def _write_metrics(self, metrics: Dict[str, Any]):
        """Write metrics.json."""
        # Ensure metrics include bundle metadata
        enhanced_metrics = {
            **metrics,
            "bundle_metadata": {
                "bundle_hash": self.bundle_hash,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "schema_version": "1.0"
            }
        }
        
        metrics_path = self.bundle_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(enhanced_metrics, f, indent=2)
    
    def _write_label_map(self):
        """Write frozen label_map.json."""
        # Use the frozen label map from contracts
        label_map_source = Path(__file__).parent.parent.parent / "label_map.json"
        label_map_dest = self.bundle_dir / "label_map.json"
        
        if label_map_source.exists():
            shutil.copy2(label_map_source, label_map_dest)
        else:
            # Fallback: create from contract
            label_contract = LabelContract()
            with open(label_map_dest, 'w') as f:
                json.dump({
                    "version": "1.0.0",
                    "frozen": True,
                    "last_updated": datetime.utcnow().strftime("%Y-%m-%d"),
                    "label_map": label_contract.label_map,
                    "cardinality": label_contract.cardinality,
                    "reserved_ids": [11],
                    "description": "Frozen behavioral motif labels for production deployment. DO NOT MODIFY - breaking changes require major version bump."
                }, f, indent=2)
    
    def _write_version(self):
        """Write VERSION file."""
        version_path = self.bundle_dir / "VERSION"
        
        # Try to get version from git tags, fallback to timestamp
        try:
            version = subprocess.check_output(
                ["git", "describe", "--tags", "--always"],
                cwd=self.bundle_dir.parent.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            version = f"dev-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        with open(version_path, 'w') as f:
            f.write(f"{version}\n")
    
    def _write_commit_sha(self):
        """Write COMMIT_SHA file."""
        commit_path = self.bundle_dir / "COMMIT_SHA"
        
        # Get current commit SHA
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.bundle_dir.parent.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit_sha = "unknown"
        
        with open(commit_path, 'w') as f:
            f.write(f"{commit_sha}\n")
    
    def _copy_model(self, model_path: Path, model_format: str):
        """Copy model file to bundle."""
        if model_format not in self.MODEL_FORMATS:
            raise ValueError(f"Unsupported model format: {model_format}. Supported: {list(self.MODEL_FORMATS.keys())}")
        
        dest_name = self.MODEL_FORMATS[model_format]
        dest_path = self.bundle_dir / dest_name
        
        if model_path.is_dir():
            # For directories (e.g., CoreML packages)
            shutil.copytree(model_path, dest_path, dirs_exist_ok=True)
        else:
            # For files
            shutil.copy2(model_path, dest_path)
    
    def verify(self) -> Dict[str, Any]:
        """Verify bundle structure and contents."""
        verification = {
            "bundle_hash": self.bundle_hash,
            "bundle_dir": str(self.bundle_dir),
            "verified_at": datetime.utcnow().isoformat() + "Z",
            "checks": {},
            "valid": True,
            "errors": []
        }
        
        # Check required files exist
        for required_file in self.REQUIRED_FILES:
            file_path = self.bundle_dir / required_file
            verification["checks"][required_file] = file_path.exists()
            if not file_path.exists():
                verification["valid"] = False
                verification["errors"].append(f"Missing required file: {required_file}")
        
        # Check at least one model format exists
        model_files = []
        for format_name, filename in self.MODEL_FORMATS.items():
            file_path = self.bundle_dir / filename
            if file_path.exists():
                model_files.append(format_name)
                verification["checks"][f"model_{format_name}"] = True
        
        if not model_files:
            verification["valid"] = False
            verification["errors"].append("No model files found. At least one format required.")
        else:
            verification["checks"]["has_model"] = True
            verification["model_formats"] = model_files
        
        # Validate label map matches contract
        if self.bundle_dir / "label_map.json" in [self.bundle_dir / f for f in self.REQUIRED_FILES if (self.bundle_dir / f).exists()]:
            try:
                with open(self.bundle_dir / "label_map.json", 'r') as f:
                    label_data = json.load(f)
                
                # Basic validation
                if "label_map" not in label_data:
                    verification["errors"].append("label_map.json missing 'label_map' field")
                    verification["valid"] = False
                elif len(label_data["label_map"]) != 12:
                    verification["errors"].append(f"Expected 12 labels, found {len(label_data['label_map'])}")
                    verification["valid"] = False
                else:
                    verification["checks"]["label_map_valid"] = True
                    
            except (json.JSONDecodeError, KeyError) as e:
                verification["errors"].append(f"Invalid label_map.json: {e}")
                verification["valid"] = False
        
        # Validate metrics.json
        if self.bundle_dir / "metrics.json" in [self.bundle_dir / f for f in self.REQUIRED_FILES if (self.bundle_dir / f).exists()]:
            try:
                with open(self.bundle_dir / "metrics.json", 'r') as f:
                    metrics_data = json.load(f)
                
                # Check for bundle metadata
                if "bundle_metadata" in metrics_data:
                    verification["checks"]["metrics_metadata"] = True
                else:
                    verification["errors"].append("metrics.json missing bundle_metadata")
                    verification["valid"] = False
                    
            except json.JSONDecodeError as e:
                verification["errors"].append(f"Invalid metrics.json: {e}")
                verification["valid"] = False
        
        return verification
    
    def get_info(self) -> Dict[str, Any]:
        """Get bundle information."""
        info = {
            "bundle_hash": self.bundle_hash,
            "bundle_dir": str(self.bundle_dir),
            "exists": self.bundle_dir.exists(),
            "files": {},
            "model_formats": []
        }
        
        if not self.bundle_dir.exists():
            return info
        
        # File information
        for file in self.bundle_dir.iterdir():
            if file.is_file():
                info["files"][file.name] = {
                    "size": file.stat().st_size,
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                }
            elif file.is_dir():
                info["files"][file.name] = {
                    "type": "directory",
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                }
        
        # Model formats
        for format_name, filename in self.MODEL_FORMATS.items():
            if (self.bundle_dir / filename).exists():
                info["model_formats"].append(format_name)
        
        # Load config if available
        config_path = self.bundle_dir / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    info["config"] = yaml.safe_load(f)
            except yaml.YAMLError:
                info["config"] = "invalid"
        
        # Load metrics if available
        metrics_path = self.bundle_dir / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    info["metrics"] = metrics_data
                    if "bundle_metadata" in metrics_data:
                        info["created_at"] = metrics_data["bundle_metadata"].get("created_at")
            except json.JSONDecodeError:
                info["metrics"] = "invalid"
        
        return info
    
    @classmethod
    def list_bundles(cls, artifacts_dir: Path = Path("artifacts")) -> List[Dict[str, Any]]:
        """List all available bundles."""
        bundles = []
        
        if not artifacts_dir.exists():
            return bundles
        
        for bundle_dir in artifacts_dir.iterdir():
            if bundle_dir.is_dir() and len(bundle_dir.name) == 12:  # Assuming 12-char hashes
                bundle = cls(bundle_dir)
                bundle_info = bundle.get_info()
                bundles.append(bundle_info)
        
        return sorted(bundles, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def archive(self, format: str = "tar.gz") -> Path:
        """Create archive of bundle."""
        if format not in ["tar.gz", "zip"]:
            raise ValueError(f"Unsupported archive format: {format}")
        
        archive_dir = self.bundle_dir.parent / "archives"
        archive_dir.mkdir(exist_ok=True)
        
        if format == "tar.gz":
            import tarfile
            archive_path = archive_dir / f"{self.bundle_hash}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(self.bundle_dir, arcname=self.bundle_hash)
        else:  # zip
            import zipfile
            archive_path = archive_dir / f"{self.bundle_hash}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in self.bundle_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = str(file_path.relative_to(self.bundle_dir.parent))
                        zipf.write(file_path, arcname)
        
        return archive_path