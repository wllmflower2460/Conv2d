"""Artifact bundler for creating deployment packages.

Creates standardized deployment bundles containing:
- Model artifacts (ONNX, CoreML, Hailo)
- Configuration and metadata
- Label mappings and metrics
- Version information
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
import numpy as np


class DeploymentBundle:
    """Container for deployment bundle information."""
    
    def __init__(self, bundle_dir: Path, exp_hash: str):
        """Initialize bundle.
        
        Args:
            bundle_dir: Bundle directory path
            exp_hash: Experiment hash
        """
        self.bundle_dir = Path(bundle_dir)
        self.exp_hash = exp_hash
        
        # Standard file paths
        self.model_onnx = self.bundle_dir / "model.onnx"
        self.model_coreml = self.bundle_dir / "coreml.mlpackage"
        self.model_hailo = self.bundle_dir / "hailo.hef"
        self.label_map = self.bundle_dir / "label_map.json"
        self.config_file = self.bundle_dir / "config.yaml"
        self.metrics_file = self.bundle_dir / "metrics.json"
        self.version_file = self.bundle_dir / "VERSION"
        self.commit_file = self.bundle_dir / "COMMIT_SHA"
        
    def exists(self) -> bool:
        """Check if bundle directory exists."""
        return self.bundle_dir.exists()
    
    def get_manifest(self) -> Dict[str, Any]:
        """Get bundle manifest with file information.
        
        Returns:
            Dictionary with bundle contents and metadata
        """
        manifest = {
            "bundle_hash": self.exp_hash,
            "created": datetime.now().isoformat(),
            "files": {},
        }
        
        # Check each standard file
        for name, path in [
            ("model_onnx", self.model_onnx),
            ("model_coreml", self.model_coreml),
            ("model_hailo", self.model_hailo),
            ("label_map", self.label_map),
            ("config", self.config_file),
            ("metrics", self.metrics_file),
            ("version", self.version_file),
            ("commit_sha", self.commit_file),
        ]:
            if path.exists():
                if path.is_file():
                    size = path.stat().st_size
                    manifest["files"][name] = {
                        "path": str(path.relative_to(self.bundle_dir)),
                        "size_bytes": size,
                        "exists": True,
                    }
                elif path.is_dir():  # CoreML package is a directory
                    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    manifest["files"][name] = {
                        "path": str(path.relative_to(self.bundle_dir)),
                        "size_bytes": size,
                        "exists": True,
                        "type": "directory",
                    }
            else:
                manifest["files"][name] = {"exists": False}
        
        return manifest
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        manifest = self.get_manifest()
        
        lines = [
            f"Deployment Bundle: {self.exp_hash}",
            f"Location: {self.bundle_dir}",
            f"Created: {manifest['created']}",
            "",
            "Contents:",
        ]
        
        for name, info in manifest["files"].items():
            if info["exists"]:
                size_mb = info["size_bytes"] / (1024 * 1024)
                status = f"✓ {info['path']} ({size_mb:.1f} MB)"
            else:
                status = f"✗ {name} (missing)"
            lines.append(f"  {status}")
        
        return "\n".join(lines)


class ArtifactBundler:
    """Creates deployment bundles with all necessary artifacts."""
    
    def __init__(self, output_base: Union[str, Path] = "artifacts"):
        """Initialize bundler.
        
        Args:
            output_base: Base directory for artifacts
        """
        self.output_base = Path(output_base)
        self.output_base.mkdir(exist_ok=True)
    
    def create_bundle(
        self,
        config: Dict[str, Any],
        model_path: Optional[Path] = None,
        metrics: Optional[Dict[str, Any]] = None,
        label_mapping: Optional[Dict[int, str]] = None,
        exp_name: str = "experiment",
    ) -> DeploymentBundle:
        """Create a complete deployment bundle.
        
        Args:
            config: Experiment configuration
            model_path: Path to trained model
            metrics: Evaluation metrics
            label_mapping: Motif ID to behavior name mapping
            exp_name: Experiment name
            
        Returns:
            DeploymentBundle with all artifacts
        """
        # Generate experiment hash
        exp_hash = self._generate_exp_hash(config, exp_name)
        
        # Create bundle directory
        bundle_dir = self.output_base / exp_hash
        bundle_dir.mkdir(exist_ok=True)
        
        bundle = DeploymentBundle(bundle_dir, exp_hash)
        
        # Save configuration
        self._save_config(bundle.config_file, config)
        
        # Save metrics if provided
        if metrics:
            self._save_metrics(bundle.metrics_file, metrics)
        
        # Save label mapping
        if label_mapping:
            self._save_label_mapping(bundle.label_map, label_mapping)
        else:
            # Create default mapping
            n_classes = config.get("model", {}).get("n_classes", 4)
            default_mapping = {i: f"behavior_{i}" for i in range(n_classes)}
            self._save_label_mapping(bundle.label_map, default_mapping)
        
        # Export models if source model provided
        if model_path and model_path.exists():
            self._export_models(model_path, bundle, config)
        
        # Save version information
        self._save_version_info(bundle)
        
        return bundle
    
    def _generate_exp_hash(self, config: Dict[str, Any], exp_name: str) -> str:
        """Generate deterministic experiment hash."""
        # Create reproducible hash from config + name
        config_str = json.dumps(config, sort_keys=True)
        hash_input = f"{exp_name}_{config_str}"
        
        hash_obj = hashlib.sha256(hash_input.encode())
        return hash_obj.hexdigest()[:16]
    
    def _save_config(self, config_file: Path, config: Dict[str, Any]) -> None:
        """Save configuration to YAML file."""
        # Add metadata
        config_with_meta = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "framework": "Conv2d-VQ-HDP-HSMM",
                "version": "1.0",
            },
            **config
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_with_meta, f, default_flow_style=False)
    
    def _save_metrics(self, metrics_file: Path, metrics: Dict[str, Any]) -> None:
        """Save metrics to JSON file."""
        # Ensure all metrics are JSON serializable
        serializable_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        # Add metadata
        metrics_with_meta = {
            "metadata": {
                "computed": datetime.now().isoformat(),
                "framework": "Conv2d",
            },
            "metrics": serializable_metrics,
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)
    
    def _save_label_mapping(self, label_file: Path, mapping: Dict[int, str]) -> None:
        """Save label mapping to JSON file."""
        # Convert keys to strings for JSON
        str_mapping = {str(k): v for k, v in mapping.items()}
        
        mapping_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "description": "Motif ID to behavior name mapping",
            },
            "mapping": str_mapping,
            "reverse_mapping": {v: int(k) for k, v in mapping.items()},
        }
        
        with open(label_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
    
    def _export_models(
        self, 
        model_path: Path, 
        bundle: DeploymentBundle,
        config: Dict[str, Any],
    ) -> None:
        """Export model to different formats."""
        try:
            from .exporter import ModelExporter
            
            exporter = ModelExporter()
            
            # Export to ONNX
            if not bundle.model_onnx.exists():
                print(f"Exporting ONNX model...")
                exporter.export_onnx(model_path, bundle.model_onnx, config)
            
            # Export to CoreML (if ONNX exists)
            if bundle.model_onnx.exists() and not bundle.model_coreml.exists():
                print(f"Exporting CoreML model...")
                exporter.export_coreml(bundle.model_onnx, bundle.model_coreml, config)
            
            # Export to Hailo (if ONNX exists)
            if bundle.model_onnx.exists() and not bundle.model_hailo.exists():
                print(f"Exporting Hailo model...")
                exporter.export_hailo(bundle.model_onnx, bundle.model_hailo, config)
                
        except Exception as e:
            print(f"Warning: Model export failed: {e}")
            # Continue without model files
    
    def _save_version_info(self, bundle: DeploymentBundle) -> None:
        """Save version and commit information."""
        # Version info
        version_info = {
            "framework": "Conv2d",
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "python": self._get_python_version(),
            "dependencies": self._get_dependencies(),
        }
        
        with open(bundle.version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Git commit SHA
        commit_sha = self._get_commit_sha()
        with open(bundle.commit_file, 'w') as f:
            f.write(commit_sha)
    
    def _get_python_version(self) -> str:
        """Get Python version string."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get key dependency versions."""
        deps = {}
        
        try:
            import torch
            deps["torch"] = torch.__version__
        except ImportError:
            pass
            
        try:
            import numpy as np
            deps["numpy"] = np.__version__
        except ImportError:
            pass
            
        try:
            import sklearn
            deps["scikit-learn"] = sklearn.__version__
        except ImportError:
            pass
        
        return deps
    
    def _get_commit_sha(self) -> str:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        return "unknown"
    
    def list_bundles(self) -> List[DeploymentBundle]:
        """List all existing bundles.
        
        Returns:
            List of DeploymentBundle objects
        """
        bundles = []
        
        if self.output_base.exists():
            for bundle_dir in self.output_base.iterdir():
                if bundle_dir.is_dir():
                    exp_hash = bundle_dir.name
                    bundle = DeploymentBundle(bundle_dir, exp_hash)
                    bundles.append(bundle)
        
        return sorted(bundles, key=lambda b: b.bundle_dir.stat().st_mtime, reverse=True)
    
    def clean_old_bundles(self, keep_recent: int = 10) -> int:
        """Clean old bundles, keeping only recent ones.
        
        Args:
            keep_recent: Number of recent bundles to keep
            
        Returns:
            Number of bundles removed
        """
        bundles = self.list_bundles()
        
        if len(bundles) <= keep_recent:
            return 0
        
        removed = 0
        for bundle in bundles[keep_recent:]:
            try:
                shutil.rmtree(bundle.bundle_dir)
                removed += 1
                print(f"Removed old bundle: {bundle.exp_hash}")
            except Exception as e:
                print(f"Failed to remove bundle {bundle.exp_hash}: {e}")
        
        return removed
    
    def copy_bundle(
        self, 
        source_bundle: DeploymentBundle, 
        dest_path: Path,
    ) -> None:
        """Copy bundle to another location.
        
        Args:
            source_bundle: Bundle to copy
            dest_path: Destination path
        """
        dest_path = Path(dest_path)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Copy entire bundle directory
        dest_bundle = dest_path / source_bundle.exp_hash
        
        if dest_bundle.exists():
            shutil.rmtree(dest_bundle)
        
        shutil.copytree(source_bundle.bundle_dir, dest_bundle)
        print(f"Copied bundle to: {dest_bundle}")
    
    def archive_bundle(self, bundle: DeploymentBundle, format: str = "zip") -> Path:
        """Create archive of bundle.
        
        Args:
            bundle: Bundle to archive
            format: Archive format (zip, tar, tar.gz)
            
        Returns:
            Path to created archive
        """
        archive_name = f"{bundle.exp_hash}.{format}"
        archive_path = bundle.bundle_dir.parent / archive_name
        
        if format == "zip":
            shutil.make_archive(
                str(archive_path.with_suffix("")),
                "zip",
                bundle.bundle_dir.parent,
                bundle.bundle_dir.name,
            )
        elif format in ["tar", "tar.gz"]:
            compression = "gz" if format == "tar.gz" else None
            shutil.make_archive(
                str(archive_path.with_suffix("").with_suffix("")),
                "tar",
                bundle.bundle_dir.parent,
                bundle.bundle_dir.name,
                compression=compression,
            )
        else:
            raise ValueError(f"Unsupported archive format: {format}")
        
        return archive_path