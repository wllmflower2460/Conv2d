"""Deployment packaging command implementation."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import tarfile
import hashlib
import json
import os


def create_package(
    model_dir: Path,
    eval_dir: Optional[Path],
    format: str,
    compress: bool
) -> Dict[str, Any]:
    """Create deployment package."""
    contents = []
    
    # Add model files
    model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.onnx"))
    for file in model_files:
        size = os.path.getsize(file)
        file_hash = calculate_file_hash(file)
        contents.append({
            'name': file.name,
            'path': str(file),
            'size': format_size(size),
            'hash': file_hash
        })
    
    # Add evaluation results if provided
    if eval_dir and eval_dir.exists():
        eval_files = list(eval_dir.glob("*"))
        for file in eval_files:
            if file.is_file():
                size = os.path.getsize(file)
                file_hash = calculate_file_hash(file)
                contents.append({
                    'name': f"eval/{file.name}",
                    'path': str(file),
                    'size': format_size(size),
                    'hash': file_hash
                })
    
    # Calculate total size
    total_bytes = sum(os.path.getsize(c['path']) for c in contents if os.path.exists(c['path']))
    
    # Generate package hash
    package_hash = hashlib.md5(
        "".join([c['hash'] for c in contents]).encode()
    ).hexdigest()
    
    return {
        'contents': contents,
        'total_size': format_size(total_bytes),
        'package_hash': package_hash,
        'format': format,
        'compressed': compress
    }


def save_package(output_file: Path, package_info: Dict[str, Any]):
    """Save deployment package to tar.gz."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create tar archive
    mode = 'w:gz' if package_info['compressed'] else 'w'
    
    with tarfile.open(output_file, mode) as tar:
        # Add all files
        for item in package_info['contents']:
            if os.path.exists(item['path']):
                arcname = item['name']
                tar.add(item['path'], arcname=arcname)
        
        # Add manifest
        manifest = {
            'package_hash': package_info['package_hash'],
            'format': package_info['format'],
            'contents': [
                {'name': c['name'], 'hash': c['hash'], 'size': c['size']}
                for c in package_info['contents']
            ]
        }
        
        manifest_path = output_file.parent / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        tar.add(manifest_path, arcname="manifest.json")
        manifest_path.unlink()  # Clean up temp file


def calculate_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def format_size(n_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f}TB"