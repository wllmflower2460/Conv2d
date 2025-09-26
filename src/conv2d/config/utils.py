"""Configuration utilities including hashing and diffing."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Set, Tuple

from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf


# Keys to exclude from config hash (volatile/system-specific)
VOLATILE_KEYS = {
    "paths.output_dir",
    "paths.log_dir", 
    "paths.checkpoint_dir",
    "paths.tensorboard_dir",
    "experiment.run_id",
    "experiment.config_hash",
    "hydra",
    "hardware.device_id",
    "hardware.max_memory_gb",
    "hardware.max_threads",
    "data.num_workers",
    "logging.console",
    "logging.file",
}


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """Flatten nested dictionary with dot-separated keys.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items: List[Tuple[str, Any]] = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, (list, tuple)):
            # Convert lists/tuples to strings for hashing
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
            
    return dict(items)


def compute_config_hash(
    config: Dict[str, Any] | DictConfig,
    exclude_keys: Set[str] | None = None,
    hash_length: int = 8,
) -> str:
    """Compute deterministic hash of configuration.
    
    Args:
        config: Configuration dictionary or DictConfig
        exclude_keys: Additional keys to exclude from hash
        hash_length: Number of hash characters to return
        
    Returns:
        Truncated SHA256 hash of configuration
    """
    # Convert OmegaConf to dict if needed
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    
    # Flatten configuration
    flat_config = flatten_dict(config)
    
    # Remove volatile keys
    all_exclude = VOLATILE_KEYS.copy()
    if exclude_keys:
        all_exclude.update(exclude_keys)
        
    # Filter out excluded keys
    filtered_config = {
        k: v for k, v in flat_config.items()
        if not any(k.startswith(ex) for ex in all_exclude)
    }
    
    # Sort keys for deterministic ordering
    sorted_config = dict(sorted(filtered_config.items()))
    
    # Convert to JSON string for hashing
    config_str = json.dumps(sorted_config, sort_keys=True, default=str)
    
    # Compute SHA256 hash
    hash_obj = hashlib.sha256(config_str.encode())
    full_hash = hash_obj.hexdigest()
    
    # Return truncated hash
    return full_hash[:hash_length]


def get_config_diff(
    config1: Dict[str, Any] | DictConfig,
    config2: Dict[str, Any] | DictConfig,
    ignore_keys: Set[str] | None = None,
) -> Dict[str, Any]:
    """Get differences between two configurations.
    
    Args:
        config1: First configuration
        config2: Second configuration
        ignore_keys: Keys to ignore in comparison
        
    Returns:
        Dictionary describing differences
    """
    # Convert to dicts if needed
    if isinstance(config1, DictConfig):
        config1 = OmegaConf.to_container(config1, resolve=True)
    if isinstance(config2, DictConfig):
        config2 = OmegaConf.to_container(config2, resolve=True)
    
    # Compute diff
    all_ignore = VOLATILE_KEYS.copy()
    if ignore_keys:
        all_ignore.update(ignore_keys)
        
    diff = DeepDiff(
        config1,
        config2,
        ignore_order=True,
        exclude_regex_paths=[f".*{key}.*" for key in all_ignore],
        verbose_level=2,
    )
    
    # Format diff for readability
    result = {}
    
    if 'values_changed' in diff:
        result['changed'] = {
            k.replace("root", ""): {
                'old': v['old_value'],
                'new': v['new_value']
            }
            for k, v in diff['values_changed'].items()
        }
        
    if 'dictionary_item_added' in diff:
        result['added'] = list(diff['dictionary_item_added'])
        
    if 'dictionary_item_removed' in diff:
        result['removed'] = list(diff['dictionary_item_removed'])
        
    return result


def validate_config_reproducibility(
    config: Dict[str, Any] | DictConfig,
) -> Tuple[bool, List[str]]:
    """Check if configuration is reproducible.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple of (is_reproducible, list_of_issues)
    """
    issues = []
    
    # Convert to dict if needed
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
        
    # Check for seed
    if 'seed' not in config or config['seed'] is None:
        issues.append("No random seed specified")
        
    # Check for deterministic settings
    if 'hardware' in config:
        hw = config['hardware']
        if hw.get('device') == 'cuda':
            if not hw.get('cudnn_deterministic', False):
                issues.append("CUDNN deterministic mode not enabled")
                
    # Check for data shuffling seed
    if 'data' in config:
        data = config['data']
        if data.get('shuffle_train', True) and 'seed' not in config:
            issues.append("Training data shuffling without seed")
            
    # Check for relative paths
    if 'paths' in config:
        for key, value in config['paths'].items():
            if value and not str(value).startswith('/'):
                issues.append(f"Relative path in paths.{key}")
                
    is_reproducible = len(issues) == 0
    return is_reproducible, issues


def save_config_with_hash(
    config: Dict[str, Any] | DictConfig,
    output_dir: str,
) -> str:
    """Save configuration with hash in filename.
    
    Args:
        config: Configuration to save
        output_dir: Directory to save to
        
    Returns:
        Path to saved configuration file
    """
    from pathlib import Path
    import yaml
    
    # Compute hash
    config_hash = compute_config_hash(config)
    
    # Add hash to config
    if isinstance(config, DictConfig):
        OmegaConf.set_struct(config, False)
        config.experiment.config_hash = config_hash
        OmegaConf.set_struct(config, True)
    else:
        if 'experiment' not in config:
            config['experiment'] = {}
        config['experiment']['config_hash'] = config_hash
    
    # Create output path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with hash in filename
    config_file = output_dir / f"config_{config_hash}.yaml"
    
    if isinstance(config, DictConfig):
        OmegaConf.save(config, config_file)
    else:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    return str(config_file)