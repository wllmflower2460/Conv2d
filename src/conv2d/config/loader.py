"""Configuration loader with Hydra and Pydantic validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from conv2d.config.models import Config
from conv2d.config.utils import (
    compute_config_hash,
    save_config_with_hash,
    validate_config_reproducibility,
)

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader with validation and hashing."""
    
    def __init__(
        self,
        config_dir: str | Path = "conf",
        config_name: str = "base",
    ) -> None:
        """Initialize config loader.
        
        Args:
            config_dir: Path to configuration directory
            config_name: Name of base config file
        """
        self.config_dir = Path(config_dir).resolve()
        self.config_name = config_name
        self._config: Optional[Config] = None
        self._raw_config: Optional[DictConfig] = None
        
    def load(
        self,
        overrides: list[str] | None = None,
        validate: bool = True,
        compute_hash: bool = True,
    ) -> Config:
        """Load configuration with overrides.
        
        Args:
            overrides: List of config overrides (Hydra format)
            validate: Whether to validate with Pydantic
            compute_hash: Whether to compute config hash
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: If config validation fails
        """
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Initialize Hydra with config directory
        with initialize_config_dir(
            config_dir=str(self.config_dir),
            version_base="1.3",
        ):
            # Compose configuration with overrides
            self._raw_config = compose(
                config_name=self.config_name,
                overrides=overrides or [],
            )
            
        # Convert to container for validation
        config_dict = OmegaConf.to_container(
            self._raw_config,
            resolve=True,
            throw_on_missing=True,
        )
        
        # Add config hash if requested
        if compute_hash:
            config_hash = compute_config_hash(config_dict)
            if 'experiment' not in config_dict:
                config_dict['experiment'] = {}
            config_dict['experiment']['config_hash'] = config_hash
            logger.info(f"Config hash: {config_hash}")
        
        # Validate with Pydantic if requested
        if validate:
            try:
                self._config = Config(**config_dict)
                logger.info("Configuration validation successful")
            except ValidationError as e:
                logger.error(f"Configuration validation failed:\n{e}")
                raise
        else:
            # Create config without validation
            self._config = Config.model_construct(**config_dict)
            
        # Check reproducibility
        is_reproducible, issues = validate_config_reproducibility(config_dict)
        if not is_reproducible:
            logger.warning("Configuration may not be reproducible:")
            for issue in issues:
                logger.warning(f"  - {issue}")
                
        return self._config
    
    def save(
        self,
        output_dir: str | Path,
        include_hash: bool = True,
    ) -> Path:
        """Save configuration to directory.
        
        Args:
            output_dir: Directory to save configuration
            include_hash: Whether to include hash in filename
            
        Returns:
            Path to saved configuration file
        """
        if self._config is None:
            raise ValueError("No configuration loaded")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for saving
        config_dict = self._config.model_dump()
        
        if include_hash:
            config_file = save_config_with_hash(config_dict, str(output_dir))
        else:
            config_file = output_dir / "config.yaml"
            OmegaConf.save(config_dict, config_file)
            
        logger.info(f"Configuration saved to: {config_file}")
        return Path(config_file)
    
    @property
    def config(self) -> Config:
        """Get loaded configuration."""
        if self._config is None:
            raise ValueError("No configuration loaded")
        return self._config
    
    @property
    def raw_config(self) -> DictConfig:
        """Get raw Hydra configuration."""
        if self._raw_config is None:
            raise ValueError("No configuration loaded")
        return self._raw_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        if self._config is None:
            raise ValueError("No configuration loaded")
        return self._config.model_dump()
    
    def print_config(self, resolve: bool = True) -> None:
        """Print configuration in readable format."""
        if self._raw_config is None:
            raise ValueError("No configuration loaded")
            
        print(OmegaConf.to_yaml(self._raw_config, resolve=resolve))


def load_config(
    config_path: str | Path | None = None,
    config_dir: str | Path = "conf",
    config_name: str = "base",
    overrides: list[str] | None = None,
) -> Config:
    """Load configuration from file or directory.
    
    Args:
        config_path: Path to specific config file (overrides dir/name)
        config_dir: Configuration directory (if config_path not provided)
        config_name: Base config name (if config_path not provided)
        overrides: List of config overrides
        
    Returns:
        Validated configuration
    """
    if config_path is not None:
        # Load from specific file
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load with OmegaConf
        raw_config = OmegaConf.load(config_path)
        
        # Apply overrides
        if overrides:
            cli_config = OmegaConf.from_dotlist(overrides)
            raw_config = OmegaConf.merge(raw_config, cli_config)
            
        # Convert and validate
        config_dict = OmegaConf.to_container(raw_config, resolve=True)
        config = Config(**config_dict)
        
    else:
        # Load with Hydra
        loader = ConfigLoader(config_dir, config_name)
        config = loader.load(overrides)
        
    return config


# Hydra decorator for main functions
def hydra_main(
    config_path: str = "../../conf",
    config_name: str = "base",
    version_base: str = "1.3",
):
    """Decorator for Hydra-based main functions.
    
    Usage:
        @hydra_main()
        def main(cfg: DictConfig) -> None:
            config = Config(**OmegaConf.to_container(cfg))
            ...
    """
    def decorator(func):
        return hydra.main(
            config_path=config_path,
            config_name=config_name,
            version_base=version_base,
        )(func)
    
    return decorator