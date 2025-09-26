"""Pydantic models for configuration validation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class Device(str, Enum):
    """Supported devices."""
    
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    HAILO = "hailo"
    
    
class NaNStrategy(str, Enum):
    """NaN handling strategies."""
    
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"
    INTERPOLATE = "interpolate"
    DROP = "drop"
    RAISE = "raise"


class OptimizerType(str, Enum):
    """Optimizer types."""
    
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    

class SchedulerType(str, Enum):
    """Learning rate scheduler types."""
    
    COSINE = "cosine"
    STEP = "step"
    PLATEAU = "plateau"
    ONECYCLE = "onecycle"
    LINEAR = "linear"


class PathsConfig(BaseModel):
    """Paths configuration."""
    
    root_dir: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))
    output_dir: Path = Field(default=Path("outputs"))
    log_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    tensorboard_dir: Optional[Path] = None
    
    @model_validator(mode='after')
    def resolve_paths(self) -> 'PathsConfig':
        """Resolve relative paths to absolute."""
        # Make paths absolute if relative
        if not self.data_dir.is_absolute():
            self.data_dir = self.root_dir / self.data_dir
        if not self.models_dir.is_absolute():
            self.models_dir = self.root_dir / self.models_dir
        if not self.output_dir.is_absolute():
            self.output_dir = self.root_dir / self.output_dir
            
        # Set derived paths if not specified
        if self.log_dir is None:
            self.log_dir = self.output_dir / "logs"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / "checkpoints"
        if self.tensorboard_dir is None:
            self.tensorboard_dir = self.output_dir / "tensorboard"
            
        return self


class DataQualityConfig(BaseModel):
    """Data quality configuration."""
    
    nan_strategy: NaNStrategy = NaNStrategy.INTERPOLATE
    nan_threshold_warn: float = Field(5.0, ge=0, le=100)
    nan_threshold_error: float = Field(20.0, ge=0, le=100)
    edge_method: Literal["extrapolate", "constant", "ffill"] = "constant"
    nan_fallback: Literal["zero", "mean", "median"] = "median"
    
    @field_validator('nan_threshold_error')
    @classmethod
    def validate_thresholds(cls, v: float, info) -> float:
        """Ensure error threshold > warn threshold."""
        if 'nan_threshold_warn' in info.data:
            if v <= info.data['nan_threshold_warn']:
                raise ValueError('nan_threshold_error must be > nan_threshold_warn')
        return v


class DataConfig(BaseModel):
    """Data configuration."""
    
    dataset_name: str
    dataset_path: Optional[Path] = None
    batch_size: int = Field(32, gt=0)
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = Field(2, gt=0)
    
    # Preprocessing
    window_size: int = Field(100, gt=0)
    step_size: int = Field(50, gt=0)
    sampling_rate: float = Field(100.0, gt=0)
    normalize: bool = True
    
    # Quality
    quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    
    # Augmentation
    augmentation_enabled: bool = False
    noise_std: float = Field(0.01, ge=0)
    time_warp: float = Field(0.1, ge=0)
    magnitude_warp: float = Field(0.1, ge=0)
    
    # Splits
    train_split: float = Field(0.7, gt=0, le=1)
    val_split: float = Field(0.15, gt=0, le=1)
    test_split: float = Field(0.15, gt=0, le=1)
    
    @field_validator('step_size')
    @classmethod
    def validate_step_size(cls, v: int, info) -> int:
        """Ensure step_size <= window_size."""
        if 'window_size' in info.data:
            if v > info.data['window_size']:
                raise ValueError('step_size must be <= window_size')
        return v
    
    @model_validator(mode='after')
    def validate_splits(self) -> 'DataConfig':
        """Ensure splits sum to 1.0."""
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'Splits must sum to 1.0, got {total}')
        return self


class FSQConfig(BaseModel):
    """FSQ quantization configuration."""
    
    levels: List[int] = Field(default=[8, 6, 5])
    embedding_dim: int = Field(64, gt=0)
    commitment_weight: float = Field(0.25, ge=0)
    ema_decay: float = Field(0.99, gt=0, lt=1)
    epsilon: float = Field(1e-5, gt=0)
    
    @property
    def codebook_size(self) -> int:
        """Calculate total codebook size."""
        size = 1
        for level in self.levels:
            size *= level
        return size
    
    @field_validator('levels')
    @classmethod
    def validate_levels(cls, v: List[int]) -> List[int]:
        """Ensure all levels are positive."""
        if not all(level > 1 for level in v):
            raise ValueError('All FSQ levels must be > 1')
        return v


class HSMMConfig(BaseModel):
    """HSMM configuration."""
    
    num_states: int = Field(32, gt=0)
    observation_dim: int = Field(64, gt=0)
    duration_type: Literal["negative_binomial", "poisson", "gaussian"] = "negative_binomial"
    min_duration: int = Field(1, gt=0)
    max_duration: int = Field(50, gt=0)
    
    @field_validator('max_duration')
    @classmethod
    def validate_durations(cls, v: int, info) -> int:
        """Ensure max_duration > min_duration."""
        if 'min_duration' in info.data:
            if v <= info.data['min_duration']:
                raise ValueError('max_duration must be > min_duration')
        return v


class ModelConfig(BaseModel):
    """Model configuration."""
    
    name: str = "conv2d_fsq_hsmm"
    architecture: Literal["conv2d", "tcn", "transformer", "integrated"] = "integrated"
    
    # Components
    input_channels: int = Field(9, gt=0)
    hidden_channels: List[int] = Field(default=[32, 64, 128])
    
    # FSQ config
    fsq: FSQConfig = Field(default_factory=FSQConfig)
    
    # HSMM config
    hsmm: HSMMConfig = Field(default_factory=HSMMConfig)
    
    # Additional settings
    dropout: float = Field(0.1, ge=0, le=1)
    activation: Literal["relu", "gelu", "silu", "tanh"] = "relu"
    batch_norm: bool = True


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""
    
    type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    betas: List[float] = Field(default=[0.9, 0.999])
    eps: float = Field(1e-8, gt=0)
    amsgrad: bool = False
    
    @field_validator('betas')
    @classmethod
    def validate_betas(cls, v: List[float]) -> List[float]:
        """Validate beta parameters for Adam."""
        if len(v) != 2:
            raise ValueError('betas must have exactly 2 values')
        if not (0 <= v[0] < 1 and 0 <= v[1] < 1):
            raise ValueError('betas must be in [0, 1)')
        return v


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""
    
    type: SchedulerType = SchedulerType.COSINE
    warmup_epochs: int = Field(5, ge=0)
    min_lr_ratio: float = Field(0.01, gt=0, le=1)
    step_size: Optional[int] = Field(None, gt=0)
    gamma: float = Field(0.1, gt=0, le=1)


class TrainingConfig(BaseModel):
    """Training configuration."""
    
    epochs: int = Field(100, gt=0)
    gradient_accumulation_steps: int = Field(1, gt=0)
    gradient_clip: Optional[float] = Field(1.0, gt=0)
    
    # Optimizer
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    
    # Scheduler
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    
    # Evaluation
    eval_interval: int = Field(500, gt=0)
    save_interval: int = Field(1000, gt=0)
    log_interval: int = Field(100, gt=0)
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = Field(20, gt=0)
    early_stopping_min_delta: float = Field(1e-4, gt=0)
    early_stopping_mode: Literal["min", "max"] = "min"
    
    # Checkpointing
    save_best: bool = True
    save_last: bool = True
    keep_last_k: int = Field(3, gt=0)
    
    # Mixed precision
    mixed_precision_enabled: bool = False
    mixed_precision_opt_level: Literal["O0", "O1", "O2", "O3"] = "O1"


class HardwareConfig(BaseModel):
    """Hardware configuration."""
    
    device: Device = Device.CPU
    device_id: int = Field(0, ge=0)
    mixed_precision: bool = False
    compile_model: bool = False
    
    # Resource limits
    max_memory_gb: Optional[float] = Field(None, gt=0)
    max_threads: Optional[int] = Field(None, gt=0)
    
    # CUDA settings
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v: Device) -> Device:
        """Validate device availability."""
        import torch
        
        if v == Device.CUDA and not torch.cuda.is_available():
            raise ValueError('CUDA device requested but not available')
        if v == Device.MPS and not torch.backends.mps.is_available():
            raise ValueError('MPS device requested but not available')
        return v


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""
    
    name: str
    run_id: str
    config_hash: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    console_logging: bool = True
    file_logging: bool = True
    tensorboard: bool = True
    
    # Weights & Biases
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = Field(default_factory=list)


class Config(BaseModel):
    """Complete configuration model."""
    
    # Project info
    project_name: str = "Conv2d-FSQ-HSMM"
    project_version: str = "0.2.0"
    seed: int = Field(42, ge=0)
    
    # Sub-configurations
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    experiment: ExperimentConfig
    
    # Hydra config (stored but not validated)
    hydra: Optional[Dict[str, Any]] = None
    
    @model_validator(mode='after')
    def post_validation(self) -> 'Config':
        """Post-validation adjustments."""
        # Ensure model observation dim matches FSQ embedding dim
        self.model.hsmm.observation_dim = self.model.fsq.embedding_dim
        
        # Adjust batch size for hardware
        if self.hardware.device == Device.HAILO:
            self.data.batch_size = min(self.data.batch_size, 1)
        elif self.hardware.device == Device.CPU:
            self.data.batch_size = min(self.data.batch_size, 16)
            
        # Enable mixed precision based on hardware
        if self.hardware.mixed_precision:
            self.training.mixed_precision_enabled = True
            
        return self
    
    class Config:
        """Pydantic config."""
        
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"  # Fail on unknown fields