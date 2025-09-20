"""
Training Configuration - Centralized Baseline and Target Management
Addresses Copilot feedback about hardcoded baseline values
"""

class TrainingConfig:
    """Centralized configuration for training baselines and targets"""
    
    # Model Performance Baselines
    BASELINE_ACCURACY = 0.5768  # Previous best baseline
    CURRENT_BEST = 0.8653      # Current best (86.53%)
    TARGET_ACCURACY = 0.90     # 90% target goal
    
    # Training Targets by Model Type
    TARGETS = {
        'baseline': 0.60,      # 60% initial target
        'enhanced': 0.8653,    # Beat current best
        'production': 0.90,    # Production target
        'quadruped': 0.7812    # 78.12% quadruped model
    }
    
    # Training Parameters
    DEFAULT_PATIENCE = 50
    ENHANCED_PATIENCE = 75
    OVERNIGHT_PATIENCE = 100
    
    # Performance Thresholds
    MIN_IMPROVEMENT_PCT = 1.0  # Minimum 1% improvement to be considered significant
    
    @classmethod
    def get_baseline(cls, model_type='baseline'):
        """Get baseline accuracy for a specific model type"""
        if model_type == 'enhanced':
            return cls.CURRENT_BEST
        elif model_type == 'quadruped':
            return cls.TARGETS['quadruped']
        else:
            return cls.BASELINE_ACCURACY
    
    @classmethod
    def get_target(cls, model_type='baseline'):
        """Get target accuracy for a specific model type"""
        return cls.TARGETS.get(model_type, cls.TARGET_ACCURACY)
    
    @classmethod
    def calculate_improvement(cls, current_accuracy, baseline=None):
        """Calculate improvement percentage over baseline"""
        if baseline is None:
            baseline = cls.BASELINE_ACCURACY
        return ((current_accuracy / baseline) - 1) * 100
    
    @classmethod
    def calculate_target_progress(cls, current_accuracy, target=None):
        """Calculate progress toward target as percentage"""
        if target is None:
            target = cls.TARGET_ACCURACY
        return (current_accuracy / target) * 100
    
    @classmethod
    def is_significant_improvement(cls, current_accuracy, baseline=None):
        """Check if improvement is significant"""
        improvement = cls.calculate_improvement(current_accuracy, baseline)
        return improvement >= cls.MIN_IMPROVEMENT_PCT


# Legacy constants for backward compatibility (deprecated)
BASELINE_ACCURACY = TrainingConfig.BASELINE_ACCURACY
CURRENT_BEST_ACCURACY = TrainingConfig.CURRENT_BEST
TARGET_ACCURACY = TrainingConfig.TARGET_ACCURACY