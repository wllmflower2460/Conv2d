# Improved Training Configuration for Tomorrow's Run
# Target: >60% validation accuracy

IMPROVED_CONFIG = {
    # Training Schedule - More conservative to prevent collapse
    "epochs": 200,
    "batch_size": 64,  # Increased from 32 for better gradient estimates
    
    # Learning Rate Strategy - More stable
    "learning_rate": 5e-4,  # Reduced from 1e-3 to prevent explosion
    "lr_scheduler": {
        "type": "CosineAnnealingWarmRestarts", 
        "T_0": 10,
        "T_mult": 2
    },
    
    # Loss Balancing - Better stability
    "beta": 0.3,          # Reduced KL weight for stability
    "lambda_act": 3.0,    # Increased activity focus
    "lambda_dom": 0.05,   # Reduced domain confusion
    
    # Gradient Management - Prevent explosion
    "grad_clip_norm": 0.5,  # More aggressive clipping
    "weight_decay": 1e-4,   # Stronger regularization
    
    # Architecture Improvements
    "dropout_rate": 0.3,    # Add regularization
    "layer_norm": True,     # Stabilize activations
    
    # Early Stopping - More patience
    "patience": 30,         # Allow more exploration
    "min_delta": 0.001,     # Smaller improvement threshold
    
    # Data Augmentation
    "noise_std": 0.02,      # Sensor noise injection
    "time_warp": 0.1,       # Temporal augmentation
    
    # Progressive Training
    "warmup_epochs": 5,     # Gradual learning start
    "curriculum_learning": True  # Easy→hard samples
}