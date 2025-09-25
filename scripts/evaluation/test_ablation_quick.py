#!/usr/bin/env python3
"""
Quick test of ablation framework with minimal epochs.
Run this before the overnight study to ensure everything works.
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import the upgraded framework
from experiments.Ablation_Framework_Upgraded import AblationRunner, set_all_seeds

def main():
    print("=" * 60)
    print("QUICK ABLATION TEST (2 epochs, 3 configs)")
    print("=" * 60)
    
    # Set seeds
    set_all_seeds(42)
    
    # Create small synthetic dataset
    n_train = 256
    n_val = 64
    X_train = torch.randn(n_train, 9, 2, 100)
    y_train = torch.randint(0, 12, (n_train,))
    X_val = torch.randn(n_val, 9, 2, 100)
    y_val = torch.randint(0, 12, (n_val,))
    
    # Create loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create runner
    runner = AblationRunner(
        train_loader,
        val_loader,
        device=device,
        output_dir="ablation_test_quick"
    )
    
    # Override configs to test just 3 configurations
    from experiments.Ablation_Framework_Upgraded import AblationConfig
    
    def get_test_configs():
        return [
            AblationConfig("baseline", False, False, False, False, False, 
                         description="Baseline test"),
            AblationConfig("vq_only", True, False, False, False, False, 
                         description="VQ only test"),
            AblationConfig("full", True, True, True, True, True, 
                         description="Full model test"),
        ]
    
    # Monkey-patch the config getter for testing
    runner.get_ablation_configs = get_test_configs
    
    # Run quick ablation (2 epochs)
    print("\nRunning quick test with 2 epochs...")
    results = runner.run_ablation(epochs=2, seed=42)
    
    # Generate report
    runner.generate_report(results)
    
    print("\n" + "=" * 60)
    print("✅ QUICK TEST COMPLETE")
    print("If this worked, the overnight run should work too!")
    print("Results in: ablation_test_quick/")
    print("=" * 60)


if __name__ == "__main__":
    main()