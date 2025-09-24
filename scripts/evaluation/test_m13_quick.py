#!/usr/bin/env python3
"""
Quick test of M1.3 training pipeline with reduced epochs.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_fsq_real_data_m13 import RealDataTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Override epochs for quick test
class QuickTrainer(RealDataTrainer):
    def train_model(self, X_train, y_train, X_val, y_val):
        """Quick training with only 5 epochs."""
        from models.conv2d_fsq_calibrated import CalibratedConv2dFSQ
        from training.train_fsq_calibrated import FSQCalibrationTrainer
        from torch.utils.data import DataLoader, TensorDataset
        
        logger.info("Running quick test with 5 epochs...")
        
        # Create model
        model = CalibratedConv2dFSQ(
            input_channels=9,
            hidden_dim=128,
            num_classes=10,
            fsq_levels=[8]*8,
            alpha=0.1
        )
        
        # Create trainer
        trainer = FSQCalibrationTrainer(model, self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train[:1000], y_train[:1000])  # Subset
        val_dataset = TensorDataset(X_val[:500], y_val[:500])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train with only 5 epochs
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=5,  # Quick test
            learning_rate=1e-3,
            early_stopping_patience=10
        )
        
        self.best_model = trainer.model
        self.best_metrics = trainer.model.calibration_metrics
        
        return history

def main():
    """Quick test run."""
    trainer = QuickTrainer()
    
    # Load small dataset
    X, y, data_source = trainer.load_real_data()
    logger.info(f"Data: {X.shape} from {data_source}")
    
    # Small augmentation
    X_aug, y_aug = trainer.augment_data(X[:1000], y[:1000])
    
    # Quick split
    n = len(X_aug)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    X_train = X_aug[:n_train]
    y_train = y_aug[:n_train]
    X_val = X_aug[n_train:n_train+n_val]
    y_val = y_aug[n_train:n_train+n_val]
    X_test = X_aug[n_train+n_val:]
    y_test = y_aug[n_train+n_val:]
    
    # Quick train
    history = trainer.train_model(X_train, y_train, X_val, y_val)
    
    # Quick validation
    results, passed = trainer.validate_m13_requirements(X_test, y_test)
    
    logger.info(f"\nQuick Test Results:")
    logger.info(f"Accuracy: {results['accuracy']:.3f}")
    logger.info(f"ECE: {results.get('ece', 'N/A')}")
    logger.info(f"Latency: {results['latency_p95_ms']:.1f} ms")
    
    # Export if reasonable accuracy
    if results['accuracy'] > 0.70:
        export_dir = trainer.export_for_hailo()
        logger.info(f"Exported to: {export_dir}")
        
        # Run verification
        from verify_hailo_deployment import HailoDeploymentVerifier
        verifier = HailoDeploymentVerifier(str(export_dir))
        ready = verifier.run_full_verification()
        
        if ready:
            logger.info("\n✅ Quick test successful - ready for full training")
        else:
            logger.info("\n⚠️ Issues found - check deployment report")
    else:
        logger.info("\n⚠️ Low accuracy in quick test - check data/model")
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"\n✅ Quick test complete: {results['accuracy']:.1%} accuracy")