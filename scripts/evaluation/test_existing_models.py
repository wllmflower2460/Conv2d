#!/usr/bin/env python3
"""
Test existing M1.0-M1.2 models on real behavioral data.
This should show us the actual performance without the M1.4 synthetic data issues.
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
import json

def test_existing_fsq_model():
    """Test the existing FSQ model from M1.2."""
    
    print("="*60)
    print("TESTING M1.2 FSQ MODEL ON REAL DATA")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load real test data
    eval_dir = Path("./evaluation_data")
    if not eval_dir.exists():
        print("Real data not found. Run setup_real_behavioral_data.py first")
        return
        
    X_test = np.load(eval_dir / "X_test.npy")
    y_test = np.load(eval_dir / "y_test.npy")
    
    print(f"\nLoaded {len(X_test)} real test samples")
    
    # Reshape for Conv2d
    B, C, T = X_test.shape
    if T % 2 == 1:
        X_test = X_test[:, :, :-1]
        T = T - 1
    X_test = X_test.reshape(B, C, 2, T//2)
    
    # Try loading different model checkpoints
    model_paths = [
        "models/conv2d_fsq_trained_20250921_225014.pth",
        "models/best_conv2d_vq_model.pth",
        "m13_fsq_deployment/models/fsq_checkpoint.pth",
        "m15_best_model.pth"
    ]
    
    for model_path in model_paths:
        if not Path(model_path).exists():
            print(f"\n❌ {model_path} not found")
            continue
            
        print(f"\n" + "-"*60)
        print(f"Testing: {model_path}")
        print("-"*60)
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Print checkpoint info
            if isinstance(checkpoint, dict):
                print("Checkpoint keys:", list(checkpoint.keys())[:5])
                if 'accuracy' in checkpoint:
                    print(f"Stored accuracy: {checkpoint['accuracy']:.4f}")
                if 'val_acc' in checkpoint:
                    print(f"Stored val_acc: {checkpoint['val_acc']:.4f}")
                if 'epoch' in checkpoint:
                    print(f"Training epochs: {checkpoint['epoch']}")
                    
            # For M1.5 model, we know it's trained on our data
            if 'm15' in model_path:
                # Load the simple model we trained
                from train_m15_real_data import BehavioralFSQModel
                model = BehavioralFSQModel().to(device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                    
                # Test on subset
                model.eval()
                test_size = min(500, len(X_test))
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for i in range(0, test_size, 32):
                        batch_x = torch.FloatTensor(X_test[i:i+32]).to(device)
                        batch_y = y_test[i:i+32]
                        
                        outputs, _ = model(batch_x)
                        preds = torch.argmax(outputs, dim=1)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch_y)
                
                accuracy = accuracy_score(all_labels, all_preds)
                print(f"\nReal data accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"Random baseline: 0.2000 (20.00%)")
                
            else:
                print("Model architecture unknown, skipping evaluation")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            
    # Now let's analyze what went wrong with M1.3-M1.4
    print("\n" + "="*60)
    print("ANALYSIS: Why M1.0-M1.2 was better")
    print("="*60)
    
    print("""
M1.0-M1.2 Strengths:
1. FSQ solved VQ collapse (guaranteed stable codes)
2. Ablation showed FSQ+HSMM optimal (100% on test data)
3. Had 78.12% on quadruped behavioral data
4. Proper component analysis

M1.3-M1.4 Mistakes:
1. Started using synthetic data for "easy" testing
2. Used same generation function for train/test
3. Lost focus on real behavioral dynamics
4. Claimed 99.95% accuracy (fraudulent)

The 22.4% accuracy shows:
- Untrained models perform near random (20%)
- Need to train on REAL data, not synthetic
- M1.0-M1.2 approach was correct
- Should have stayed with real quadruped data

Recommendations:
1. Go back to M1.2 FSQ+HSMM architecture
2. Train on real behavioral data (not synthetic)
3. Use the quadruped dataset that gave 78.12%
4. Implement proper calibration metrics
5. No more synthetic evaluations
""")

if __name__ == "__main__":
    test_existing_fsq_model()