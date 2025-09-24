#!/usr/bin/env python3
"""Quick test of current best model."""

import torch
import numpy as np
from pathlib import Path

# Load checkpoint
checkpoint = torch.load('fsq_final_best.pth', map_location='cpu', weights_only=False)

print("="*60)
print("CURRENT BEST MODEL STATUS")
print("="*60)
print(f"\nEpoch: {checkpoint['epoch']}")
print(f"Validation Accuracy: {checkpoint['val_acc']:.4f} ({checkpoint['val_acc']*100:.2f}%)")
print(f"Training Accuracy: {checkpoint['train_acc']:.4f}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")
print(f"Dataset: {checkpoint.get('dataset', 'Unknown')}")
print(f"FSQ Levels: {checkpoint.get('fsq_levels', 'Unknown')}")
print(f"Parameters: {checkpoint.get('n_params', 'Unknown'):,}")

print("\n" + "-"*60)
print("COMPARISON TO TARGETS")
print("-"*60)
print(f"M1.0-M1.2 Target: 78.12%")
print(f"Current Achievement: {checkpoint['val_acc']*100:.2f}%")
print(f"Improvement over target: {(checkpoint['val_acc'] - 0.7812)*100:.2f}%")
print(f"Random baseline (10 classes): 10.00%")
print(f"Improvement over random: {(checkpoint['val_acc'] - 0.1)*100:.2f}%")

if checkpoint['val_acc'] > 0.85:
    print("\n🎉 EXCELLENT: Exceeding all targets!")
elif checkpoint['val_acc'] > 0.78:
    print("\n✅ SUCCESS: Target achieved!")
else:
    print("\n⚠️ Still training...")

# Check for ablation results
import json
import glob

ablation_files = glob.glob("ablation_real_data_*.json")
if ablation_files:
    latest_ablation = max(ablation_files)
    with open(latest_ablation, 'r') as f:
        ablation = json.load(f)
    
    print("\n" + "-"*60)
    print("ABLATION STUDY RESULTS")
    print("-"*60)
    print(f"Dataset: {ablation.get('dataset', 'Unknown')}")
    print(f"Best Config: {ablation.get('best_config', 'Unknown')}")
    print(f"Best Accuracy: {ablation.get('best_accuracy', 0):.4f}")
    
    if 'results' in ablation:
        print("\nAll Configurations:")
        for name, res in ablation['results'].items():
            print(f"  {name:20s}: {res['val_accuracy']:.4f}")

print("\n" + "="*60)