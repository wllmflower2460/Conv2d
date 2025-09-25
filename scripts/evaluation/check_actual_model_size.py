#!/usr/bin/env python3
"""
Check actual model size and recalculate performance.
750KB is much larger than expected for 57k parameters.
"""

import torch
import os
from pathlib import Path

def check_model_sizes():
    """Check actual model file sizes."""
    
    print("="*60)
    print("ACTUAL MODEL SIZE INVESTIGATION")
    print("="*60)
    
    # Check our FSQ model
    from models.conv2d_fsq_model import Conv2dFSQ
    
    model = Conv2dFSQ(
        input_channels=9,
        hidden_dim=128,
        num_classes=10,
        fsq_levels=[8, 6, 5]
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    # Calculate sizes
    fp32_size = n_params * 4 / 1024  # KB
    fp16_size = n_params * 2 / 1024  # KB
    int8_size = n_params * 1 / 1024  # KB
    
    print(f"\nFSQ Model Parameter Count: {n_params:,}")
    print(f"\nTheoretical sizes:")
    print(f"  FP32: {fp32_size:.1f} KB")
    print(f"  FP16: {fp16_size:.1f} KB")
    print(f"  INT8: {int8_size:.1f} KB")
    
    # Check saved model files
    print("\n" + "-"*60)
    print("Checking saved model files:")
    print("-"*60)
    
    model_files = [
        "fsq_final_best.pth",
        "m15_fsq_best_qa.pth",
        "m15_best_model.pth"
    ]
    
    for file in model_files:
        if Path(file).exists():
            size_kb = os.path.getsize(file) / 1024
            print(f"{file:30s}: {size_kb:.1f} KB")
    
    # Save just the model (no optimizer state)
    print("\n" + "-"*60)
    print("Saving minimal model for deployment:")
    print("-"*60)
    
    # Save state dict only
    torch.save(model.state_dict(), 'fsq_minimal.pth')
    minimal_size = os.path.getsize('fsq_minimal.pth') / 1024
    print(f"State dict only: {minimal_size:.1f} KB")
    
    # Export to ONNX
    dummy_input = torch.randn(1, 9, 2, 50)
    torch.onnx.export(
        model,
        dummy_input,
        'fsq_model.onnx',
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
    onnx_size = os.path.getsize('fsq_model.onnx') / 1024
    print(f"ONNX export: {onnx_size:.1f} KB")
    
    print("\n" + "="*60)
    print("750KB MODEL ANALYSIS")
    print("="*60)
    
    if minimal_size > 700:  # If our model is around 750KB
        print(f"""
Your 750KB model size suggests:
1. Model includes optimizer state and metadata
2. Or it's a different/larger architecture
3. Or includes multiple components (FSQ + HSMM + HDP)

For Hailo deployment, you'd typically:
1. Quantize to INT8 → ~{int8_size:.1f} KB
2. Compile with Hailo Dataflow Compiler
3. Deploy as HEF (Hailo Executable Format)
""")
    
    # Recalculate performance for 750KB model
    print("\n" + "="*60)
    print("PERFORMANCE WITH 750KB MODEL")
    print("="*60)
    
    # If 750KB in INT8, that's roughly 750k parameters
    estimated_params = 750 * 1024  # 768,000 parameters
    estimated_ops = estimated_params * 200  # Rough estimate: 200 ops per param
    
    print(f"\nIf your model is really 750KB in INT8:")
    print(f"  Estimated parameters: {estimated_params:,}")
    print(f"  Estimated ops/inference: {estimated_ops:,} ({estimated_ops/1e9:.1f}G)")
    
    # Recalculate with 26 TOPS
    tops = 26e12
    theoretical_max = tops / estimated_ops
    
    print(f"\nHailo-8 Performance (750KB model):")
    print(f"  Theoretical max: {theoretical_max:,.0f} FPS")
    print(f"  Realistic (50%): {theoretical_max*0.5:,.0f} FPS")
    print(f"  Conservative (10%): {theoretical_max*0.1:,.0f} FPS")
    
    # Check if 31,000 is plausible
    required_efficiency = 31_000 / theoretical_max
    print(f"\n31,000 FPS would require {required_efficiency:.1%} efficiency")
    
    if required_efficiency < 0.1:
        print("✓ PLAUSIBLE with Hailo-8")
    elif required_efficiency < 0.5:
        print("🤔 POSSIBLE with optimization")
    else:
        print("❌ UNLIKELY for this model size")
    
    # Final assessment
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    print(f"""
For a 750KB INT8 model on Hailo-8:
  
  • 31,000 FPS requires only {required_efficiency:.1%} efficiency
  • This is {"PLAUSIBLE" if required_efficiency < 0.2 else "SUSPICIOUS"}
  • More likely: measuring wrong thing or batch confusion
  
Most realistic explanation:
  1. The 750KB includes extra metadata/optimizer state
  2. Actual inference model is smaller
  3. Or measurement included batching
  
To verify:
  1. Export minimal model for inference
  2. Compile with Hailo Dataflow Compiler
  3. Measure with hailo_benchmarks tool
  4. Check batch size in measurements
""")

if __name__ == "__main__":
    check_model_sizes()