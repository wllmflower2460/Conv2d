#!/usr/bin/env python3
"""
Example usage of the enhanced movement_diagnostics.py with Quality Control.

This script demonstrates how to use the new GIGO prevention features:
1. Input validation checks for tensor shape and data quality
2. Codebook health monitoring for VQ models
3. Signal quality assessment and data consistency checks
4. Quality trends analysis and reporting
"""

import torch
import numpy as np
from pathlib import Path

# Import the enhanced diagnostics system
from preprocessing.movement_diagnostics import (
    BehavioralDataDiagnostics,
    QualityThresholds,
    QualityControl
)

def main():
    print("🛡️ Quality Control System Demonstration")
    print("=" * 50)
    
    # ================================
    # 1. BASIC QUALITY CONTROL SETUP
    # ================================
    
    print("\n1. Setting up Quality Control System")
    
    # Define custom quality thresholds
    custom_thresholds = QualityThresholds(
        max_nan_percentage=5.0,      # Stricter NaN threshold
        min_codebook_usage=0.8,      # Higher usage requirement
        min_perplexity=6.0,          # Higher diversity requirement
        min_snr_db=5.0,              # Minimum SNR requirement
        max_gap_length=25            # Smaller allowed gaps
    )
    
    # Initialize diagnostics with quality control
    diagnostics = BehavioralDataDiagnostics(
        sampling_rate=100.0,
        output_dir='./quality_control_demo',
        quality_thresholds=custom_thresholds,
        enable_quality_gates=True,
        strict_quality_mode=False  # Set to True for production
    )
    
    print("✓ Quality control system initialized with custom thresholds")
    
    # ================================
    # 2. GOOD DATA EXAMPLE
    # ================================
    
    print("\n2. Testing with Good Quality Data")
    
    # Create high-quality test data
    B, C, S, T = 8, 9, 2, 100
    good_data = torch.randn(B, C, S, T) * 0.5 + 1.0  # Good signal level
    
    # Add realistic patterns
    t = torch.linspace(0, 10, T)
    for b in range(B):
        good_data[b, 0, 0, :] += 2 * torch.sin(2 * np.pi * 0.5 * t)
        good_data[b, 1, 0, :] += 1.5 * torch.cos(2 * np.pi * 1.0 * t)
    
    # Add minimal NaN values (under threshold)
    dropout_mask = torch.rand(B, C, S, T) < 0.02  # 2% dropout
    good_data[dropout_mask] = float('nan')
    
    # Good codebook info
    good_codebook_info = {
        'perplexity': 8.5,
        'usage': 0.85,
        'entropy': 4.2,
        'dead_codes': 20,
        'total_codes': 512
    }
    
    # Run quality gates only
    quality_report = diagnostics.run_quality_gates_only(
        good_data,
        codebook_info=good_codebook_info,
        save_report=True
    )
    
    print(f"Good data result: {'PASSED' if quality_report.overall_pass else 'FAILED'}")
    
    # ================================
    # 3. BAD DATA EXAMPLE
    # ================================
    
    print("\n3. Testing with Poor Quality Data")
    
    # Create poor-quality test data
    bad_data = torch.randn(B, C, S, T) * 0.001  # Very low signal
    
    # Add excessive NaN values (over threshold)
    dropout_mask = torch.rand(B, C, S, T) < 0.15  # 15% dropout
    bad_data[dropout_mask] = float('nan')
    
    # Poor codebook info
    bad_codebook_info = {
        'perplexity': 2.1,   # Very low diversity
        'usage': 0.3,        # Low usage
        'entropy': 1.5,      # Low entropy
        'dead_codes': 300,   # Many dead codes
        'total_codes': 512
    }
    
    # Run quality gates
    bad_quality_report = diagnostics.run_quality_gates_only(
        bad_data,
        codebook_info=bad_codebook_info,
        save_report=True
    )
    
    print(f"Bad data result: {'PASSED' if bad_quality_report.overall_pass else 'FAILED'}")
    
    # ================================
    # 4. QUALITY TRENDS ANALYSIS
    # ================================
    
    print("\n4. Quality Trends Analysis")
    
    # Run a few more tests to build history
    for i in range(3):
        test_data = torch.randn(B, C, S, T)
        test_codebook = {
            'perplexity': 6.0 + np.random.normal(0, 1),
            'usage': 0.7 + np.random.normal(0, 0.1),
            'entropy': 3.5 + np.random.normal(0, 0.5),
            'dead_codes': int(50 + np.random.normal(0, 20)),
            'total_codes': 512
        }
        diagnostics.run_quality_gates_only(test_data, test_codebook, save_report=False)
    
    # Get quality trends
    status = diagnostics.get_quality_control_status()
    print("Quality System Status:")
    for key, value in status.items():
        if key != 'trends':
            print(f"  {key}: {value}")
    
    trends = status.get('trends', {})
    if trends and not trends.get('insufficient_data', False):
        print("\nQuality Trends:")
        print(f"  Pass rate: {trends.get('pass_rate', 0):.1%}")
        print(f"  Avg NaN%: {trends.get('avg_nan_percentage', 0):.2f}%")
        print(f"  Avg perplexity: {trends.get('avg_perplexity', 0):.2f}")
        print(f"  Avg usage: {trends.get('avg_usage', 0):.3f}")
    
    # ================================
    # 5. THRESHOLD UPDATES
    # ================================
    
    print("\n5. Dynamic Threshold Updates")
    
    # Update thresholds based on observed data quality
    print("Updating thresholds based on data patterns...")
    diagnostics.update_quality_thresholds(
        max_nan_percentage=8.0,    # Relax NaN threshold
        min_perplexity=4.5,        # Adjust perplexity requirement
        min_snr_db=3.0             # Lower SNR requirement
    )
    
    # ================================
    # 6. INTEGRATION WITH TRAINING
    # ================================
    
    print("\n6. Integration with Model Training")
    
    def simulate_training_step(data, model_info):
        """Simulate a training step with quality control."""
        
        # Run quality gates before processing
        quality_report = diagnostics.run_quality_gates_only(
            data, 
            codebook_info=model_info,
            save_report=False
        )
        
        if not quality_report.overall_pass:
            print("  ⚠️ Quality gates failed - skipping batch")
            return False, quality_report.recommendations
        
        # Simulate processing
        print("  ✓ Quality gates passed - processing batch")
        return True, []
    
    # Simulate training loop
    print("\nSimulating training loop with quality control:")
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        
        # Simulate varying data quality
        if epoch == 1:
            # Introduce poor quality data in epoch 2
            data = torch.randn(B, C, S, T) * 0.001
            data[:, :, :, :30] = float('nan')
            codebook = bad_codebook_info.copy()
        else:
            data = good_data.clone()
            codebook = good_codebook_info.copy()
        
        success, recommendations = simulate_training_step(data, codebook)
        
        if not success:
            print(f"    Recommendations: {recommendations[:2]}")
    
    # ================================
    # 7. EXPORT COMPREHENSIVE REPORT
    # ================================
    
    print("\n7. Exporting Comprehensive Quality Report")
    
    # Export trends report
    trends_report_path = diagnostics.export_quality_trends_report()
    print(f"📊 Comprehensive quality trends report: {trends_report_path}")
    
    # Final summary
    final_status = diagnostics.get_quality_control_status()
    print(f"\n📈 Final Statistics:")
    print(f"  Total quality reports generated: {final_status['total_reports']}")
    print(f"  Quality control enabled: {final_status['enabled']}")
    print(f"  Codebook monitoring: {final_status['codebook_monitoring']}")
    
    print("\n🎉 Quality Control Demo Complete!")
    print("\n💡 Key Benefits:")
    print("  • Prevents GIGO (Garbage In, Garbage Out)")
    print("  • Provides actionable recommendations")
    print("  • Monitors model health over time") 
    print("  • Integrates seamlessly with existing diagnostics")
    print("  • Supports both strict and flexible quality modes")
    print("  • Generates comprehensive JSON reports for analysis")


if __name__ == "__main__":
    main()