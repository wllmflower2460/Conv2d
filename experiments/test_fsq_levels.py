#!/usr/bin/env python3
"""Test script demonstrating FSQ level configuration and analysis."""

import numpy as np
from typing import List, Dict, Any

def analyze_fsq_config(levels: List[int]) -> Dict[str, Any]:
    """Analyze properties of FSQ configuration."""
    total_codes = np.prod(levels)
    bits_per_dim = [np.log2(l) for l in levels]
    total_bits = sum(bits_per_dim)
    
    # Estimate memory and speed impacts
    memory_mb = (total_codes * 4) / (1024 * 1024)  # Assuming float32
    speed_factor = 512 / total_codes  # Relative to 512 baseline
    
    return {
        'levels': levels,
        'total_codes': int(total_codes),
        'bits_per_dim': [round(b, 2) for b in bits_per_dim],
        'total_bits': round(total_bits, 2),
        'memory_mb': round(memory_mb, 2),
        'speed_factor': round(speed_factor, 2),
        'uniformity': round(np.std(levels) / np.mean(levels), 2)  # Lower = more uniform
    }

def compare_configurations():
    """Compare different FSQ configurations."""
    
    configs = {
        'Minimal (Debug)': [4, 4, 4, 4],
        'Speed-Optimized': [8, 4, 4, 4],
        'Balanced (Default)': [8, 5, 5, 4],
        'Accuracy-Focused': [16, 8, 8, 6],
        'Uniform Small': [6, 6, 6, 6],
        'Uniform Large': [10, 10, 10],
        'Imbalanced (Bad)': [64, 2, 2, 2],
    }
    
    print("="*80)
    print("FSQ LEVEL CONFIGURATION COMPARISON")
    print("="*80)
    print("\nAnalyzing different FSQ configurations for trade-offs...")
    print("\nConfiguration Details:")
    print("-"*80)
    
    results = []
    for name, levels in configs.items():
        analysis = analyze_fsq_config(levels)
        results.append((name, analysis))
        
        print(f"\n{name}: {levels}")
        print(f"  Total codes: {analysis['total_codes']:,}")
        print(f"  Bits per dim: {analysis['bits_per_dim']} (total: {analysis['total_bits']} bits)")
        print(f"  Memory: {analysis['memory_mb']} MB")
        print(f"  Speed factor: {analysis['speed_factor']}x")
        print(f"  Uniformity: {analysis['uniformity']} (0=uniform, 1=varied)")
    
    # Create comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"\n{'Configuration':<20} {'Codes':<10} {'Memory(MB)':<12} {'Speed':<10} {'Use Case':<25}")
    print("-"*80)
    
    use_cases = {
        'Minimal (Debug)': 'Testing & debugging',
        'Speed-Optimized': 'Quick experiments',
        'Balanced (Default)': 'General purpose',
        'Accuracy-Focused': 'Best accuracy',
        'Uniform Small': 'Simple patterns',
        'Uniform Large': 'Complex uniform data',
        'Imbalanced (Bad)': '❌ Avoid - poor balance'
    }
    
    for name, analysis in results:
        print(f"{name:<20} {analysis['total_codes']:<10} "
              f"{analysis['memory_mb']:<12.1f} "
              f"{analysis['speed_factor']:<10.2f}x "
              f"{use_cases[name]:<25}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS BY SCENARIO")
    print("="*80)
    
    scenarios = [
        ("🚀 Real-time inference on edge device", [8, 4, 4, 4], "512 codes, 2MB, fast"),
        ("🎯 Best accuracy with sufficient data", [16, 8, 8, 6], "6144 codes, 24MB, slower"),
        ("⚖️ Balanced performance (recommended)", [8, 5, 5, 4], "800 codes, 3MB, good speed"),
        ("🐛 Quick prototyping", [4, 4, 4, 4], "256 codes, 1MB, very fast"),
        ("📊 High-dimensional data", [8, 6, 5, 5, 4], "4800 codes, 5 dimensions"),
    ]
    
    for scenario, levels, reason in scenarios:
        analysis = analyze_fsq_config(levels)
        print(f"\n{scenario}")
        print(f"  Levels: {levels}")
        print(f"  Reason: {reason}")
        print(f"  Codes: {analysis['total_codes']:,}, Memory: {analysis['memory_mb']}MB")
    
    # Warning about common mistakes
    print("\n" + "="*80)
    print("⚠️ COMMON MISTAKES TO AVOID")
    print("="*80)
    
    mistakes = [
        ("Too many codes", [32, 32, 32], "32768 codes - massive overfitting!"),
        ("Too few codes", [2, 2, 2, 2], "16 codes - severe underfitting!"),
        ("Extreme imbalance", [128, 2, 2], "First dim dominates, others wasted"),
        ("Non-power-of-2 waste", [15, 15, 15], "Not utilizing bit efficiency"),
    ]
    
    for mistake, levels, issue in mistakes:
        analysis = analyze_fsq_config(levels)
        print(f"\n❌ {mistake}: {levels}")
        print(f"   Issue: {issue}")
        print(f"   Result: {analysis['total_codes']:,} codes, {analysis['memory_mb']}MB")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\n1. Default [8,5,5,4] is well-balanced for most tasks")
    print("2. Powers of 2 (4,8,16) are computationally efficient")
    print("3. Start small and increase if utilization >80%")
    print("4. Monitor perplexity to detect underutilization")
    print("5. Consider your hardware constraints (memory/speed)")
    
    print("\nFor detailed guidance, see FSQ_LEVELS_GUIDE.md")

if __name__ == "__main__":
    compare_configurations()