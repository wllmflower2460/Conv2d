"""
Codebook Usage and Utilization Metrics
Provides per-dimension occupancy and perplexity analysis for committee review
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from collections import Counter


class CodebookUsageAnalyzer:
    """
    Analyze FSQ codebook utilization with per-dimension metrics.
    
    Addresses committee requirement for:
    - 8-64 code sweep evidence
    - Per-dimension occupancy visibility
    - Perplexity metrics to diagnose under-utilization
    """
    
    def __init__(self, levels: List[int]):
        """
        Initialize usage analyzer.
        
        Args:
            levels: List of quantization levels per dimension
        """
        self.levels = levels
        self.dim = len(levels)
        self.total_codes = int(np.prod(levels))
        
        # Tracking structures
        self.joint_code_counts = Counter()
        self.per_dim_counts = [Counter() for _ in range(self.dim)]
        self.total_samples = 0
        
    def update(self, codes: torch.Tensor, joint_indices: Optional[torch.Tensor] = None):
        """
        Update usage statistics with batch of codes.
        
        Args:
            codes: Per-dimension codes (B, D) with values in [0, L_i)
            joint_indices: Optional joint codebook indices (B,)
        """
        batch_size = codes.shape[0]
        self.total_samples += batch_size
        
        # Update per-dimension counts
        for d in range(self.dim):
            dim_codes = codes[:, d].cpu().numpy()
            for code in dim_codes:
                self.per_dim_counts[d][int(code)] += 1
        
        # Update joint counts if provided
        if joint_indices is not None:
            for idx in joint_indices.cpu().numpy():
                self.joint_code_counts[int(idx)] += 1
        else:
            # Compute joint indices from per-dim codes
            joint_indices = self._compute_joint_indices(codes)
            for idx in joint_indices:
                self.joint_code_counts[idx] += 1
    
    def _compute_joint_indices(self, codes: torch.Tensor) -> np.ndarray:
        """
        Convert per-dimension codes to joint indices.
        
        Vectorized implementation using numpy's dot product
        for better performance with large batches.
        
        Args:
            codes: Per-dimension codes (B, D)
            
        Returns:
            joint_indices: Joint codebook indices (B,)
        """
        # Convert to numpy once (avoid repeated conversions in loop)
        codes_np = codes.cpu().numpy()
        
        # Compute multipliers for each dimension (vectorized)
        # For levels [L0, L1, L2], multipliers are [L1*L2, L2, 1]
        multipliers = np.ones(self.dim, dtype=np.int64)
        for d in range(self.dim - 2, -1, -1):
            multipliers[d] = multipliers[d + 1] * self.levels[d + 1]
        
        # Vectorized computation using dot product
        # This replaces the loop: sum(codes[i] * multiplier[i] for all i)
        joint_indices = np.dot(codes_np, multipliers).astype(np.int64)
        
        return joint_indices
    
    def get_per_dim_usage(self) -> List[Dict]:
        """
        Compute per-dimension usage statistics.
        
        Returns:
            List of per-dimension statistics including perplexity
        """
        usage_stats = []
        
        for d, L in enumerate(self.levels):
            # Get counts for this dimension
            counts = np.zeros(L)
            for code, count in self.per_dim_counts[d].items():
                if code < L:  # Safety check
                    counts[code] = count
            
            # Compute probabilities
            probs = counts / max(counts.sum(), 1.0)
            
            # Compute entropy and perplexity
            H = 0.0
            for p in probs:
                if p > 0:
                    H -= p * np.log(p)
            
            perplexity = np.exp(H)
            
            # Count active codes
            active_codes = int(np.sum(counts > 0))
            
            # Compute uniformity (how evenly distributed)
            if active_codes > 0:
                active_probs = probs[probs > 0]
                uniformity = 1.0 - np.std(active_probs) / (np.mean(active_probs) + 1e-8)
            else:
                uniformity = 0.0
            
            usage_stats.append({
                'dim': d,
                'levels': L,
                'active_codes': active_codes,
                'utilization': active_codes / L * 100,
                'perplexity': perplexity,
                'max_perplexity': L,  # Maximum possible perplexity
                'normalized_perplexity': perplexity / L * 100,
                'uniformity': uniformity,
                'most_frequent': int(np.argmax(counts)),
                'most_frequent_prob': float(np.max(probs))
            })
        
        return usage_stats
    
    def get_joint_usage(self) -> Dict:
        """
        Compute joint codebook usage statistics.
        
        Returns:
            Joint usage statistics
        """
        # Count unique codes used
        unique_codes = len(self.joint_code_counts)
        
        # Compute joint distribution entropy
        counts = np.array(list(self.joint_code_counts.values()))
        probs = counts / counts.sum()
        
        H = -np.sum(probs * np.log(probs + 1e-10))
        perplexity = np.exp(H)
        
        # Get top-k most frequent codes
        top_k = 10
        most_common = self.joint_code_counts.most_common(top_k)
        
        return {
            'total_codes': self.total_codes,
            'unique_codes_used': unique_codes,
            'utilization': unique_codes / self.total_codes * 100,
            'perplexity': perplexity,
            'max_perplexity': self.total_codes,
            'normalized_perplexity': perplexity / self.total_codes * 100,
            'total_samples': self.total_samples,
            'top_k_codes': [
                {
                    'index': idx,
                    'count': count,
                    'probability': count / self.total_samples
                }
                for idx, count in most_common
            ]
        }
    
    def generate_report(self) -> str:
        """
        Generate committee-friendly utilization report.
        
        Returns:
            Formatted report string
        """
        per_dim = self.get_per_dim_usage()
        joint = self.get_joint_usage()
        
        report = []
        report.append("="*70)
        report.append("CODEBOOK UTILIZATION REPORT")
        report.append("="*70)
        
        # Joint statistics
        report.append("\nJOINT CODEBOOK STATISTICS:")
        report.append(f"  Total codes available: {joint['total_codes']:,}")
        report.append(f"  Unique codes used: {joint['unique_codes_used']:,}")
        report.append(f"  Utilization: {joint['utilization']:.1f}%")
        report.append(f"  Perplexity: {joint['perplexity']:.1f} / {joint['max_perplexity']:,} "
                     f"({joint['normalized_perplexity']:.1f}%)")
        report.append(f"  Total samples: {joint['total_samples']:,}")
        
        # Per-dimension breakdown
        report.append("\nPER-DIMENSION BREAKDOWN:")
        report.append("-"*70)
        report.append(f"{'Dim':<5} {'Levels':<8} {'Active':<8} {'Util%':<8} "
                     f"{'Perplexity':<12} {'Uniform':<10} {'Top Code':<10}")
        report.append("-"*70)
        
        for stat in per_dim:
            report.append(
                f"{stat['dim']:<5} {stat['levels']:<8} {stat['active_codes']:<8} "
                f"{stat['utilization']:<8.1f} {stat['perplexity']:<6.1f}/{stat['levels']:<5} "
                f"{stat['uniformity']:<10.2f} {stat['most_frequent']:<3} "
                f"({stat['most_frequent_prob']*100:.1f}%)"
            )
        
        # Diagnosis
        report.append("\nDIAGNOSIS:")
        
        # Check for dimension bottlenecks
        bottlenecks = [s for s in per_dim if s['utilization'] < 50]
        if bottlenecks:
            report.append("  ⚠ Under-utilized dimensions detected:")
            for b in bottlenecks:
                report.append(f"    - Dim {b['dim']}: only {b['utilization']:.1f}% utilization")
        
        # Check for over-parameterization
        if joint['utilization'] < 10:
            report.append(f"  ⚠ Severe under-utilization: Only {joint['utilization']:.1f}% of codes used")
            report.append("    → Consider reducing codebook size or levels per dimension")
        elif joint['utilization'] < 50:
            report.append(f"  ⚠ Moderate under-utilization: {joint['utilization']:.1f}% of codes used")
        
        # Check perplexity
        if joint['normalized_perplexity'] < 10:
            report.append(f"  ⚠ Low perplexity ({joint['normalized_perplexity']:.1f}%): "
                         "Distribution is highly concentrated")
        
        report.append("="*70)
        
        return "\n".join(report)
    
    def plot_usage_heatmap(self, save_path: Optional[str] = None):
        """
        Create visualization of per-dimension usage patterns.
        
        Args:
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Per-dimension utilization
        per_dim = self.get_per_dim_usage()
        dims = [s['dim'] for s in per_dim]
        utils = [s['utilization'] for s in per_dim]
        perps = [s['normalized_perplexity'] for s in per_dim]
        
        ax1 = axes[0]
        x = np.arange(len(dims))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, utils, width, label='Utilization %', color='steelblue')
        bars2 = ax1.bar(x + width/2, perps, width, label='Normalized Perplexity %', color='coral')
        
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Percentage')
        ax1.set_title('Per-Dimension Codebook Utilization')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Dim {d}\n(L={s['levels']})" for d, s in enumerate(per_dim)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Distribution of code usage (top dimensions)
        ax2 = axes[1]
        
        # Show distribution for first 4 dimensions
        for d in range(min(4, self.dim)):
            counts = np.zeros(self.levels[d])
            for code, count in self.per_dim_counts[d].items():
                if code < self.levels[d]:
                    counts[code] = count
            
            if counts.sum() > 0:
                probs = counts / counts.sum()
                ax2.plot(probs, label=f'Dim {d} (L={self.levels[d]})', alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Code Index')
        ax2.set_ylabel('Probability')
        ax2.set_title('Code Usage Distribution (First 4 Dimensions)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved usage plot to {save_path}")
        
        return fig


class CodebookSweepAnalyzer:
    """Run and analyze codebook sweeps for committee review."""
    
    def __init__(self):
        """Initialize sweep analyzer."""
        self.results = []
    
    def run_sweep(self, data_generator, sizes=[8, 16, 32, 64], n_samples=10000):
        """
        Run codebook sweep with different sizes.
        
        Args:
            data_generator: Function that returns (features, codes) for each size
            sizes: List of codebook sizes to test
            n_samples: Number of samples to generate per size
            
        Returns:
            Sweep results
        """
        for size in sizes:
            print(f"\nTesting codebook size: {size}")
            
            # Determine level allocation (uniform for simplicity)
            dim = int(np.ceil(np.log2(size)))
            if dim <= 3:
                levels = [2] * dim
            else:
                # Distribute bits across dimensions
                base_level = int(np.ceil(size ** (1/4)))
                levels = [base_level] * 4
            
            # Initialize analyzer
            analyzer = CodebookUsageAnalyzer(levels)
            
            # Generate and analyze data
            features, codes = data_generator(size, levels, n_samples)
            
            # Process in batches
            batch_size = 1000
            for i in range(0, n_samples, batch_size):
                batch_codes = codes[i:i+batch_size]
                analyzer.update(batch_codes)
            
            # Collect results
            result = {
                'target_size': size,
                'levels': levels,
                'actual_size': int(np.prod(levels)),
                'per_dim_stats': analyzer.get_per_dim_usage(),
                'joint_stats': analyzer.get_joint_usage(),
                'report': analyzer.generate_report()
            }
            
            self.results.append(result)
            
            # Print summary
            joint = result['joint_stats']
            print(f"  Actual size: {joint['total_codes']}")
            print(f"  Utilization: {joint['utilization']:.1f}%")
            print(f"  Perplexity: {joint['perplexity']:.1f}")
        
        return self.results
    
    def generate_committee_table(self) -> str:
        """
        Generate committee-ready comparison table.
        
        Returns:
            Formatted table
        """
        if not self.results:
            return "No results available"
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append("COMMITTEE CODEBOOK SWEEP RESULTS")
        lines.append("="*80)
        
        # Header
        lines.append(f"{'Target':<10} {'Actual':<10} {'Dims':<15} {'Utilization':<15} "
                    f"{'Perplexity':<15} {'Diagnosis':<20}")
        lines.append("-"*80)
        
        for r in self.results:
            joint = r['joint_stats']
            levels_str = str(r['levels'])
            
            # Diagnosis
            if joint['utilization'] < 10:
                diagnosis = "SEVERE under-use"
            elif joint['utilization'] < 50:
                diagnosis = "Moderate under-use"
            elif joint['utilization'] > 90:
                diagnosis = "Good utilization"
            else:
                diagnosis = "Acceptable"
            
            lines.append(
                f"{r['target_size']:<10} {joint['total_codes']:<10} {levels_str:<15} "
                f"{joint['utilization']:<15.1f}% {joint['perplexity']:<15.1f} {diagnosis:<20}"
            )
        
        lines.append("="*80)
        
        return "\n".join(lines)


# Example usage and testing
def example_data_generator(target_size, levels, n_samples):
    """
    Example data generator for testing.
    Simulates realistic FSQ codes with some clustering.
    """
    torch.manual_seed(42)
    dim = len(levels)
    
    # Generate features with some structure (not uniform)
    features = torch.randn(n_samples, dim)
    
    # Add some clustering to make usage non-uniform
    n_clusters = min(target_size // 4, 10)
    cluster_centers = torch.randn(n_clusters, dim) * 0.5
    
    for i in range(n_samples):
        if torch.rand(1).item() < 0.7:  # 70% clustered
            cluster_idx = torch.randint(0, n_clusters, (1,)).item()
            features[i] += cluster_centers[cluster_idx]
    
    # Quantize to get codes
    codes = torch.zeros(n_samples, dim, dtype=torch.long)
    for d in range(dim):
        # Map to [0, L_d)
        scaled = torch.sigmoid(features[:, d]) * levels[d]
        codes[:, d] = torch.clamp(scaled.long(), 0, levels[d] - 1)
    
    return features, codes


if __name__ == "__main__":
    # Test single codebook analysis
    print("Single Codebook Analysis:")
    print("-"*40)
    
    levels = [4, 4, 3, 3, 2, 2]
    analyzer = CodebookUsageAnalyzer(levels)
    
    # Generate test data
    features, codes = example_data_generator(64, levels, 5000)
    
    # Update analyzer
    for i in range(0, 5000, 500):
        analyzer.update(codes[i:i+500])
    
    # Generate report
    print(analyzer.generate_report())
    
    # Run codebook sweep
    print("\n\nCodebook Sweep Analysis:")
    print("-"*40)
    
    sweep = CodebookSweepAnalyzer()
    sweep.run_sweep(example_data_generator, sizes=[8, 16, 32, 64])
    
    # Generate committee table
    print(sweep.generate_committee_table())
    
    # Create visualization
    analyzer.plot_usage_heatmap("codebook_usage.png")
