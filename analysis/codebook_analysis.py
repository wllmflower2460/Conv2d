"""
Codebook Analysis and Visualization Tools for Conv2d-VQ Model
Analyzes learned behavioral primitives and their usage patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.conv2d_vq_model import Conv2dVQModel
from torch.utils.data import DataLoader


class CodebookAnalyzer:
    """Comprehensive analysis of VQ codebook patterns"""
    
    def __init__(self, model: Conv2dVQModel, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Extract codebook
        self.codebook = model.vq.embedding.detach().cpu().numpy()  # (num_codes, code_dim)
        self.num_codes = self.codebook.shape[0]
        self.code_dim = self.codebook.shape[1]
        
        # Storage for analysis results
        self.code_frequencies = Counter()
        self.transition_matrix = np.zeros((self.num_codes, self.num_codes))
        self.human_codes = []
        self.dog_codes = []
        
    def collect_code_statistics(self, dataloader: DataLoader, max_batches: int = None):
        """Collect usage statistics from data"""
        print("Collecting code statistics...")
        
        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if max_batches and batch_count >= max_batches:
                    break
                
                # Handle dict format from enhanced pipeline
                if isinstance(batch, dict):
                    data = batch['input'].to(self.device)
                else:
                    data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                
                # Get quantized indices
                _, _, indices = self.model.encode_only(data)
                indices = indices.cpu().numpy()  # (B, H, T)
                
                # Separate human and dog codes (H dimension)
                for b in range(indices.shape[0]):
                    if indices.shape[1] == 2:  # Has device dimension
                        human_seq = indices[b, 0, :]
                        dog_seq = indices[b, 1, :]
                        self.human_codes.extend(human_seq.tolist())
                        self.dog_codes.extend(dog_seq.tolist())
                        
                        # Update transitions for both
                        self._update_transitions(human_seq)
                        self._update_transitions(dog_seq)
                    else:
                        # Single device
                        seq = indices[b, 0, :]
                        self.human_codes.extend(seq.tolist())
                        self._update_transitions(seq)
                    
                    # Update frequency counts
                    for idx in indices[b].flatten():
                        self.code_frequencies[int(idx)] += 1
                
                batch_count += 1
        
        # Normalize transition matrix
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix = self.transition_matrix / row_sums
        
        print(f"Analyzed {batch_count} batches")
        print(f"Total codes collected: {len(self.human_codes) + len(self.dog_codes)}")
        
    def _update_transitions(self, sequence):
        """Update transition matrix from a sequence"""
        for i in range(len(sequence) - 1):
            self.transition_matrix[sequence[i], sequence[i+1]] += 1
    
    def visualize_codebook_structure(self, save_path: str = None):
        """Visualize codebook organization using dimensionality reduction"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. PCA visualization
        pca = PCA(n_components=2)
        codes_pca = pca.fit_transform(self.codebook)
        
        # Color by usage frequency
        usage = np.array([self.code_frequencies.get(i, 0) for i in range(self.num_codes)])
        usage_log = np.log1p(usage)  # Log scale for better visibility
        
        scatter = axes[0, 0].scatter(codes_pca[:, 0], codes_pca[:, 1], 
                                     c=usage_log, cmap='viridis', s=50, alpha=0.6)
        axes[0, 0].set_title('PCA of Codebook (colored by usage)')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.colorbar(scatter, ax=axes[0, 0], label='Log Usage')
        
        # 2. t-SNE visualization
        if self.num_codes > 30:  # t-SNE needs enough points
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, self.num_codes//2))
            codes_tsne = tsne.fit_transform(self.codebook)
            
            scatter = axes[0, 1].scatter(codes_tsne[:, 0], codes_tsne[:, 1],
                                         c=usage_log, cmap='viridis', s=50, alpha=0.6)
            axes[0, 1].set_title('t-SNE of Codebook')
            axes[0, 1].set_xlabel('t-SNE 1')
            axes[0, 1].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[0, 1], label='Log Usage')
        
        # 3. Usage distribution
        usage_sorted = sorted(usage, reverse=True)
        axes[0, 2].plot(usage_sorted, 'b-', linewidth=2)
        axes[0, 2].set_title('Code Usage Distribution')
        axes[0, 2].set_xlabel('Code Rank')
        axes[0, 2].set_ylabel('Usage Count')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Pairwise distances heatmap
        distances = np.linalg.norm(self.codebook[:, None] - self.codebook[None, :], axis=2)
        top_codes = sorted(range(self.num_codes), key=lambda x: usage[x], reverse=True)[:50]
        dist_subset = distances[np.ix_(top_codes, top_codes)]
        
        sns.heatmap(dist_subset, ax=axes[1, 0], cmap='coolwarm', cbar_kws={'label': 'L2 Distance'})
        axes[1, 0].set_title('Pairwise Distances (Top 50 codes)')
        axes[1, 0].set_xlabel('Code Index')
        axes[1, 0].set_ylabel('Code Index')
        
        # 5. Transition matrix (top codes)
        trans_subset = self.transition_matrix[np.ix_(top_codes[:30], top_codes[:30])]
        sns.heatmap(trans_subset, ax=axes[1, 1], cmap='Blues', cbar_kws={'label': 'Transition Prob'})
        axes[1, 1].set_title('Transition Matrix (Top 30 codes)')
        axes[1, 1].set_xlabel('Next Code')
        axes[1, 1].set_ylabel('Current Code')
        
        # 6. Human vs Dog code usage
        if self.human_codes and self.dog_codes:
            human_freq = Counter(self.human_codes)
            dog_freq = Counter(self.dog_codes)
            
            # Get shared and unique codes
            human_set = set(human_freq.keys())
            dog_set = set(dog_freq.keys())
            shared = human_set & dog_set
            human_only = human_set - dog_set
            dog_only = dog_set - human_set
            
            labels = ['Shared', 'Human Only', 'Dog Only']
            sizes = [len(shared), len(human_only), len(dog_only)]
            colors = ['#2ecc71', '#3498db', '#e74c3c']
            
            axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[1, 2].set_title('Code Usage: Human vs Dog')
        
        plt.suptitle(f'Codebook Analysis: {self.num_codes} codes, {self.code_dim}D', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def find_behavioral_clusters(self, n_clusters: int = 10):
        """Cluster codes to find behavioral primitives"""
        # Cluster the codebook
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.codebook)
        
        # Analyze each cluster
        clusters = {}
        for i in range(n_clusters):
            cluster_codes = np.where(cluster_labels == i)[0]
            
            # Get usage statistics for this cluster
            cluster_usage = sum(self.code_frequencies.get(code, 0) for code in cluster_codes)
            
            # Find most used codes in cluster
            top_codes = sorted(cluster_codes, 
                             key=lambda x: self.code_frequencies.get(x, 0), 
                             reverse=True)[:5]
            
            clusters[f"Cluster_{i}"] = {
                'size': len(cluster_codes),
                'total_usage': cluster_usage,
                'avg_usage': cluster_usage / max(len(cluster_codes), 1),
                'top_codes': top_codes,
                'center': kmeans.cluster_centers_[i].tolist()[:10]  # First 10 dims for display
            }
        
        # Sort clusters by usage
        clusters = dict(sorted(clusters.items(), 
                             key=lambda x: x[1]['total_usage'], 
                             reverse=True))
        
        return clusters
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in code sequences"""
        results = {}
        
        # Calculate code persistence (how long codes stay active)
        def calculate_persistence(codes):
            if not codes:
                return {}
            
            persistence = []
            current_code = codes[0]
            count = 1
            
            for code in codes[1:]:
                if code == current_code:
                    count += 1
                else:
                    persistence.append((current_code, count))
                    current_code = code
                    count = 1
            persistence.append((current_code, count))
            
            # Aggregate by code
            code_persistence = {}
            for code, duration in persistence:
                if code not in code_persistence:
                    code_persistence[code] = []
                code_persistence[code].append(duration)
            
            return {
                code: {
                    'mean_duration': np.mean(durations),
                    'std_duration': np.std(durations),
                    'max_duration': max(durations),
                    'occurrences': len(durations)
                }
                for code, durations in code_persistence.items()
            }
        
        # Analyze human codes
        if self.human_codes:
            results['human_persistence'] = calculate_persistence(self.human_codes)
            
        # Analyze dog codes  
        if self.dog_codes:
            results['dog_persistence'] = calculate_persistence(self.dog_codes)
        
        # Find most stable codes (high mean duration)
        all_persistence = {}
        for key in ['human_persistence', 'dog_persistence']:
            if key in results:
                all_persistence.update(results[key])
        
        if all_persistence:
            stable_codes = sorted(all_persistence.items(), 
                                key=lambda x: x[1]['mean_duration'], 
                                reverse=True)[:10]
            results['most_stable_codes'] = stable_codes
        
        # Calculate entropy of transitions
        if np.any(self.transition_matrix > 0):
            # Row-wise entropy (uncertainty of next state)
            row_entropy = []
            for row in self.transition_matrix:
                if row.sum() > 0:
                    row_norm = row / row.sum()
                    entropy = -np.sum(row_norm * np.log2(row_norm + 1e-10))
                    row_entropy.append(entropy)
            
            results['transition_entropy'] = {
                'mean': np.mean(row_entropy),
                'std': np.std(row_entropy),
                'max': np.max(row_entropy),
                'min': np.min(row_entropy)
            }
        
        return results
    
    def compare_species_patterns(self):
        """Compare behavioral patterns between human and dog"""
        if not self.human_codes or not self.dog_codes:
            return None
        
        results = {}
        
        # Frequency comparison
        human_freq = Counter(self.human_codes)
        dog_freq = Counter(self.dog_codes)
        
        # Normalize frequencies
        human_total = sum(human_freq.values())
        dog_total = sum(dog_freq.values())
        
        human_prob = {k: v/human_total for k, v in human_freq.items()}
        dog_prob = {k: v/dog_total for k, v in dog_freq.items()}
        
        # Calculate KL divergence
        all_codes = set(human_prob.keys()) | set(dog_prob.keys())
        kl_div = 0
        for code in all_codes:
            p_human = human_prob.get(code, 1e-10)
            p_dog = dog_prob.get(code, 1e-10)
            if p_human > 0:
                kl_div += p_human * np.log2(p_human / p_dog)
        
        results['kl_divergence'] = kl_div
        
        # Find species-specific codes
        human_specific = [code for code in human_freq 
                         if human_freq[code]/human_total > 
                            dog_freq.get(code, 0)/dog_total * 2]
        
        dog_specific = [code for code in dog_freq 
                       if dog_freq[code]/dog_total > 
                          human_freq.get(code, 0)/human_total * 2]
        
        results['human_specific_codes'] = human_specific[:20]
        results['dog_specific_codes'] = dog_specific[:20]
        
        # Calculate synchrony (codes used at same time steps)
        # This would need aligned sequences, simplified here
        shared_codes = set(human_freq.keys()) & set(dog_freq.keys())
        results['shared_code_ratio'] = len(shared_codes) / len(all_codes)
        
        return results
    
    def save_analysis_report(self, save_dir: str):
        """Save comprehensive analysis report"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        report = {
            'codebook_stats': {
                'num_codes': self.num_codes,
                'code_dim': self.code_dim,
                'active_codes': len(self.code_frequencies),
                'usage_ratio': len(self.code_frequencies) / self.num_codes
            },
            'usage_stats': {
                'total_codes_seen': sum(self.code_frequencies.values()),
                'most_used': self.code_frequencies.most_common(10),
                'least_used': [k for k in range(self.num_codes) 
                             if self.code_frequencies.get(k, 0) == 0][:10]
            }
        }
        
        # Add behavioral clusters
        clusters = self.find_behavioral_clusters()
        report['behavioral_clusters'] = clusters
        
        # Add temporal patterns
        temporal = self.analyze_temporal_patterns()
        report['temporal_patterns'] = temporal
        
        # Add species comparison
        species = self.compare_species_patterns()
        if species:
            report['species_comparison'] = species
        
        # Save JSON report
        with open(save_dir / 'codebook_analysis.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save visualizations
        self.visualize_codebook_structure(str(save_dir / 'codebook_visualization.png'))
        
        print(f"Analysis report saved to {save_dir}")
        return report


def main():
    """Run codebook analysis on a trained model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze VQ codebook')
    parser.add_argument('--checkpoint', type=str, 
                       default='models/best_conv2d_vq_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, 
                       default='analysis/codebook_results',
                       help='Directory to save results')
    parser.add_argument('--max_batches', type=int, default=50,
                       help='Maximum batches to analyze')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    
    model = Conv2dVQModel(
        input_channels=9,
        input_height=2,
        num_codes=512,
        code_dim=64
    )
    
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}, using random initialization")
    
    # Create analyzer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    analyzer = CodebookAnalyzer(model, device)
    
    # Load data
    try:
        from preprocessing.enhanced_pipeline import get_dataset
        
        dataset = get_dataset(
            approach='cross_species',
            config_path='configs/enhanced_dataset_schema.yaml',
            mode='val',
            enforce_hailo_constraints=True
        )
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        print(f"Loaded dataset with {len(dataset)} samples")
        
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Creating dummy data for demonstration")
        
        from torch.utils.data import TensorDataset
        dummy_data = torch.randn(100, 9, 2, 100)
        dummy_dataset = TensorDataset(dummy_data)
        dataloader = DataLoader(dummy_dataset, batch_size=32)
    
    # Collect statistics
    analyzer.collect_code_statistics(dataloader, max_batches=args.max_batches)
    
    # Generate report
    report = analyzer.save_analysis_report(args.save_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("CODEBOOK ANALYSIS SUMMARY")
    print("="*50)
    print(f"Active codes: {report['codebook_stats']['active_codes']}/{report['codebook_stats']['num_codes']}")
    print(f"Usage ratio: {report['codebook_stats']['usage_ratio']:.1%}")
    
    if 'species_comparison' in report and report['species_comparison']:
        print(f"\nSpecies Comparison:")
        print(f"  KL Divergence: {report['species_comparison']['kl_divergence']:.3f}")
        print(f"  Shared code ratio: {report['species_comparison']['shared_code_ratio']:.1%}")
    
    if 'temporal_patterns' in report and 'transition_entropy' in report['temporal_patterns']:
        print(f"\nTemporal Patterns:")
        print(f"  Mean transition entropy: {report['temporal_patterns']['transition_entropy']['mean']:.2f}")
    
    print(f"\n✅ Analysis complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()