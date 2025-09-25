"""
FSQ Rate-Distortion Rounding Improvements
Addresses committee feedback on integer-level allocation with marginal cost optimization
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FSQRateDistortionOptimizer:
    """Improved FSQ level allocation using marginal cost greedy algorithm"""
    
    def __init__(self, target_rate, Lmin=2, Lmax=256):
        """
        Initialize FSQ R-D optimizer.
        
        Args:
            target_rate: Target rate in bits
            Lmin: Minimum levels per dimension
            Lmax: Maximum levels per dimension
        """
        self.target_rate = target_rate
        self.Lmin = Lmin
        self.Lmax = Lmax
        
    def water_filling_allocation(self, variances):
        """
        Initial water-filling allocation for continuous rates.
        
        Args:
            variances: Per-dimension variances
            
        Returns:
            continuous_rates: Real-valued bit allocations
        """
        dim = len(variances)
        lambda_wf = 2 ** (2 * self.target_rate / dim)
        
        # Water-filling formula: b_i = 0.5 * log2(lambda * sigma_i^2)
        continuous_rates = 0.5 * np.log2(lambda_wf * np.array(variances))
        continuous_rates = np.maximum(continuous_rates, np.log2(self.Lmin))
        
        return continuous_rates
    
    def marginal_cost(self, variances, L, idx):
        """
        Compute marginal distortion increase per bit saved.
        
        IMPROVEMENT: Properly accounts for the distortion-rate tradeoff
        when reducing levels at each dimension.
        
        Args:
            variances: Per-dimension variances
            L: Current level allocation
            idx: Dimension index to evaluate
            
        Returns:
            cost: Marginal cost (distortion increase per bit saved)
        """
        L_i = L[idx]
        if L_i <= self.Lmin:
            return float('inf')  # Can't reduce below minimum
        
        # High-rate scalar quantization: D ~ sigma^2 / L^2
        d_now = variances[idx] / (L_i ** 2)
        d_less = variances[idx] / ((L_i - 1) ** 2)
        
        # Distortion increase
        delta_d = d_less - d_now
        
        # Bits saved by reducing L_i to L_i-1
        delta_bits = np.log2(L_i) - np.log2(L_i - 1)
        
        # Cost = distortion increase per bit saved (want to minimize)
        return delta_d / (delta_bits + 1e-12)
    
    def optimize_integer_levels(self, variances):
        """
        Optimize integer level allocation with marginal cost greedy adjustment.
        
        REPLACES: Simple decrement of largest L
        NEW: Greedy selection based on marginal distortion cost
        
        Args:
            variances: Per-dimension variances
            
        Returns:
            Dict with optimized levels and metrics
        """
        dim = len(variances)
        
        # Step 1: Get continuous rates from water-filling
        continuous_rates = self.water_filling_allocation(variances)
        
        # Step 2: Round to integer levels
        L = np.round(2 ** continuous_rates).astype(int)
        L = np.clip(L, self.Lmin, self.Lmax)
        
        # Step 3: Greedy adjustment using marginal costs
        iteration = 0
        while np.sum(np.log2(L)) > self.target_rate + 1e-9:
            iteration += 1
            
            # Compute marginal costs for all dimensions
            costs = [self.marginal_cost(variances, L, i) for i in range(dim)]
            
            # Find dimension with minimum marginal cost (best to reduce)
            idx = np.argmin(costs)
            
            if L[idx] > self.Lmin:
                L[idx] -= 1
                logger.debug(f"  Iter {iteration}: Reduced L[{idx}] to {L[idx]} (cost={costs[idx]:.4f})")
            else:
                # Fallback: reduce any dimension still above Lmin
                candidates = np.where(L > self.Lmin)[0]
                if len(candidates) == 0:
                    logger.warning("Cannot achieve target rate - all dimensions at Lmin")
                    break
                L[candidates[0]] -= 1
        
        # Compute final metrics
        achieved_rate = np.sum(np.log2(L))
        proxy_distortion = np.sum(variances / (L ** 2))
        
        result = {
            'levels': L.tolist(),
            'target_rate': self.target_rate,
            'achieved_rate': achieved_rate,
            'rate_gap': achieved_rate - self.target_rate,
            'proxy_distortion': proxy_distortion,
            'iterations': iteration,
            'per_dim_bits': np.log2(L).tolist()
        }
        
        # Log for committee visibility
        logger.info(f"FSQ Rate-Distortion Optimization Complete:")
        logger.info(f"  Target rate: {self.target_rate:.2f} bits")
        logger.info(f"  Achieved rate: {achieved_rate:.2f} bits (gap: {result['rate_gap']:.4f})")
        logger.info(f"  Proxy distortion: {proxy_distortion:.6f}")
        logger.info(f"  Levels: {L}")
        
        return result


class FSQCodebookSweep:
    """Run codebook size sweeps for committee review"""
    
    def __init__(self, variances):
        """
        Initialize sweep runner.
        
        Args:
            variances: Per-dimension variances from data
        """
        self.variances = variances
        self.dim = len(variances)
    
    def run_sweep(self, codebook_sizes=[8, 16, 32, 64]):
        """
        Run sweep over different codebook sizes.
        
        Args:
            codebook_sizes: List of target codebook sizes to test
            
        Returns:
            results: List of optimization results
        """
        results = []
        
        print("\n" + "="*60)
        print("FSQ CODEBOOK SIZE SWEEP (Committee Table)")
        print("="*60)
        
        for size in codebook_sizes:
            target_rate = np.log2(size)  # Total bits for joint codebook
            
            optimizer = FSQRateDistortionOptimizer(target_rate)
            result = optimizer.optimize_integer_levels(self.variances)
            result['codebook_size'] = size
            result['actual_codebook_size'] = int(np.prod(result['levels']))
            
            results.append(result)
            
            print(f"\nCodebook Size: {size}")
            print(f"  Target Rate: {target_rate:.2f} bits")
            print(f"  Achieved Rate: {result['achieved_rate']:.2f} bits")
            print(f"  Actual Codebook: {result['actual_codebook_size']}")
            print(f"  Distortion: {result['proxy_distortion']:.6f}")
            print(f"  Levels: {result['levels']}")
        
        return results


# Example usage and tests
if __name__ == "__main__":
    # Simulated variances from PCA or encoder features
    variances = np.array([2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1])
    
    # Test single optimization
    print("Single Optimization Test:")
    optimizer = FSQRateDistortionOptimizer(target_rate=6.0)  # 64 codes
    result = optimizer.optimize_integer_levels(variances)
    
    # Run committee sweep
    sweep = FSQCodebookSweep(variances)
    sweep_results = sweep.run_sweep([8, 16, 32, 64])
    
    # Generate committee-friendly summary
    print("\n" + "="*60)
    print("COMMITTEE SUMMARY TABLE")
    print("="*60)
    print(f"{'Codebook':<10} {'Rate (bits)':<12} {'Distortion':<12} {'Efficiency':<12}")
    print("-"*46)
    
    for res in sweep_results:
        efficiency = res['achieved_rate'] / np.log2(res['codebook_size']) * 100
        print(f"{res['codebook_size']:<10} {res['achieved_rate']:<12.2f} "
              f"{res['proxy_distortion']:<12.4f} {efficiency:<12.1f}%")
