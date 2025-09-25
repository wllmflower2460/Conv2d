"""
Mutual Information and Transfer Entropy Fixes
Addresses committee feedback on:
1. Bessel ratio correction for von Mises entropy
2. Consistent log base handling (nats vs bits)
3. k-NN CMI estimator for transfer entropy
"""

import numpy as np
import torch
from scipy.special import i0, i1
from torch import cdist


class MITEFixes:
    """Corrected MI and TE calculations for committee review"""
    
    def compute_von_mises_entropy(self, kappa_val):
        """
        Compute von Mises differential entropy with correct Bessel ratio.
        Fixed: was using i0(kappa)/i0(kappa), now uses i1(kappa)/i0(kappa)
        
        Args:
            kappa_val: Concentration parameter
            
        Returns:
            h_vm: Differential entropy in nats
        """
        # FIXED: Correct Bessel ratio (was i0/i0, now i1/i0)
        bessel_ratio = i1(kappa_val) / i0(kappa_val)
        
        # Von Mises entropy in nats (natural log)
        h_vm = np.log(2 * np.pi * i0(kappa_val)) - kappa_val * bessel_ratio
        
        return h_vm
    
    def compute_mutual_information(self, h_marginal, h_conditional):
        """
        Compute MI as I(X;Y) = H(X) - H(X|Y)
        
        Args:
            h_marginal: Marginal entropy H(X) in nats
            h_conditional: Conditional entropy H(X|Y) in nats
            
        Returns:
            Dict with MI in both nats and bits
        """
        mi_nats = h_marginal - h_conditional
        mi_bits = mi_nats / np.log(2)  # Convert to bits for R-D comparison
        
        return {
            'mi_nats': mi_nats,
            'mi_bits': mi_bits,
            'units_note': 'MI computed in nats, converted to bits for rate-distortion comparison'
        }
    
    def estimate_cmi_knn(self, X, Y, Z, k=5):
        """
        k-NN based conditional mutual information estimator.
        Estimates I(X;Y|Z) using Kraskov-type estimator.
        
        REPLACES: Placeholder TE estimator that returned fixed 0.1
        
        Args:
            X: (N, d_x) tensor - past states
            Y: (N, d_y) tensor - future states  
            Z: (N, d_z) tensor - conditioning variable (past of Y)
            k: Number of nearest neighbors
            
        Returns:
            cmi: Estimated CMI in nats
        """
        # Convert to float tensors
        X = X.float()
        Y = Y.float()
        Z = Z.float()
        
        def digamma(n):
            """Fast digamma (psi) function"""
            return torch.digamma(torch.tensor(float(n))).item()
        
        # Get epsilon based on k-th neighbor in Z-space
        NZ = Z.shape[0]
        Dz = cdist(Z, Z, p=float('inf'))  # Chebyshev distance
        
        # k+1 because we exclude self (add large value to diagonal)
        eps = torch.kthvalue(Dz + torch.eye(NZ) * 1e9, k + 1, dim=1).values
        
        def count_within(U, radius):
            """Count neighbors within radius using Chebyshev balls"""
            DU = cdist(U, U, p=float('inf'))
            return (DU <= radius.unsqueeze(1)).sum(dim=1) - 1  # Exclude self
        
        # Count neighbors in different joint spaces
        nxz = count_within(torch.cat([X, Z], dim=1), eps)
        nyz = count_within(torch.cat([Y.unsqueeze(1) if Y.dim() == 1 else Y, Z], dim=1), eps)
        nxyz = count_within(torch.cat([X, Y.unsqueeze(1) if Y.dim() == 1 else Y, Z], dim=1), eps)
        nz = count_within(Z, eps)
        
        # Kraskov-type CMI estimate
        cmi = (digamma(k) + 
               (nyz.float().add(1e-9)).digamma().neg() +
               (nxz.float().add(1e-9)).digamma().neg() + 
               (nz.float().add(1e-9)).digamma())
        
        return cmi.mean().clamp_min(0.0)
    
    def compute_transfer_entropy(self, X_past, Y_future, Y_past, k=5):
        """
        Compute transfer entropy T(X→Y) = I(X_past; Y_future | Y_past)
        
        Args:
            X_past: Past values of source process
            Y_future: Future values of target process
            Y_past: Past values of target process
            k: Number of neighbors for k-NN estimator
            
        Returns:
            Dict with TE in nats and bits
        """
        te_nats = self.estimate_cmi_knn(X_past, Y_future, Y_past, k=k)
        te_bits = te_nats / np.log(2)
        
        return {
            'te_nats': float(te_nats),
            'te_bits': float(te_bits),
            'method': 'k-NN CMI estimator (Kraskov-type)',
            'k_neighbors': k
        }


# Example usage and tests
if __name__ == "__main__":
    fixes = MITEFixes()
    
    # Test 1: Von Mises entropy with various kappa
    print("Von Mises Entropy Tests:")
    for kappa in [0.5, 1.0, 2.0, 5.0]:
        h_vm = fixes.compute_von_mises_entropy(kappa)
        print(f"  κ={kappa}: H(Φ) = {h_vm:.4f} nats = {h_vm/np.log(2):.4f} bits")
    
    # Test 2: MI calculation with unit conversion
    print("\nMutual Information Test:")
    h_marginal = 2.5  # nats
    h_conditional = 1.2  # nats
    mi_result = fixes.compute_mutual_information(h_marginal, h_conditional)
    print(f"  H(X) = {h_marginal:.2f} nats")
    print(f"  H(X|Y) = {h_conditional:.2f} nats")
    print(f"  I(X;Y) = {mi_result['mi_nats']:.4f} nats = {mi_result['mi_bits']:.4f} bits")
    
    # Test 3: Transfer entropy with k-NN
    print("\nTransfer Entropy Test:")
    N = 1000
    X_past = torch.randn(N, 3)
    Y_future = torch.randn(N, 1) 
    Y_past = torch.randn(N, 2)
    
    te_result = fixes.compute_transfer_entropy(X_past, Y_future, Y_past)
    print(f"  T(X→Y) = {te_result['te_nats']:.4f} nats = {te_result['te_bits']:.4f} bits")
    print(f"  Method: {te_result['method']}")
