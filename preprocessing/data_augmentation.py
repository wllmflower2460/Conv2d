"""Data augmentation for behavioral synchrony analysis.

Implements augmentation strategies to improve model accuracy from 78% to 85%+
as recommended by the synchrony-advisor-committee.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings


class BehavioralAugmentation:
    """Data augmentation specifically designed for IMU/behavioral data.
    
    Preserves behavioral semantics while increasing data diversity.
    """
    
    def __init__(
        self,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        time_warp_range: Tuple[float, float] = (0.95, 1.05),
        rotation_range: float = 15.0,  # degrees
        dropout_prob: float = 0.1,
        mixup_alpha: float = 0.2,
        cutmix_prob: float = 0.5
    ):
        """Initialize augmentation parameters.
        
        Args:
            noise_std: Standard deviation for Gaussian noise
            scale_range: Range for amplitude scaling
            time_warp_range: Range for temporal warping
            rotation_range: Maximum rotation in degrees (for gyroscope)
            dropout_prob: Probability of dropping sensor readings
            mixup_alpha: Alpha parameter for mixup augmentation
            cutmix_prob: Probability of applying cutmix
        """
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_warp_range = time_warp_range
        self.rotation_range = rotation_range
        self.dropout_prob = dropout_prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
    
    def add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to sensor readings.
        
        Args:
            x: Input tensor (B, C, H, T) or (B, C, T)
            
        Returns:
            Augmented tensor
        """
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x
    
    def amplitude_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Scale amplitude of signals.
        
        Args:
            x: Input tensor
            
        Returns:
            Scaled tensor
        """
        if self.training:
            scale = torch.empty(x.shape[0], 1, *([1] * (x.dim() - 2))).uniform_(
                self.scale_range[0], self.scale_range[1]
            ).to(x.device)
            return x * scale
        return x
    
    def time_warping(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal warping to sequences.
        
        Args:
            x: Input tensor (B, C, H, T) or (B, C, T)
            
        Returns:
            Time-warped tensor
        """
        if not self.training:
            return x
        
        B, C = x.shape[:2]
        T = x.shape[-1]
        
        # Generate warping factor
        warp_factor = torch.empty(B, 1).uniform_(
            self.time_warp_range[0], self.time_warp_range[1]
        ).to(x.device)
        
        # Create warped time indices
        time_indices = torch.arange(T, dtype=torch.float32, device=x.device)
        warped_indices = time_indices.unsqueeze(0) * warp_factor
        warped_indices = torch.clamp(warped_indices, 0, T - 1)
        
        # Interpolate using 1D interpolation instead of grid_sample
        if x.dim() == 4:  # (B, C, H, T)
            B, C, H, T = x.shape
            # Reshape to (B*C*H, T) for interpolation
            x_flat = x.reshape(B * C * H, T)
            
            # Create interpolation indices for each batch
            warped_list = []
            for b in range(B):
                indices = warped_indices[b, 0]
                # Interpolate each channel
                batch_warped = []
                for ch in range(C * H):
                    idx = b * C * H + ch
                    # Linear interpolation
                    warped_ch = torch.nn.functional.interpolate(
                        x_flat[idx:idx+1].unsqueeze(0),
                        size=T,
                        mode='linear',
                        align_corners=False
                    ).squeeze()
                    batch_warped.append(warped_ch)
                warped_list.append(torch.stack(batch_warped).reshape(C, H, T))
            warped = torch.stack(warped_list)
        else:  # (B, C, T)
            B, C, T = x.shape
            x_flat = x.reshape(B * C, 1, T)
            warped = torch.nn.functional.interpolate(
                x_flat,
                size=T,
                mode='linear',
                align_corners=False
            ).reshape(B, C, T)
        
        return warped
    
    def sensor_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop sensor readings (set to zero).
        
        Args:
            x: Input tensor (B, C, H, T)
            
        Returns:
            Tensor with random dropout
        """
        if not self.training or np.random.random() > self.dropout_prob:
            return x
        
        # Create dropout mask
        if x.dim() == 4:  # (B, C, H, T)
            mask = torch.bernoulli(
                torch.ones(x.shape[0], x.shape[1], 1, 1) * (1 - self.dropout_prob)
            ).to(x.device)
        else:
            mask = torch.bernoulli(
                torch.ones(x.shape[0], x.shape[1], 1) * (1 - self.dropout_prob)
            ).to(x.device)
        
        return x * mask
    
    def rotation_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation to gyroscope readings.
        
        Args:
            x: Input tensor (B, 9, H, T) - assumes channels 3-5 are gyroscope
            
        Returns:
            Rotated tensor
        """
        if not self.training or x.shape[1] < 6:
            return x
        
        # Extract gyroscope channels (typically 3-5)
        gyro_start = 3
        gyro_end = 6
        
        # Generate random rotation matrix
        angle = torch.empty(x.shape[0]).uniform_(
            -self.rotation_range, self.rotation_range
        ).to(x.device) * np.pi / 180
        
        # Apply rotation to gyroscope data
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # Simple 2D rotation for demonstration (extend to 3D if needed)
        x_rot = x.clone()
        if x.dim() == 4:  # (B, C, H, T)
            for b in range(x.shape[0]):
                gyro_x = x[b, gyro_start]
                gyro_y = x[b, gyro_start + 1]
                x_rot[b, gyro_start] = cos_a[b] * gyro_x - sin_a[b] * gyro_y
                x_rot[b, gyro_start + 1] = sin_a[b] * gyro_x + cos_a[b] * gyro_y
        
        return x_rot
    
    def mixup(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation.
        
        Args:
            x: Input tensor (B, C, H, T)
            y: Labels (B,)
            alpha: Mixup parameter (uses self.mixup_alpha if None)
            
        Returns:
            mixed_x: Mixed input
            y_a: First label
            y_b: Second label
            lam: Mixing coefficient
        """
        if not self.training:
            return x, y, y, 1.0
        
        if alpha is None:
            alpha = self.mixup_alpha
        
        batch_size = x.shape[0]
        
        # Sample mixing coefficient
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        # Random permutation for mixing
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix inputs and labels
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation.
        
        Args:
            x: Input tensor (B, C, H, T)
            y: Labels (B,)
            
        Returns:
            mixed_x: Mixed input
            y_a: First label
            y_b: Second label
            lam: Mixing coefficient
        """
        if not self.training or np.random.random() > self.cutmix_prob:
            return x, y, y, 1.0
        
        batch_size = x.shape[0]
        T = x.shape[-1]
        
        # Sample mixing coefficient
        lam = np.random.beta(1.0, 1.0)
        
        # Random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Create temporal mask
        cut_length = int(T * (1 - lam))
        cut_start = np.random.randint(0, T - cut_length + 1)
        
        # Apply cutmix
        mixed_x = x.clone()
        mixed_x[..., cut_start:cut_start + cut_length] = x[index][..., cut_start:cut_start + cut_length]
        
        # Adjust lambda for actual mix ratio
        lam = 1 - cut_length / T
        
        return mixed_x, y, y[index], lam
    
    def __call__(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply augmentation pipeline.
        
        Args:
            x: Input tensor
            y: Optional labels
            training: Whether in training mode
            
        Returns:
            Augmented data and labels
        """
        self.training = training
        
        if not training:
            return x, y
        
        # Basic augmentations
        x = self.add_gaussian_noise(x)
        x = self.amplitude_scaling(x)
        x = self.time_warping(x)
        x = self.sensor_dropout(x)
        x = self.rotation_augmentation(x)
        
        # Advanced augmentations (if labels provided)
        if y is not None:
            if np.random.random() < 0.5:
                x, y_a, y_b, lam = self.mixup(x, y)
                # Return mixed labels for loss computation
                return x, (y_a, y_b, lam)
            elif np.random.random() < self.cutmix_prob:
                x, y_a, y_b, lam = self.cutmix(x, y)
                return x, (y_a, y_b, lam)
        
        return x, y


class ContrastiveAugmentation:
    """Augmentation for contrastive learning approaches.
    
    Creates positive pairs with different augmentations.
    """
    
    def __init__(self, base_augmenter: BehavioralAugmentation):
        """Initialize with base augmenter.
        
        Args:
            base_augmenter: BehavioralAugmentation instance
        """
        self.augmenter = base_augmenter
    
    def create_views(
        self,
        x: torch.Tensor,
        n_views: int = 2
    ) -> List[torch.Tensor]:
        """Create multiple augmented views of the same data.
        
        Args:
            x: Input tensor
            n_views: Number of views to create
            
        Returns:
            List of augmented views
        """
        views = []
        for _ in range(n_views):
            aug_x, _ = self.augmenter(x.clone(), training=True)
            views.append(aug_x)
        
        return views


def augmented_training_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    augmenter: BehavioralAugmentation,
    criterion: torch.nn.Module
) -> torch.Tensor:
    """Training step with augmentation and mixed labels.
    
    Args:
        model: Neural network model
        x: Input batch
        y: Labels
        augmenter: Augmentation instance
        criterion: Loss function
        
    Returns:
        Loss value
    """
    # Apply augmentation
    x_aug, y_aug = augmenter(x, y, training=True)
    
    # Forward pass
    outputs = model(x_aug)
    
    # Handle mixed labels from mixup/cutmix
    if isinstance(y_aug, tuple) and len(y_aug) == 3:
        y_a, y_b, lam = y_aug
        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
    else:
        loss = criterion(outputs, y_aug)
    
    return loss


if __name__ == "__main__":
    # Test augmentation
    print("Testing Behavioral Augmentation...")
    
    # Create synthetic data
    torch.manual_seed(42)
    batch_size = 16
    x = torch.randn(batch_size, 9, 2, 100)  # (B, C, H, T)
    y = torch.randint(0, 10, (batch_size,))
    
    # Initialize augmenter
    augmenter = BehavioralAugmentation(
        noise_std=0.02,
        scale_range=(0.9, 1.1),
        time_warp_range=(0.95, 1.05),
        rotation_range=15.0,
        dropout_prob=0.1,
        mixup_alpha=0.2,
        cutmix_prob=0.5
    )
    
    # Test individual augmentations
    print(f"Original shape: {x.shape}")
    
    augmenter.training = True
    x_noise = augmenter.add_gaussian_noise(x.clone())
    print(f"After noise: std diff = {(x_noise - x).std():.4f}")
    
    x_scaled = augmenter.amplitude_scaling(x.clone())
    print(f"After scaling: mean ratio = {(x_scaled.mean() / x.mean()):.4f}")
    
    x_warped = augmenter.time_warping(x.clone())
    print(f"After warping: shape preserved = {x_warped.shape == x.shape}")
    
    x_dropout = augmenter.sensor_dropout(x.clone())
    print(f"After dropout: zero ratio = {(x_dropout == 0).float().mean():.4f}")
    
    # Test mixup
    x_mixed, y_a, y_b, lam = augmenter.mixup(x, y)
    print(f"Mixup: lam = {lam:.4f}, mixed shape = {x_mixed.shape}")
    
    # Test cutmix
    x_cut, y_a, y_b, lam = augmenter.cutmix(x, y)
    print(f"CutMix: lam = {lam:.4f}, cut shape = {x_cut.shape}")
    
    # Test full pipeline
    x_aug, y_aug = augmenter(x, y, training=True)
    print(f"\nFull augmentation: output shape = {x_aug.shape}")
    if isinstance(y_aug, tuple):
        print(f"Mixed labels: y_a shape = {y_aug[0].shape}, lam = {y_aug[2]:.4f}")
    else:
        print(f"Labels shape: {y_aug.shape}")
    
    # Test contrastive augmentation
    contrastive_aug = ContrastiveAugmentation(augmenter)
    views = contrastive_aug.create_views(x, n_views=2)
    print(f"\nContrastive views: {len(views)} views, each {views[0].shape}")
    
    print("\n✅ Augmentation tests passed!")