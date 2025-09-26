"""Kinematic feature extraction for behavioral analysis.

This module extends Movement library kinematics for behavioral synchrony analysis,
extracting motion-based features that complement the Conv2d-VQ-HDP-HSMM architecture.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings
from pathlib import Path
import sys
import os

# Import Movement integration
from .movement_integration import MovementPreprocessor, MOVEMENT_AVAILABLE

if MOVEMENT_AVAILABLE:
    # Use environment variable or default to relative path
    movement_path = Path(os.environ.get('MOVEMENT_PATH', '../movement'))
    if not movement_path.is_absolute():
        # If relative, make it relative to the Development folder
        movement_path = Path(__file__).parent.parent.parent / movement_path
    
    if movement_path.exists() and str(movement_path) not in sys.path:
        sys.path.insert(0, str(movement_path))

    try:
        from movement.kinematics.kinematics import (
            compute_displacement,
            compute_speed
        )
    except ImportError:
        compute_displacement = None
        compute_speed = None

    try:
        from movement.kinematics.kinetic_energy import compute_kinetic_energy
    except ImportError:
        compute_kinetic_energy = None


class KinematicFeatureExtractor:
    """Extract behavioral kinematic features from IMU and positional data.

    Designed to work with both:
    - IMU data: (B, 9, 2, T) - accelerometer, gyroscope, magnetometer
    - Position data: (B, N_keypoints, 3, T) - 3D keypoint trajectories
    """

    def __init__(self,
                 sampling_rate: float = 100.0,
                 use_movement: bool = True):
        """Initialize kinematic feature extractor.

        Args:
            sampling_rate: Data sampling frequency in Hz
            use_movement: Whether to use Movement library functions
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.use_movement = use_movement and MOVEMENT_AVAILABLE

        if self.use_movement:
            self.movement_proc = MovementPreprocessor(sampling_rate=sampling_rate)

    def extract_imu_features(self, imu_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract kinematic features from IMU data.

        Args:
            imu_data: Tensor of shape (B, 9, 2, T) with IMU channels

        Returns:
            Dictionary of extracted features
        """
        B, C, S, T = imu_data.shape
        features = {}

        # Split IMU channels
        accel = imu_data[:, 0:3, :, :]  # Acceleration
        gyro = imu_data[:, 3:6, :, :]   # Angular velocity
        mag = imu_data[:, 6:9, :, :]    # Magnetometer

        # 1. Acceleration magnitude
        accel_mag = torch.sqrt(torch.sum(accel**2, dim=1))  # (B, 2, T)
        features['acceleration_magnitude'] = accel_mag

        # 2. Angular velocity magnitude
        gyro_mag = torch.sqrt(torch.sum(gyro**2, dim=1))  # (B, 2, T)
        features['angular_velocity_magnitude'] = gyro_mag

        # 3. Jerk (derivative of acceleration)
        jerk = torch.diff(accel, dim=-1)
        jerk = torch.cat([jerk, jerk[..., -1:]], dim=-1)  # Pad to maintain shape
        features['jerk'] = jerk

        # 4. Angular acceleration
        angular_accel = torch.diff(gyro, dim=-1)
        angular_accel = torch.cat([angular_accel, angular_accel[..., -1:]], dim=-1)
        features['angular_acceleration'] = angular_accel

        # 5. Orientation stability (using magnetometer)
        mag_stability = torch.std(mag, dim=-1, keepdim=True)
        features['orientation_stability'] = mag_stability

        # 6. Cross-sensor correlation (synchrony measure)
        if S == 2:
            # Correlation between two sensors
            sensor1 = imu_data[:, :, 0, :]  # (B, 9, T)
            sensor2 = imu_data[:, :, 1, :]  # (B, 9, T)

            # Compute rolling correlation
            corr = self._compute_rolling_correlation(sensor1, sensor2, window=10)
            features['sensor_synchrony'] = corr

        # 7. Frequency domain features
        freq_features = self._extract_frequency_features(imu_data)
        features.update(freq_features)

        # 8. Statistical moments
        features['mean'] = torch.mean(imu_data, dim=-1)
        features['std'] = torch.std(imu_data, dim=-1)
        features['skewness'] = self._compute_skewness(imu_data)
        features['kurtosis'] = self._compute_kurtosis(imu_data)

        return features

    def extract_position_features(self, position_data: torch.Tensor,
                                 masses: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract kinematic features from position/keypoint data.

        Args:
            position_data: Tensor of shape (B, N, 3, T) with 3D positions
            masses: Optional masses for kinetic energy calculation

        Returns:
            Dictionary of extracted features
        """
        B, N, D, T = position_data.shape
        features = {}

        # 1. Velocity (first derivative)
        velocity = torch.diff(position_data, dim=-1) * self.sampling_rate
        velocity = torch.cat([velocity, velocity[..., -1:]], dim=-1)
        features['velocity'] = velocity

        # 2. Speed (magnitude of velocity)
        speed = torch.sqrt(torch.sum(velocity**2, dim=2))  # (B, N, T)
        features['speed'] = speed

        # 3. Acceleration (second derivative)
        acceleration = torch.diff(velocity, dim=-1) * self.sampling_rate
        acceleration = torch.cat([acceleration, acceleration[..., -1:]], dim=-1)
        features['acceleration'] = acceleration

        # 4. Path length (cumulative distance)
        displacements = torch.sqrt(torch.sum(torch.diff(position_data, dim=-1)**2, dim=2))
        path_length = torch.cumsum(displacements, dim=-1)
        # Pad to maintain shape
        path_length = torch.cat([torch.zeros(B, N, 1, device=position_data.device),
                                 path_length], dim=-1)
        features['path_length'] = path_length

        # 5. Kinetic energy (if masses provided)
        if masses is not None:
            # KE = 0.5 * m * v^2
            kinetic_energy = 0.5 * masses.unsqueeze(-1) * torch.sum(velocity**2, dim=2)
            features['kinetic_energy'] = kinetic_energy

            # Total system kinetic energy
            total_ke = torch.sum(kinetic_energy, dim=1)
            features['total_kinetic_energy'] = total_ke

        # 6. Center of mass (if multiple keypoints)
        if N > 1:
            if masses is not None:
                # Weighted center of mass
                total_mass = torch.sum(masses, dim=1, keepdim=True)
                weights = masses.unsqueeze(-1).unsqueeze(-1) / total_mass.unsqueeze(-1).unsqueeze(-1)
                com = torch.sum(position_data * weights, dim=1)
            else:
                # Simple average
                com = torch.mean(position_data, dim=1)

            features['center_of_mass'] = com

            # COM velocity and acceleration
            com_velocity = torch.diff(com, dim=-1) * self.sampling_rate
            com_velocity = torch.cat([com_velocity, com_velocity[..., -1:]], dim=-1)
            features['com_velocity'] = com_velocity

            com_speed = torch.sqrt(torch.sum(com_velocity**2, dim=1))
            features['com_speed'] = com_speed

        # 7. Inter-keypoint distances (for multi-keypoint data)
        if N > 1:
            distances = self._compute_pairwise_distances(position_data)
            features['inter_keypoint_distances'] = distances

            # Distance variability (measure of shape changes)
            dist_std = torch.std(distances, dim=(1,2))
            features['shape_variability'] = dist_std

        # 8. Movement smoothness (spectral arc length)
        smoothness = self._compute_movement_smoothness(velocity)
        features['movement_smoothness'] = smoothness

        return features

    def extract_synchrony_features(self, data1: torch.Tensor,
                                  data2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract synchrony features between two data streams.

        Args:
            data1: First data stream (B, *, T)
            data2: Second data stream (B, *, T)

        Returns:
            Dictionary of synchrony features
        """
        features = {}

        # Ensure same batch size and time dimension
        assert data1.shape[0] == data2.shape[0], "Batch size mismatch"
        assert data1.shape[-1] == data2.shape[-1], "Time dimension mismatch"

        B, T = data1.shape[0], data1.shape[-1]

        # Flatten middle dimensions if needed
        if data1.dim() > 2:
            data1_flat = data1.reshape(B, -1, T)
            data2_flat = data2.reshape(B, -1, T)
        else:
            data1_flat = data1
            data2_flat = data2

        # 1. Cross-correlation
        cross_corr = self._compute_cross_correlation(data1_flat, data2_flat)
        features['cross_correlation'] = cross_corr

        # 2. Phase synchrony (using Hilbert transform)
        phase_sync = self._compute_phase_synchrony(data1_flat, data2_flat)
        features['phase_synchrony'] = phase_sync

        # 3. Dynamic time warping distance
        dtw_dist = self._compute_dtw_distance(data1_flat, data2_flat)
        features['dtw_distance'] = dtw_dist

        # 4. Mutual information
        mi = self._estimate_mutual_information(data1_flat, data2_flat)
        features['mutual_information'] = mi

        # 5. Coherence (frequency domain synchrony)
        coherence = self._compute_coherence(data1_flat, data2_flat)
        features['coherence'] = coherence

        return features

    def _compute_rolling_correlation(self, x: torch.Tensor, y: torch.Tensor,
                                    window: int = 10) -> torch.Tensor:
        """Compute rolling correlation between two signals."""
        B, C, T = x.shape

        # Pad for rolling window
        pad = window // 2
        x_pad = torch.nn.functional.pad(x, (pad, pad), mode='reflect')
        y_pad = torch.nn.functional.pad(y, (pad, pad), mode='reflect')

        correlations = []
        for t in range(T):
            x_window = x_pad[:, :, t:t+window]
            y_window = y_pad[:, :, t:t+window]

            # Compute correlation for each channel
            x_mean = x_window.mean(dim=-1, keepdim=True)
            y_mean = y_window.mean(dim=-1, keepdim=True)

            x_centered = x_window - x_mean
            y_centered = y_window - y_mean

            numerator = (x_centered * y_centered).sum(dim=-1)
            denominator = torch.sqrt((x_centered**2).sum(dim=-1) * (y_centered**2).sum(dim=-1))

            corr = numerator / (denominator + 1e-8)
            correlations.append(corr)

        return torch.stack(correlations, dim=-1)

    def _extract_frequency_features(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract frequency domain features using FFT."""
        features = {}

        # Compute FFT along time axis
        fft = torch.fft.rfft(data, dim=-1)
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)

        # Power spectral density
        psd = magnitude ** 2
        features['power_spectral_density'] = psd

        # Dominant frequency
        freqs = torch.fft.rfftfreq(data.shape[-1], d=self.dt)
        dominant_freq_idx = torch.argmax(psd, dim=-1)
        dominant_freq = freqs[dominant_freq_idx]
        features['dominant_frequency'] = dominant_freq

        # Spectral entropy
        psd_norm = psd / (psd.sum(dim=-1, keepdim=True) + 1e-8)
        spectral_entropy = -(psd_norm * torch.log(psd_norm + 1e-8)).sum(dim=-1)
        features['spectral_entropy'] = spectral_entropy

        # Mean frequency
        freq_weighted = (psd * freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        mean_freq = freq_weighted / (psd.sum(dim=-1) + 1e-8)
        features['mean_frequency'] = mean_freq

        return features

    def _compute_skewness(self, data: torch.Tensor) -> torch.Tensor:
        """Compute skewness of data along time axis."""
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True)
        skew = ((data - mean) ** 3).mean(dim=-1) / (std.squeeze(-1) ** 3 + 1e-8)
        return skew

    def _compute_kurtosis(self, data: torch.Tensor) -> torch.Tensor:
        """Compute kurtosis of data along time axis."""
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True)
        kurt = ((data - mean) ** 4).mean(dim=-1) / (std.squeeze(-1) ** 4 + 1e-8) - 3
        return kurt

    def _compute_pairwise_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances between keypoints."""
        B, N, D, T = positions.shape

        # Reshape for broadcasting
        pos1 = positions.unsqueeze(2)  # (B, N, 1, D, T)
        pos2 = positions.unsqueeze(1)  # (B, 1, N, D, T)

        # Compute distances
        distances = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=3))  # (B, N, N, T)

        return distances

    def _compute_movement_smoothness(self, velocity: torch.Tensor) -> torch.Tensor:
        """Compute spectral arc length as smoothness measure."""
        # Compute FFT of velocity
        fft = torch.fft.rfft(velocity, dim=-1)
        magnitude = torch.abs(fft)

        # Normalize magnitude spectrum
        magnitude_norm = magnitude / (magnitude.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute spectral arc length
        # SAL = -integral(sqrt(1 + (dV/dw)^2) dw)
        dV_dw = torch.diff(magnitude_norm, dim=-1)
        arc_length = -torch.sqrt(1 + dV_dw**2).sum(dim=-1)

        return arc_length

    def _compute_cross_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute normalized cross-correlation."""
        # Normalize signals
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        y_norm = (y - y.mean(dim=-1, keepdim=True)) / (y.std(dim=-1, keepdim=True) + 1e-8)

        # Compute correlation
        correlation = (x_norm * y_norm).mean(dim=-1)

        return correlation

    def _compute_phase_synchrony(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute phase synchrony using Hilbert transform."""
        # Apply Hilbert transform (approximation using FFT)
        def hilbert_phase(signal):
            fft = torch.fft.fft(signal, dim=-1)
            fft[..., signal.shape[-1]//2:] = 0
            fft[..., 1:signal.shape[-1]//2] *= 2
            analytic = torch.fft.ifft(fft, dim=-1)
            phase = torch.angle(analytic)
            return phase

        phase_x = hilbert_phase(x)
        phase_y = hilbert_phase(y)

        # Phase difference
        phase_diff = phase_x - phase_y

        # Phase locking value (PLV)
        plv = torch.abs(torch.exp(1j * phase_diff).mean(dim=-1))

        return plv

    def _compute_dtw_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Simplified DTW distance computation."""
        # Note: Full DTW is computationally expensive
        # Using a simplified version based on cumulative distance

        # Vectorized Euclidean distance computation
        # Compute squared differences for all batches at once
        diff_squared = (x - y) ** 2
        
        # Sum over channel dimension and take sqrt
        distances = torch.sqrt(torch.sum(diff_squared, dim=1))
        
        # Mean over time dimension
        return torch.mean(distances, dim=1)

    def _estimate_mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Estimate mutual information using binning."""
        # Simplified MI estimation using histogram binning
        n_bins = 10

        B = x.shape[0]
        mi_values = []

        for b in range(B):
            # Flatten and discretize
            x_flat = x[b].flatten()
            y_flat = y[b].flatten()

            # Create bins
            x_bins = torch.histc(x_flat, bins=n_bins)
            y_bins = torch.histc(y_flat, bins=n_bins)

            # Joint histogram (simplified)
            xy_bins = torch.histc(x_flat + y_flat, bins=n_bins)

            # Compute MI (simplified)
            px = x_bins / x_bins.sum()
            py = y_bins / y_bins.sum()
            pxy = xy_bins / xy_bins.sum()

            # MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
            mi = (pxy * torch.log(pxy / (px.mean() * py.mean() + 1e-8) + 1e-8)).sum()
            mi_values.append(mi)

        return torch.tensor(mi_values, device=x.device)

    def _compute_coherence(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute magnitude squared coherence."""
        # Cross-spectral density
        fft_x = torch.fft.rfft(x, dim=-1)
        fft_y = torch.fft.rfft(y, dim=-1)

        Pxy = fft_x * torch.conj(fft_y)
        Pxx = fft_x * torch.conj(fft_x)
        Pyy = fft_y * torch.conj(fft_y)

        # Coherence
        coherence = torch.abs(Pxy)**2 / (torch.abs(Pxx) * torch.abs(Pyy) + 1e-8)

        # Mean coherence across frequencies
        mean_coherence = coherence.mean(dim=-1)

        return mean_coherence


if __name__ == "__main__":
    print("Testing kinematic feature extraction...")

    # Test with IMU data
    B, C, S, T = 2, 9, 2, 100
    imu_data = torch.randn(B, C, S, T)

    extractor = KinematicFeatureExtractor(sampling_rate=100.0)

    # Extract IMU features
    imu_features = extractor.extract_imu_features(imu_data)
    print(f"\nIMU features extracted: {list(imu_features.keys())}")

    # Test with position data
    N_keypoints = 5
    position_data = torch.randn(B, N_keypoints, 3, T)
    masses = torch.ones(B, N_keypoints)

    # Extract position features
    pos_features = extractor.extract_position_features(position_data, masses)
    print(f"\nPosition features extracted: {list(pos_features.keys())}")

    # Test synchrony features
    data1 = torch.randn(B, 10, T)
    data2 = torch.randn(B, 10, T)

    sync_features = extractor.extract_synchrony_features(data1, data2)
    print(f"\nSynchrony features extracted: {list(sync_features.keys())}")

    print("\n✅ Kinematic feature extraction test complete!")