"""
srm_filter.py - Steganalysis Rich Model (SRM) High-Pass Frequency Filter Layer
Implements SRM high-pass convolutional filters for extracting residual noise,
compression boundary artifacts, and neural blending anomalies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def get_srm_kernels() -> torch.Tensor:
    """
    Returns the 3 canonical Steganalysis Rich Model (SRM) high-pass forensic kernels:
    1. 1st-order horizontal/vertical residual kernel (KB / edge derivative)
    2. 2nd-order square / cross residual kernel (KV / cross derivative)
    3. Laplacian high-pass residual kernel (Laplace / point noise)
    
    Returns:
        torch.Tensor of shape (3, 1, 5, 5) with normalized high-pass coefficients.
    """
    # Filter 1: 1st-order edge residual (normalized)
    k1 = np.array([
        [0,  0,  0,  0,  0],
        [0, -1,  2, -1,  0],
        [0,  2, -4,  2,  0],
        [0, -1,  2, -1,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 4.0

    # Filter 2: Square / 2nd-order boundary residual (normalized)
    k2 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8,-12,  8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ], dtype=np.float32) / 12.0

    # Filter 3: 3x3 Laplacian in 5x5 container (high-frequency noise)
    k3 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  1,  0,  0],
        [0,  1, -4,  1,  0],
        [0,  0,  1,  0,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 4.0

    kernels = np.stack([k1, k2, k3], axis=0)  # Shape: (3, 5, 5)
    kernels = np.expand_dims(kernels, axis=1)  # Shape: (3, 1, 5, 5)
    return torch.tensor(kernels, dtype=torch.float32)


class SRMConv2d(nn.Module):
    """
    Steganalysis Rich Model (SRM) Convolutional Layer.
    Extracts high-pass spatial-frequency residual patterns from RGB images.
    
    Input:
        x: Tensor of shape (B, 3, H, W) normalized to [0, 1] or [-1, 1].
    Output:
        residuals: Tensor of shape (B, 3, H, W) if fused, or (B, 9, H, W) if per-channel.
    """
    def __init__(self, in_channels: int = 3, out_mode: str = "rgb_residual", threshold: float = 3.0):
        """
        Args:
            in_channels: Number of input channels (typically 3 for RGB).
            out_mode:
                - 'rgb_residual': 3 output channels (fused across the 3 SRM filters).
                - 'stacked_residual': 9 output channels (3 filters x 3 channels).
                - 'concat_with_rgb': 6 output channels (3 RGB + 3 fused residuals).
            threshold: Truncation threshold to clip strong image edges and preserve noise.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_mode = out_mode
        self.threshold = threshold

        kernels = get_srm_kernels()  # (3, 1, 5, 5)
        
        # Apply 3 filters independently across RGB channels: depthwise group conv
        # Weight shape: (3 * in_channels, 1, 5, 5)
        expanded_weights = kernels.repeat(in_channels, 1, 1, 1)  # (9, 1, 5, 5)
        
        self.srm_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=3 * in_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=in_channels,
            bias=False
        )
        self.srm_conv.weight.data = expanded_weights
        self.srm_conv.weight.requires_grad = False  # Fixed forensic coefficients

        # Projection back to 3 channels if 'rgb_residual' or 'concat_with_rgb'
        self.fuse_proj = nn.Conv2d(3 * in_channels, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.fuse_proj.weight, 1.0 / 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying SRM residual extraction.
        """
        # High pass convolution
        residuals = self.srm_conv(x)  # (B, 9, H, W)
        
        # Non-linear truncation to suppress macro edges and preserve subtle stego residuals
        if self.threshold > 0:
            residuals = torch.clamp(residuals, -self.threshold, self.threshold)

        if self.out_mode == "stacked_residual":
            return residuals
        
        fused = self.fuse_proj(residuals)  # (B, 3, H, W)
        
        if self.out_mode == "concat_with_rgb":
            return torch.cat([x, fused], dim=1)  # (B, 6, H, W)
            
        return fused  # (B, 3, H, W)

    def get_frequency_heatmap(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a 2D forensic residual heatmap for explainability & dashboard.
        Args:
            frame_tensor: (1, 3, H, W) or (3, H, W)
        Returns:
            heatmap_norm: 2D numpy array [0, 1] of frequency residuals.
        """
        self.eval()
        with torch.no_grad():
            if frame_tensor.ndim == 3:
                frame_tensor = frame_tensor.unsqueeze(0)
            res = self.forward(frame_tensor)  # (1, 3, H, W)
            res_mag = torch.mean(torch.abs(res), dim=1).squeeze(0).cpu().numpy()
            
            # Normalize to [0, 1]
            p_min, p_max = res_mag.min(), res_mag.max()
            if p_max - p_min > 1e-7:
                heatmap_norm = (res_mag - p_min) / (p_max - p_min)
            else:
                heatmap_norm = np.zeros_like(res_mag)
            return heatmap_norm
