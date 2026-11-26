"""
Unit tests for Steganalysis Rich Model (SRM) High-Pass Frequency Filter Layer.
"""

import torch
import numpy as np
from src.srm_filter import SRMConv2d, get_srm_kernels


def test_srm_kernels_shape_and_sum():
    kernels = get_srm_kernels()
    assert kernels.shape == (3, 1, 5, 5)
    # Check that high-pass filters sum approximately to 0 (pass high frequencies, suppress DC)
    for i in range(3):
        k_sum = kernels[i, 0].sum().item()
        assert abs(k_sum) < 1e-4, f"Kernel {i} sum is {k_sum}, should be near 0"


def test_srm_conv2d_forward_pass():
    # Batch of 2 RGB images (2, 3, 64, 64)
    x = torch.rand(2, 3, 64, 64)
    srm = SRMConv2d(in_channels=3, out_mode="rgb_residual")
    
    out = srm(x)
    assert out.shape == (2, 3, 64, 64)
    assert not srm.srm_conv.weight.requires_grad  # SRM weights should be fixed


def test_srm_heatmap_generation():
    frame = torch.rand(3, 128, 128)
    srm = SRMConv2d(in_channels=3)
    heatmap = srm.get_frequency_heatmap(frame)
    assert heatmap.shape == (128, 128)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0
