# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch
import torch.nn.functional as F


def si_sdr(est_target, target, eps=1e-8):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        est_target (torch.Tensor): Estimated target signal.
        target (torch.Tensor): Ground truth target signal.
        eps (float): Small epsilon for numerical stability.
        
    Returns:
        torch.Tensor: SI-SDR value.
    """
    # Zero-mean signals
    est_target = est_target - torch.mean(est_target, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # Compute scaling factor
    dot = torch.sum(est_target * target, dim=-1, keepdim=True)
    s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    scaled_target = dot * target / s_target_energy
    
    # Compute SI-SDR
    e_noise = est_target - scaled_target
    sisdr = 10 * torch.log10(
        (torch.sum(scaled_target ** 2, dim=-1) + eps) / 
        (torch.sum(e_noise ** 2, dim=-1) + eps)
    )
    
    return sisdr


def singlesrc_neg_sisdr(est_target, target):
    """Negative SI-SDR loss for single source.
    
    Args:
        est_target (torch.Tensor): Estimated target signal.
        target (torch.Tensor): Ground truth target signal.
        
    Returns:
        torch.Tensor: Negative SI-SDR loss.
    """
    return -si_sdr(est_target, target)