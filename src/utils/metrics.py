# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import numpy as np
import torch
from scipy import signal
from scipy.stats import pearsonr


def si_sdr_numpy(est_target, target, eps=1e-8):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in numpy."""
    # Zero-mean signals
    est_target = est_target - np.mean(est_target)
    target = target - np.mean(target)
    
    # Compute scaling factor
    dot = np.sum(est_target * target)
    s_target_energy = np.sum(target ** 2) + eps
    scaled_target = dot * target / s_target_energy
    
    # Compute SI-SDR
    e_noise = est_target - scaled_target
    sisdr = 10 * np.log10(
        (np.sum(scaled_target ** 2) + eps) / 
        (np.sum(e_noise ** 2) + eps)
    )
    
    return sisdr


def sdr(est_target, target, eps=1e-8):
    """Signal-to-Distortion Ratio (SDR)."""
    # Zero-mean signals
    est_target = est_target - np.mean(est_target)
    target = target - np.mean(target)
    
    # Compute SDR
    num = np.sum(target ** 2)
    den = np.sum((est_target - target) ** 2) + eps
    return 10 * np.log10(num / den)


def sir(est_target, target, interference, eps=1e-8):
    """Signal-to-Interference Ratio (SIR)."""
    # Project interference onto target
    dot = np.sum(target * interference)
    target_energy = np.sum(target ** 2) + eps
    proj_interference = dot * target / target_energy
    
    # Compute SIR
    num = np.sum(target ** 2)
    den = np.sum(proj_interference ** 2) + eps
    return 10 * np.log10(num / den)


def sar(est_target, target, eps=1e-8):
    """Signal-to-Artifacts Ratio (SAR)."""
    # Zero-mean signals
    est_target = est_target - np.mean(est_target)
    target = target - np.mean(target)
    
    # Compute scaling factor
    dot = np.sum(est_target * target)
    target_energy = np.sum(target ** 2) + eps
    scaled_target = dot * target / target_energy
    
    # Compute SAR
    artifacts = est_target - scaled_target
    num = np.sum(scaled_target ** 2)
    den = np.sum(artifacts ** 2) + eps
    return 10 * np.log10(num / den)


def stoi(est_target, target, sample_rate=8000):
    """Short-Time Objective Intelligibility (STOI)."""
    try:
        from pystoi import stoi as pystoi_stoi
        return pystoi_stoi(target, est_target, sample_rate, extended=False)
    except ImportError:
        # Fallback to correlation-based approximation
        return pearsonr(target, est_target)[0]


def get_metrics(mix, target, est_target, sample_rate=8000, metrics_list=None):
    """Compute multiple metrics.
    
    Args:
        mix (np.ndarray): Mixture signal.
        target (np.ndarray): Target signal.
        est_target (np.ndarray): Estimated target signal.
        sample_rate (int): Sample rate.
        metrics_list (list): List of metrics to compute.
        
    Returns:
        dict: Dictionary of computed metrics.
    """
    if metrics_list is None:
        metrics_list = ["si_sdr", "sdr", "sir", "sar", "stoi"]
    
    # Ensure single-channel signals
    if mix.ndim > 1:
        mix = mix[0]
    if target.ndim > 1:
        target = target[0]
    if est_target.ndim > 1:
        est_target = est_target[0]
    
    results = {}
    
    # Input metrics (mixture vs target)
    for metric in metrics_list:
        if metric == "si_sdr":
            results[f"input_{metric}"] = si_sdr_numpy(mix, target)
        elif metric == "sdr":
            results[f"input_{metric}"] = sdr(mix, target)
        elif metric == "sir":
            # For input SIR, use mixture as interference
            results[f"input_{metric}"] = sir(mix, target, mix - target)
        elif metric == "sar":
            results[f"input_{metric}"] = sar(mix, target)
        elif metric == "stoi":
            results[f"input_{metric}"] = stoi(mix, target, sample_rate)
    
    # Output metrics (estimated vs target)
    for metric in metrics_list:
        if metric == "si_sdr":
            results[metric] = si_sdr_numpy(est_target, target)
        elif metric == "sdr":
            results[metric] = sdr(est_target, target)
        elif metric == "sir":
            # For output SIR, use estimation error as interference
            results[metric] = sir(est_target, target, est_target - target)
        elif metric == "sar":
            results[metric] = sar(est_target, target)
        elif metric == "stoi":
            results[metric] = stoi(est_target, target, sample_rate)
    
    return results


def normalize_estimates(est_target, mix):
    """Normalize estimated target to have same energy as mixture."""
    if est_target.ndim > 1:
        est_target = est_target[0:1]  # Keep first dimension
    if mix.ndim > 1:
        mix = mix[0:1]
        
    # Compute energy ratio
    est_energy = np.sum(est_target ** 2, axis=-1, keepdims=True)
    mix_energy = np.sum(mix ** 2, axis=-1, keepdims=True)
    
    # Avoid division by zero
    est_energy = np.maximum(est_energy, 1e-8)
    
    # Normalize
    scale = np.sqrt(mix_energy / est_energy)
    return est_target * scale