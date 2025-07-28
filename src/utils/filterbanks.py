# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """Base encoder class."""
    def __init__(self):
        super().__init__()
        
    @property
    def n_feats_out(self):
        """Number of output features."""
        raise NotImplementedError


class Decoder(nn.Module):
    """Base decoder class."""
    def __init__(self):
        super().__init__()


class FreeEncoder(Encoder):
    """Free (learnable) encoder with 1D convolution."""
    
    def __init__(self, kernel_size, n_filters, stride=None, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride if stride is not None else kernel_size // 2
        
        self.conv1d = nn.Conv1d(1, n_filters, kernel_size, stride=self.stride, bias=False)
        
    @property
    def n_feats_out(self):
        return self.n_filters
        
    def forward(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Input waveform of shape (batch, n_channels, time)
            
        Returns:
            torch.Tensor: Encoded features of shape (batch, n_filters, time_frames)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # Add channel dimension
        elif waveform.dim() == 3 and waveform.shape[1] > 1:
            # If multiple channels, take the first one
            waveform = waveform[:, :1, :]
            
        return F.relu(self.conv1d(waveform))


class FreeDecoder(Decoder):
    """Free (learnable) decoder with 1D transposed convolution."""
    
    def __init__(self, kernel_size, n_filters, stride=None, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride if stride is not None else kernel_size // 2
        
        self.conv1d_transpose = nn.ConvTranspose1d(
            n_filters, 1, kernel_size, stride=self.stride, bias=False
        )
        
    def forward(self, encoded):
        """
        Args:
            encoded (torch.Tensor): Encoded features of shape (batch, n_filters, time_frames)
            
        Returns:
            torch.Tensor: Decoded waveform of shape (batch, 1, time)
        """
        return self.conv1d_transpose(encoded)


def make_enc_dec(fb_name, kernel_size, n_filters, stride=None, sample_rate=8000, **fb_kwargs):
    """Create encoder and decoder pair.
    
    Args:
        fb_name (str): Filterbank name. Currently supports 'free'.
        kernel_size (int): Kernel size for convolution.
        n_filters (int): Number of filters.
        stride (int, optional): Stride for convolution.
        sample_rate (float): Sample rate (for compatibility).
        **fb_kwargs: Additional keyword arguments.
        
    Returns:
        tuple: (encoder, decoder) pair.
    """
    if fb_name == "free":
        encoder = FreeEncoder(kernel_size, n_filters, stride, **fb_kwargs)
        decoder = FreeDecoder(kernel_size, n_filters, stride, **fb_kwargs)
        return encoder, decoder
    else:
        raise ValueError(f"Unsupported filterbank: {fb_name}")