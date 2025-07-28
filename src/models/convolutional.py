# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)."""
    
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))
        
    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (((y - mean) ** 2).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True))
        gLN_y = self.gamma * (y - mean) / (var + 1e-8).sqrt() + self.beta
        return gLN_y


class CumulativeLayerNorm(nn.Module):
    """Cumulative Layer Normalization (cLN)."""
    
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))
        
    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        batch_size, channel_size, length = y.shape
        cum_sum = torch.cumsum(y.sum(dim=1, keepdim=True), dim=2)  # [M, 1, K]
        cum_pow_sum = torch.cumsum((y ** 2).sum(dim=1, keepdim=True), dim=2)  # [M, 1, K]
        
        entry_cnt = torch.arange(channel_size, channel_size * (length + 1), channel_size, 
                                dtype=y.dtype, device=y.device).view(1, 1, length)
        cum_mean = cum_sum / entry_cnt  # [M, 1, K]
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean ** 2
        cum_std = (cum_var + 1e-8).sqrt()
        
        cLN_y = self.gamma * (y - cum_mean) / cum_std + self.beta
        return cLN_y


def select_norm(norm_type, channel_size):
    """Select normalization layer."""
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return CumulativeLayerNorm(channel_size)
    elif norm_type == "BN":
        return nn.BatchNorm1d(channel_size)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


class Conv1DBlock(nn.Module):
    """1D convolutional block with normalization and activation."""
    
    def __init__(self, in_chan, hid_chan, skip_chan, kernel_size, 
                 padding, dilation, norm_type="gLN", causal=False):
        super().__init__()
        self.skip_chan = skip_chan
        
        # 1x1 conv
        self.conv1x1 = nn.Conv1d(in_chan, hid_chan, 1)
        self.norm1 = select_norm(norm_type, hid_chan)
        
        # Depthwise conv
        if causal:
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = padding
        self.dconv = nn.Conv1d(hid_chan, hid_chan, kernel_size, 
                              padding=self.padding, dilation=dilation, groups=hid_chan)
        self.norm2 = select_norm(norm_type, hid_chan)
        
        # Output conv
        if skip_chan:
            self.sconv = nn.Conv1d(hid_chan, skip_chan, 1)
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        
        self.causal = causal
        
    def forward(self, x):
        """
        Args:
            x: [batch, in_chan, time]
        Returns:
            residual: [batch, in_chan, time]
            skip: [batch, skip_chan, time] (if skip_chan > 0)
        """
        # 1x1 conv + norm + PReLU
        y = F.prelu(self.norm1(self.conv1x1(x)), torch.tensor(0.25, device=x.device))
        
        # Depthwise conv + norm + PReLU
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.padding]
        y = F.prelu(self.norm2(y), torch.tensor(0.25, device=x.device))
        
        # Output convs
        if self.skip_chan:
            skip = self.sconv(y)
            residual = self.res_conv(y)
            return residual, skip
        else:
            residual = self.res_conv(y)
            return residual


class TDConvNet(nn.Module):
    """Temporal Convolutional Network (TCN)."""
    
    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3,
                 bn_chan=128, hid_chan=512, skip_chan=128, conv_kernel_size=3,
                 norm_type="gLN", mask_act="relu", causal=False):
        super().__init__()
        
        self.in_chan = in_chan
        self.n_src = n_src
        self.out_chan = out_chan if out_chan else in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(in_chan, bn_chan, 1)
        
        # TCN blocks
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                dilation = 2 ** x
                padding = (conv_kernel_size - 1) * dilation // 2
                self.TCN.append(
                    Conv1DBlock(bn_chan, hid_chan, skip_chan, conv_kernel_size,
                               padding, dilation, norm_type, causal)
                )
        
        # Mask generation
        if skip_chan:
            self.mask_net = nn.Conv1d(skip_chan, self.out_chan * n_src, 1)
        else:
            self.mask_net = nn.Conv1d(bn_chan, self.out_chan * n_src, 1)
            
        # Output activation
        if mask_act == "relu":
            self.output_act = nn.ReLU()
        elif mask_act == "sigmoid":
            self.output_act = nn.Sigmoid()
        elif mask_act == "tanh":
            self.output_act = nn.Tanh()
        elif mask_act == "linear":
            self.output_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {mask_act}")
    
    def forward(self, mixture_w):
        """
        Args:
            mixture_w: [batch, n_filters, n_frames]
        Returns:
            est_mask: [batch, n_src, n_filters, n_frames]
        """
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.zeros(1, device=mixture_w.device, dtype=mixture_w.dtype)
        
        for layer in self.TCN:
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
            
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask
    
    def get_config(self):
        """Get model configuration."""
        return {
            'in_chan': self.in_chan,
            'n_src': self.n_src,
            'out_chan': self.out_chan,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'skip_chan': self.skip_chan,
            'conv_kernel_size': self.conv_kernel_size,
            'norm_type': self.norm_type,
            'mask_act': self.mask_act,
            'causal': self.causal
        }