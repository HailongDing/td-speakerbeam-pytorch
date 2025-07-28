# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch
import torch.nn.functional as F


def pad_x_to_y(x, y, axis=-1):
    """Pad tensor x to have the same size as tensor y along specified axis.
    
    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Reference tensor.
        axis (int): Axis along which to pad.
        
    Returns:
        torch.Tensor: Padded tensor.
    """
    if axis != -1:
        raise NotImplementedError("Only axis=-1 is supported")
    
    inp_len = x.shape[axis]
    target_len = y.shape[axis]
    
    if inp_len >= target_len:
        return x[..., :target_len]
    else:
        pad_len = target_len - inp_len
        pad_tuple = (0, pad_len)
        return F.pad(x, pad_tuple)


def jitable_shape(tensor):
    """Get tensor shape in a torchscript-compatible way."""
    return tensor.shape


def tensors_to_device(tensor_list, device):
    """Move list of tensors to device."""
    return [tensor.to(device) for tensor in tensor_list]