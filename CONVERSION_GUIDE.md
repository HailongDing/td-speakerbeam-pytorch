# SpeakerBeam Asteroid-to-PyTorch Conversion Guide

This document provides a comprehensive guide for converting the original Asteroid-based SpeakerBeam implementation to a pure PyTorch implementation without external dependencies.

## Overview

The conversion process involves replacing Asteroid components with equivalent PyTorch implementations while maintaining the same functionality and API compatibility.

## 1. Project Structure Setup

### 1.1 Create New Project Directory
```bash
mkdir td-speakerbeam-pytorch
cd td-speakerbeam-pytorch
```

### 1.2 Directory Structure
```
td-speakerbeam-pytorch/
├── README.md
├── INSTALLATION.md
├── PROJECT_STRUCTURE.md
├── CONVERSION_GUIDE.md
├── requirements.txt
├── path.sh
├── test_installation.py
├── src/
│   ├── models/
│   ├── datasets/
│   └── utils/
├── egs/libri2mix/
├── example/
└── notebooks/
```

## 2. Dependencies Replacement

### 2.1 Update requirements.txt
Replace Asteroid with pure PyTorch dependencies:
```
torch>=1.8.0
torchaudio>=0.8.0
pytorch-lightning>=1.5.0
soundfile
librosa
pandas
pyyaml
tqdm
scipy
numpy
matplotlib
jupyter
tensorboard
```

### 2.2 Remove Asteroid Imports
Original imports to replace:
```python
# Remove these
from asteroid.engine import System
from asteroid_filterbanks import make_enc_dec
from asteroid.masknn.convolutional import TDConvNet
from asteroid.losses import singlesrc_neg_sisdr
from asteroid.metrics import get_metrics
from asteroid.utils.torch_utils import pad_x_to_y, tensors_to_device
from asteroid.engine.optimizers import make_optimizer
from asteroid.data import LibriMix
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
```

## 3. Core Component Implementations

### 3.1 Utility Functions (`src/utils/`)

#### 3.1.1 torch_utils.py
Replace `asteroid.utils.torch_utils`:
```python
import torch
import torch.nn.functional as F

def pad_x_to_y(x, y, axis=-1):
    """Pad tensor x to have the same size as tensor y along specified axis."""
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
```

#### 3.1.2 filterbanks.py
Replace `asteroid_filterbanks`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FreeEncoder(nn.Module):
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
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        elif waveform.dim() == 3 and waveform.shape[1] > 1:
            waveform = waveform[:, :1, :]
        return F.relu(self.conv1d(waveform))

class FreeDecoder(nn.Module):
    def __init__(self, kernel_size, n_filters, stride=None, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride if stride is not None else kernel_size // 2
        self.conv1d_transpose = nn.ConvTranspose1d(
            n_filters, 1, kernel_size, stride=self.stride, bias=False
        )
        
    def forward(self, encoded):
        return self.conv1d_transpose(encoded)

def make_enc_dec(fb_name, kernel_size, n_filters, stride=None, sample_rate=8000, **fb_kwargs):
    if fb_name == "free":
        encoder = FreeEncoder(kernel_size, n_filters, stride, **fb_kwargs)
        decoder = FreeDecoder(kernel_size, n_filters, stride, **fb_kwargs)
        return encoder, decoder
    else:
        raise ValueError(f"Unsupported filterbank: {fb_name}")
```

#### 3.1.3 losses.py
Replace `asteroid.losses`:
```python
import torch

def si_sdr(est_target, target, eps=1e-8):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)."""
    est_target = est_target - torch.mean(est_target, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    dot = torch.sum(est_target * target, dim=-1, keepdim=True)
    s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    scaled_target = dot * target / s_target_energy
    
    e_noise = est_target - scaled_target
    sisdr = 10 * torch.log10(
        (torch.sum(scaled_target ** 2, dim=-1) + eps) / 
        (torch.sum(e_noise ** 2, dim=-1) + eps)
    )
    return sisdr

def singlesrc_neg_sisdr(est_target, target):
    """Negative SI-SDR loss for single source."""
    return -si_sdr(est_target, target)
```

#### 3.1.4 optimizers.py
Replace `asteroid.engine.optimizers`:
```python
import torch.optim as optim

def make_optimizer(parameters, optimizer="adam", lr=0.001, weight_decay=0.0, **kwargs):
    if optimizer.lower() == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer.lower() == "sgd":
        return optim.SGD(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer.lower() == "rmsprop":
        return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
```

### 3.2 Model Components (`src/models/`)

#### 3.2.1 convolutional.py
Replace `asteroid.masknn.convolutional.TDConvNet`:

**Key Implementation Points:**
- Implement normalization layers (GlobalLayerNorm, CumulativeLayerNorm)
- Implement Conv1DBlock with proper device handling
- **CRITICAL: Causal Convolution Support**
- **Critical Fix**: Use `device=x.device` for all tensor creations

**Causal Convolution Implementation:**
```python
class Conv1DBlock(nn.Module):
    def __init__(self, in_chan, hid_chan, skip_chan, kernel_size, 
                 padding, dilation, norm_type="gLN", causal=False):
        super().__init__()
        self.skip_chan = skip_chan
        self.causal = causal
        
        # 1x1 conv
        self.conv1x1 = nn.Conv1d(in_chan, hid_chan, 1)
        self.norm1 = select_norm(norm_type, hid_chan)
        
        # Depthwise conv with causal padding
        if causal:
            # For causal convolution: pad only on the left (past)
            self.padding = (kernel_size - 1) * dilation
        else:
            # For non-causal: symmetric padding
            self.padding = padding
        
        self.dconv = nn.Conv1d(hid_chan, hid_chan, kernel_size, 
                              padding=self.padding, dilation=dilation, groups=hid_chan)
        
    def forward(self, x):
        # 1x1 conv + norm + PReLU
        y = F.prelu(self.norm1(self.conv1x1(x)), torch.tensor(0.25, device=x.device))
        
        # Depthwise conv + norm + PReLU
        y = self.dconv(y)
        
        # CRITICAL: For causal convolution, remove future padding
        if self.causal:
            y = y[:, :, :-self.padding]  # Remove right padding (future)
            
        y = F.prelu(self.norm2(y), torch.tensor(0.25, device=x.device))
        # ... rest of implementation
```

**Causal Parameter Flow:**
1. **TimeDomainSpeakerBeam**: Accepts `causal=False` (default)
2. **TDConvNetInformed**: Passes causal to parent TDConvNet
3. **TDConvNet**: Passes causal to each Conv1DBlock
4. **Conv1DBlock**: Implements causal padding logic
5. **Auxiliary Network**: Always uses `causal=False` (non-causal for enrollment)

#### 3.2.2 base_models.py
Replace `asteroid.models.base_models`:

**Key Implementation Points:**
- Implement BaseEncoderMaskerDecoder and BaseEncoderMaskerDecoderInformed
- **Critical Fix**: Proper tensor dimension handling in apply_masks:
```python
def apply_masks(self, tf_rep, est_masks):
    """Apply estimated masks to time-frequency representation."""
    # tf_rep: [batch, n_filters, n_frames]
    # est_masks: [batch, n_src, n_filters, n_frames] 
    masked = tf_rep.unsqueeze(1) * est_masks  # [batch, n_src, n_filters, n_frames]
    return masked.squeeze(1)  # [batch, n_filters, n_frames] - CRITICAL FIX
```

#### 3.2.3 system.py
Replace `asteroid.engine.System`:
```python
import pytorch_lightning as pl

class SystemInformed(pl.LightningModule):
    def __init__(self, model, loss_func, optimizer, train_loader, val_loader, 
                 scheduler=None, config=None):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        # ... implementation
    
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, enrolls = batch
        est_targets = self(inputs, enrolls)
        loss = self.loss_func(est_targets, targets)
        return loss
```

### 3.3 Dataset Components (`src/datasets/`)

#### 3.3.1 librimix.py and librimix_informed.py
Replace `asteroid.data.LibriMix`:
- Implement custom dataset classes
- Handle CSV reading and audio loading
- Maintain same API as original

## 4. Training Script Updates

### 4.1 PyTorch Lightning API Updates
**Critical Fix**: Update deprecated parameters:
```python
# OLD (deprecated)
trainer = pl.Trainer(
    gpus=-1,
    distributed_backend="ddp"
)

# NEW (current API)
if torch.cuda.is_available():
    accelerator = "gpu"
    devices = "auto"
    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
else:
    accelerator = "cpu"
    devices = "auto"
    strategy = "auto"

trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
    strategy=strategy
)
```

### 4.2 Argument Parsing
Replace `asteroid.utils.prepare_parser_from_dict`:
- Implement custom argument parsing utilities
- Handle hierarchical configuration dictionaries

## 5. Causal Convolution Feature Details

### 5.1 Causal vs Non-Causal Behavior

**Understanding Causal Convolution:**
- **Causal (causal=True)**: Only uses past and present information, no future context
- **Non-Causal (causal=False)**: Uses past, present, and future context (default for SpeakerBeam)

**Implementation Details:**
```python
# In Conv1DBlock.__init__():
if causal:
    # Left-padding only: (kernel_size - 1) * dilation
    self.padding = (kernel_size - 1) * dilation
else:
    # Symmetric padding: (kernel_size - 1) * dilation // 2
    self.padding = padding

# In Conv1DBlock.forward():
y = self.dconv(y)  # Apply convolution with padding
if self.causal:
    # Remove future context by trimming right side
    y = y[:, :, :-self.padding]
```

**Key Points:**
1. **Main Network**: Uses causal parameter from model configuration
2. **Auxiliary Network**: Always non-causal (`causal=False`) for enrollment processing
3. **Default Setting**: `causal=False` (non-causal) for better performance in offline scenarios
4. **Real-time Usage**: Set `causal=True` for streaming/real-time applications

### 5.2 Configuration Usage
```yaml
# In conf.yml (optional, defaults to False)
masknet:
  causal: false  # or true for real-time applications
```

```python
# In model instantiation
model = TimeDomainSpeakerBeam(
    causal=False,  # Non-causal for offline processing
    # ... other parameters
)
```

## 6. Critical Fixes and Common Issues

### 6.1 Device Compatibility Issues
**Problem**: `RuntimeError: Expected all tensors to be on the same device`

**Solutions**:
1. **PReLU tensors**: Always specify device
```python
F.prelu(x, torch.tensor(0.25, device=x.device))
```

2. **Skip connections**: Use proper device and dtype
```python
skip_connection = torch.zeros(1, device=mixture_w.device, dtype=mixture_w.dtype)
```

### 5.2 Tensor Dimension Issues
**Problem**: `Expected 2D or 3D input to conv_transpose1d, but got 4D`

**Solution**: Fix apply_masks method to squeeze source dimension:
```python
def apply_masks(self, tf_rep, est_masks):
    masked = tf_rep.unsqueeze(1) * est_masks
    return masked.squeeze(1)  # Remove source dimension
```

### 5.3 PyTorch Lightning API Changes
**Problem**: `TypeError: Trainer.__init__() got an unexpected keyword argument 'gpus'`

**Solution**: Use current API parameters (accelerator, devices, strategy)

## 6. Essential Files to Copy

### 6.1 Data Preparation Files
**Critical**: Copy these files from original project:
```
egs/libri2mix/data/wav8k/min/test/map_mixture2enrollment
egs/libri2mix/data/wav8k/min/dev/map_mixture2enrollment
```

### 6.2 Configuration Files
- `local/conf.yml`
- Data preparation scripts
- Path environment setup

## 7. Testing and Validation

### 7.1 Create Installation Test
```python
def test_imports():
    # Test all module imports
    
def test_model_creation():
    # Test model instantiation
    
def test_forward_pass():
    # Test model forward pass
    
def test_loss_computation():
    # Test loss functions
```

### 7.2 Validation Steps
1. Run installation test
2. Test data preparation
3. Verify model training starts
4. Check GPU compatibility
5. Validate output formats

## 8. Performance Optimizations

### 8.1 GPU Utilization
- Set `torch.set_float32_matmul_precision('medium')` for Tensor Cores
- Use appropriate batch sizes
- Enable mixed precision if needed

### 8.2 Memory Management
- Implement gradient checkpointing if needed
- Optimize data loading with proper num_workers
- Use appropriate segment lengths

## 9. Deployment Considerations

### 9.1 Environment Setup
- Update path.sh with correct repository path
- Make scripts executable (`chmod +x`)
- Set proper PYTHONPATH

### 9.2 Remote Server Setup
- Configure for specific server paths
- Handle SSH deployment
- Test on target hardware

## 10. Maintenance and Updates

### 10.1 Version Compatibility
- Keep PyTorch Lightning API updated
- Monitor PyTorch version compatibility
- Update dependencies as needed

### 10.2 Documentation
- Maintain comprehensive README
- Update installation guides
- Document known issues and solutions

## Summary

This conversion process successfully removes the Asteroid dependency while maintaining full functionality. The key challenges are:

1. **Device compatibility**: Ensure all tensors are on the same device
2. **Tensor dimensions**: Handle masking and decoding properly
3. **API updates**: Use current PyTorch Lightning syntax
4. **File dependencies**: Copy all necessary data files

Following this guide ensures a successful conversion that maintains the original model's performance and capabilities while being completely independent of the Asteroid toolkit.