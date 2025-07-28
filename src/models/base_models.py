# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch
import torch.nn as nn
from utils.torch_utils import pad_x_to_y, jitable_shape


def _shape_reconstructed(reconstructed, shape):
    """Reshape reconstructed tensor to original shape."""
    if len(shape) == 1:
        return reconstructed.squeeze(0).squeeze(0)
    elif len(shape) == 2:
        return reconstructed.squeeze(0)
    else:
        return reconstructed


def _unsqueeze_to_3d(tensor):
    """Unsqueeze tensor to 3D (batch, n_mix, time)."""
    if tensor.dim() == 1:
        return tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 2:
        return tensor.unsqueeze(1)
    else:
        return tensor


class BaseEncoderMaskerDecoder(nn.Module):
    """Base class for encoder-masker-decoder models."""
    
    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        
    def forward(self, wav):
        """Forward pass through encoder-masker-decoder."""
        # Remember shape for reconstruction
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)
        
        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)
        
        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)
    
    def forward_encoder(self, wav):
        """Forward through encoder."""
        tf_rep = self.encoder(wav)
        if self.encoder_activation is not None:
            if self.encoder_activation == "relu":
                tf_rep = torch.relu(tf_rep)
            elif self.encoder_activation == "sigmoid":
                tf_rep = torch.sigmoid(tf_rep)
            elif self.encoder_activation == "tanh":
                tf_rep = torch.tanh(tf_rep)
        return tf_rep
    
    def forward_masker(self, tf_rep):
        """Forward through masker."""
        return self.masker(tf_rep)
    
    def forward_decoder(self, masked_tf_rep):
        """Forward through decoder."""
        return self.decoder(masked_tf_rep)
    
    def apply_masks(self, tf_rep, est_masks):
        """Apply estimated masks to time-frequency representation."""
        # tf_rep: [batch, n_filters, n_frames]
        # est_masks: [batch, n_src, n_filters, n_frames] 
        # For single source extraction, we take the first source
        masked = tf_rep.unsqueeze(1) * est_masks  # [batch, n_src, n_filters, n_frames]
        return masked.squeeze(1)  # [batch, n_filters, n_frames]
    
    def serialize(self):
        """Serialize model for saving."""
        return {
            'model_name': self.__class__.__name__,
            'state_dict': self.state_dict(),
            'model_args': getattr(self, '_model_args', {}),
        }
    
    @classmethod
    def from_pretrained(cls, path):
        """Load pretrained model."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(**checkpoint.get('model_args', {}))
        model.load_state_dict(checkpoint['state_dict'])
        return model


class BaseEncoderMaskerDecoderInformed(BaseEncoderMaskerDecoder):
    """Base class for informed encoder-masker-decoder extraction models."""
    
    def __init__(self, encoder, masker, decoder, auxiliary, encoder_activation=None):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.auxiliary = auxiliary
    
    def forward(self, wav, enrollment):
        """Forward pass with enrollment information."""
        # Remember shape for reconstruction
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)
        
        # Real forward
        tf_rep = self.forward_encoder(wav)
        enroll_emb = self.auxiliary(enrollment)
        est_masks = self.forward_masker(tf_rep, enroll_emb)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)
        
        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)
    
    def forward_masker(self, tf_rep, enroll):
        """Forward through masker with enrollment."""
        return self.masker(tf_rep, enroll)