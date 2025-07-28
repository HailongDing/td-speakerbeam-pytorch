# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import pytorch_lightning as pl
import torch


class SystemInformed(pl.LightningModule):
    """PyTorch Lightning system for informed speech separation."""
    
    def __init__(self, model, loss_func, optimizer, train_loader, val_loader, 
                 scheduler=None, config=None):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        
    def forward(self, inputs, enrolls):
        """Forward pass."""
        return self.model(inputs, enrolls)
    
    def common_step(self, batch, batch_nb, train=True):
        """Common step for training and validation."""
        inputs, targets, enrolls = batch
        est_targets = self(inputs, enrolls)
        loss = self.loss_func(est_targets, targets)
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = self.common_step(batch, batch_idx, train=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = self.common_step(batch, batch_idx, train=False)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        if self.scheduler is not None:
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': {
                    'scheduler': self.scheduler,
                    'monitor': 'val_loss',
                    'frequency': 1
                }
            }
        return self.optimizer
    
    def train_dataloader(self):
        """Return training dataloader."""
        return self.train_loader
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return self.val_loader