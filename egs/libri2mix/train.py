# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import os
import argparse
import json
import yaml
from pprint import pprint

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from models.td_speakerbeam import TimeDomainSpeakerBeam
from datasets.librimix_informed import LibriMixInformed
from utils.optimizers import make_optimizer
from models.system import SystemInformed
from utils.losses import singlesrc_neg_sisdr
from utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

# Keys which are not in the conf.yml file can be added here.
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


def neg_sisdr_loss_wrapper(est_targets, targets):
    """Wrapper for negative SI-SDR loss."""
    return singlesrc_neg_sisdr(est_targets[:, 0], targets[:, 0]).mean()


def main(conf):
    """Main training function."""
    train_set = LibriMixInformed(
        csv_dir=conf["data"]["train_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        segment_aux=conf["data"]["segment_aux"],
    )

    val_set = LibriMixInformed(
        csv_dir=conf["data"]["valid_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        segment_aux=conf["data"]["segment_aux"],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    model = TimeDomainSpeakerBeam(
        **conf["filterbank"], 
        **conf["masknet"], 
        sample_rate=conf["data"]["sample_rate"],
        **conf["enroll"]
    )
    
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, 
            factor=0.5, 
            patience=conf['training']['reduce_patience']
        )
    
    # Save configuration
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function
    loss_func = neg_sisdr_loss_wrapper
    system = SystemInformed(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, 
        monitor="val_loss", 
        mode="min", 
        save_top_k=-1, 
        verbose=True
    )
    callbacks.append(checkpoint)
    
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(
            monitor="val_loss", 
            mode="min", 
            patience=conf['training']['stop_patience'], 
            verbose=True
        ))
    callbacks.append(LearningRateMonitor())

    # Configure device and strategy for current PyTorch Lightning
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
        strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    else:
        accelerator = "cpu"
        devices = "auto"
        strategy = "auto"

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    # Load configuration
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    
    # Parse arguments
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)