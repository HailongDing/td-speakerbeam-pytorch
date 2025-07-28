# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch.optim as optim


def make_optimizer(parameters, optimizer="adam", lr=0.001, weight_decay=0.0, **kwargs):
    """Create optimizer.
    
    Args:
        parameters: Model parameters.
        optimizer (str): Optimizer name.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        **kwargs: Additional optimizer arguments.
        
    Returns:
        torch.optim.Optimizer: Optimizer instance.
    """
    if optimizer.lower() == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer.lower() == "sgd":
        return optim.SGD(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer.lower() == "rmsprop":
        return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")