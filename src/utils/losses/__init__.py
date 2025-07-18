import torch.nn as nn
from utils.commons import get_classes_from_module

from .losses import *
from .losses_ae import *


def get_loss(config, class_weights):
    loss_classes = get_classes_from_module(
        module_startswith="utils.losses", parent_class=nn.Module
    )

    name = config["name"]
    kwargs = {k: v for k, v in config.items() if k != "name"}
    loss_class = loss_classes.get(name)
    if loss_class is None:
        raise ValueError(f"Unknown loss: {name}")
    return loss_class(class_weights=class_weights, **kwargs)
