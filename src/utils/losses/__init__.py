import inspect
import sys
import torch.nn as nn

from .losses import *


def get_loss_classes():
    current_module = sys.modules[__name__]

    loss_classes = {
        name: cls
        for name, cls in inspect.getmembers(current_module, inspect.isclass)
        if issubclass(cls, nn.Module)
        and cls.__module__ == current_module.__name__  # load only local classes
    }
    return loss_classes


def get_loss(config, class_weights):
    loss_classes = get_loss_classes()
    name = config["name"]
    kwargs = {k: v for k, v in config.items() if k != "name"}
    loss_class = loss_classes.get(name)()
    if loss_class is None:
        raise ValueError(f"Unknown loss: {name}")
    return loss_class(class_weights=class_weights, **kwargs)
