import os
import torch
import torch.nn as nn
from collections import OrderedDict


def get_all_subclasses(cls):
    subclasses = set()
    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(get_all_subclasses(subclass))
    return subclasses


def get_classes_from_module(module_startswith, parent_class):
    return {
        cls.__name__: cls
        for cls in get_all_subclasses(parent_class)
        if cls.__module__.startswith(module_startswith)
    }


def get_resize_transform(size):
    from torchvision import transforms as T

    return T.Compose([T.Resize((size, size)), T.ToTensor()])


def load_model(
    model_structure,
    model_name,
    checkpoints_dir="checkpoints",
    device="cpu",
    multi_gpu=False,
):
    state_dict = torch.load(
        os.path.join(checkpoints_dir, model_name + ".pth.tar"),
        weights_only=False,
        map_location=torch.device(device),
    )

    model = model_structure
    model = model.to(device)

    if multi_gpu:
        model = nn.DataParallel(model)
    else:
        state_dict = strip_data_parallel(state_dict)

    model.load_state_dict(state_dict)
    return model


def strip_data_parallel(model_state_dict):
    """
    Removes prefixes added when training model in nn.DataParallel to allow loading of models on single-device machines.

    Args:
        model_state_dict (Dict): dictionary of weights loaded by torch.
    """
    new_state_dict = OrderedDict()

    for k, v in model_state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    return new_state_dict
