import inspect
import torch.nn as nn

from .model_conv_attributes_32 import CapsuleNetworkWithAttributes32


def get_model_classes():
    def get_all_subclasses(cls):
        subclasses = set()
        for subclass in cls.__subclasses__():
            subclasses.add(subclass)
            subclasses.update(get_all_subclasses(subclass))
        return subclasses

    return {
        cls.__name__: cls
        for cls in get_all_subclasses(nn.Module)
        if cls.__module__.starstwith("models")
    }


def get_model(config, data_loader, device):
    img_shape = data_loader.dataset[0][0].numpy().shape()

    model_classes = get_model_classes()
    name = config["name"]
    kwargs = {k: v for k, v in config.items() if k != "name"}
    model_class = model_classes.get(name)()
    if model_class is None:
        raise ValueError(f"Unknown model: {name}")

    sig = inspect.signature(model_class.__init__)
    accepted_args = sig.parameters.keys()

    if "num_attributes" in accepted_args:
        kwargs["num_attributes"] = data_loader.dataset[0][2].shape[0]

    return model_class(img_shape=img_shape, device=device, **kwargs)
