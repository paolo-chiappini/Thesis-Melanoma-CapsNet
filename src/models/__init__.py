import inspect
import sys

import torch.nn as nn

from utils.commons import get_classes_from_module

from .autoencoder import ConvAutoencoder
from .model_base_capsnet import CapsuleNetworkBase
from .model_conv_attributes_32 import CapsuleNetworkWithAttributes32
from .model_pretrained import ModelPretrainedCapsnet
from .model_resnet_classifier import ResnetClassifier
from .model_sanity_check import SanityCheckModel
from .variants.model_capsnet_attr_logit_upconv import LogitCapsnet
from .variants.model_capsnet_attr_mlp import CapsNetWithAttributesMLP
from .variants.model_capsnet_attr_mlp_pre import CapsNetWithAttributesMLP_Pre
from .variants.model_capsnet_attr_mlp_upconv import CapsNetWithAttributesMLPUpconv
from .variants.model_capsnet_mlp_gated import CapsNetWithAttributesMLPGated
from .variants.model_multi_decoder import CapsNetMultiDecoder


def get_model(config, data_loader, device):
    first_key = list(data_loader.keys())[0]
    img_shape = data_loader[first_key].dataset[0]["images"].numpy().shape

    model_classes = get_classes_from_module(
        module_startswith="model", parent_class=nn.Module
    )
    name = config["name"]
    kwargs = {k: v for k, v in config.items() if k != "name"}
    model_class = model_classes.get(name)
    if model_class is None:
        raise ValueError(f"Unknown model: {name}")

    sig = inspect.signature(model_class.__init__)
    accepted_args = sig.parameters.keys()

    if "num_attributes" in accepted_args:
        kwargs["num_attributes"] = (
            data_loader[first_key].dataset[0]["visual_attributes_targets"].shape[0]
        )

    return model_class(img_shape=img_shape, device=device, **kwargs)
