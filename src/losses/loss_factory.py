import torch.functional as F
import torch.nn as nn

from utils.commons import get_classes_from_module

from .losses import *
from .losses_ae import *
from .msr_loss import *
from .tc_loss import *


class CombinedLoss(nn.Module):
    def __init__(self, loss_configs, class_weights=None, attribute_weights=None):
        super(CombinedLoss, self).__init__()
        self.loss_modules = nn.ModuleDict()
        self.loss_coefficients = {}

        self.available_losses = get_classes_from_module(
            module_startswith="losses", parent_class=nn.Module
        )

        print(f"Available losses: {self.available_losses}")

        for loss_name, config in loss_configs.items():
            if config["lambda"] > 0:
                loss_class = self.available_losses.get(loss_name)
                if loss_class is None:
                    raise ValueError(f"Unknown loss type: {loss_name}")

                loss_params = config.get("params", {})
                self.loss_modules[loss_name] = loss_class(**loss_params)
                self.loss_coefficients[loss_name] = config["lambda"]
            else:
                print(f"Skipping {loss_name} as its coefficient is 0.")

    def forward(self, model_outputs: dict, targets: dict):
        calculated_losses = {}
        for loss_name, loss_module in self.loss_modules.items():
            current_loss = None

            if loss_name == "MarginLoss":
                current_loss = loss_module(
                    inputs=model_outputs.get("attribute_logits"),
                    labels=targets.get("visual_attributes_targets"),
                )
            if loss_name == "AttributeLoss":
                current_loss = loss_module(
                    attribute_scores=model_outputs.get("attribute_logits"),
                    attribute_targets=targets.get("visual_attributes_targets"),
                )
            elif loss_name == "MalignancyLoss":
                current_loss = loss_module(
                    malignancy_scores=model_outputs.get("malignancy_scores"),
                    malignancy_targets=F.one_hot(
                        targets.get("malignancy_targets").long(), num_classes=2
                    ).float(),
                )
            elif loss_name == "TotalCorrelationLoss":
                current_loss = loss_module(
                    z_activations=model_outputs.get("attribute_poses")
                )
            else:
                raise NotImplementedError(
                    f"Forward pass not implemented for {loss_name}"
                )

            if current_loss is not None:
                calculated_losses[loss_name] = (
                    current_loss * self.loss_coefficients[loss_name]
                )

        return calculated_losses
