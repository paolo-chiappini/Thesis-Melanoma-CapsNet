import torch.optim as optim
from utils.commons import get_classes_from_module, get_all_subclasses

from .trainer_base import BaseTrainer
from .trainer_capsnet_vas import CapsNetTrainerVAs


def get_trainer(
    config,
    model,
    data_loader,
    loss_criterion,
    optimizer=None,
    scheduler=None,
    device="cuda",
    checkpoints_dir="checkpoints",
):
    trainer_classes = get_classes_from_module(
        module_startswith="trainers", parent_class=BaseTrainer
    )

    name = config["name"]
    trainer_class = trainer_classes.get(name)
    if trainer_class is None:
        raise ValueError(f"Unknown trainer: {name}")

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    if scheduler is None:
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=config["lr_decay"]
        )

    return trainer_class(
        model,
        data_loader,
        loss_criterion,
        optimizer,
        scheduler,
        device,
        checkpoints_dir,
    )
