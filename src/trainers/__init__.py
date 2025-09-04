import torch.optim as optim

from metrics import build_metrics
from utils.commons import get_classes_from_module

from .trainer_autoencoder import AutoEncoderTrainer
from .trainer_base import BaseTrainer
from .trainer_capsnet_vas import CapsNetTrainerVAs
from .trainer_msr import CapsNetTrainerMSR
from .trainer_sanity_check import SanityCheckTrainer
from .trainer_transfer import TransferTrainer


def get_trainer(
    config,
    model,
    data_loader,
    loss_criterion,
    optimizer=None,
    scheduler=None,
    device="cuda",
    checkpoints_dir="checkpoints",
    save_name="model",
):
    trainer_config = config["trainer"]
    metrics_config = config.get("metrics", None)

    trainer_classes = get_classes_from_module(
        module_startswith="trainers", parent_class=BaseTrainer
    )

    name = trainer_config["name"]
    trainer_class = trainer_classes.get(name)
    if trainer_class is None:
        raise ValueError(f"Unknown trainer: {name}")

    if optimizer is None:
        params_to_update = filter(
            lambda p: p.requires_grad, model.parameters()
        )  # optimize only on unfrozen params
        optimizer = optim.Adam(params_to_update, lr=trainer_config["learning_rate"])

    if scheduler is None:
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=trainer_config["lr_decay"]
        )

    first_key = list(data_loader.keys())[0]

    try:
        num_attributes = (
            data_loader[first_key].dataset[0]["visual_attributes_targets"].shape[0]
        )
    except:
        num_attributes = 0

    return trainer_class(
        model,
        data_loader,
        loss_criterion,
        optimizer,
        scheduler,
        device,
        checkpoints_dir,
        save_name=save_name,
        metrics=(
            build_metrics(
                metrics_config=metrics_config,
                num_attributes=num_attributes,
                device=device,
            )
            if metrics_config is not None
            else {}
        ),
    )
