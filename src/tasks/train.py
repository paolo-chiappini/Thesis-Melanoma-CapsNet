from utils.loaders import get_dataset
from models import get_model
from trainers import get_trainer
from utils.losses import get_loss
from utils.callbacks import get_callbacks, CallbackManager
from utils.commons import (
    compute_class_weights,
    compute_binary_feature_weights,
    build_dataloaders,
    get_transforms,
)
from config.device_config import get_device

import torch
import torch.nn as nn
from collections import Counter


def run_training(config, model_path=None, cpu_override=False):
    dataset_config = config["dataset"]
    model_config = config["model"]
    trainer_config = config["trainer"]
    callback_config = config.get("callbacks", [])
    system_config = config["system"]

    has_visual_attributes = system_config.get("has_visual_attributes", False)
    use_weighted_metrics = system_config.get("use_weighted_metrics", False)

    weights = {}

    device, multi_gpu = get_device(cpu_override=cpu_override)

    transform = get_transforms(
        config, is_train=dataset_config["augment"]
    )  # is_train = False is a standard resize + normalization
    dataset = get_dataset(dataset_config, transform=transform)

    class_counts = Counter(dataset.labels)
    class_weights = compute_class_weights(class_counts, device)
    print(f"Class counts: {class_counts}, Class weights: {class_weights}")

    weights["class_weights"] = class_weights

    if has_visual_attributes:
        attribute_counts_ones = torch.sum(dataset.visual_features, dim=0)
        attribute_weights = compute_binary_feature_weights(
            attribute_counts_ones, len(dataset), device
        )

        named_attribute_weights = dict(
            zip(dataset.visual_attributes, attribute_weights.cpu().numpy())
        )
        print(f"Attribute weights: {named_attribute_weights}")

        weights["attribute_weights"] = attribute_weights

    num_workers = 0 if not multi_gpu else 2

    batch_size = dataset_config["batch_size"]
    if device.type == "cuda" and multi_gpu:
        batch_size *= torch.cuda.device_count()

    loaders = build_dataloaders(
        config=config, dataset=dataset, batch_size=batch_size, num_workers=num_workers
    )

    model = get_model(model_config, data_loader=loaders, device=device)
    model = model.to(device)
    if multi_gpu:
        model = nn.DataParallel(model)

    loss_criterion = get_loss(trainer_config["loss"], class_weights)

    trainer = get_trainer(
        trainer_config,
        model,
        loaders,
        loss_criterion,
        device=device,
        checkpoints_dir=system_config["save_path"],
        save_name=system_config["save_name"],
    )

    if use_weighted_metrics:
        trainer.set_weights(weights_dict=weights)

    callbacks = get_callbacks(callback_config)
    callback_manager = CallbackManager(callbacks=callbacks)

    trainer.run(
        trainer_config["epochs"],
        callback_manager=callback_manager,
    )
    print("=" * 10, "Run finished", "=" * 10)

    # compute final metrics on validation set
    trainer.test(split="val")
