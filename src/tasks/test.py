from utils.loaders import get_dataset
from models import get_model
from trainers import get_trainer
from utils.losses import get_loss
from utils.callbacks import get_callbacks, CallbackManager
from utils.commons import get_resize_transform
from config.device_config import get_device

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import Counter
import numpy as np
import os


def compute_class_weights(class_counts, device):
    counts = np.array(list(class_counts.values()), dtype=np.float32)
    weights = 1.0 / counts
    weights /= weights.sum()
    return torch.tensor(weights).to(device)


def stratified_split(labels, val_size=0.1, test_size=0.1, seed=123):
    from sklearn.model_selection import StratifiedShuffleSplit

    indices = np.arange(len(labels))
    labels = np.array(labels)

    # split holdout set
    test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(test.split(indices, labels))

    # split train and validation
    validation = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size / (1 - test_size), random_state=seed
    )
    train_idx, val_idx = next(validation.split(train_val_idx, labels[train_val_idx]))

    return train_val_idx[train_idx], train_val_idx[val_idx], test_idx


def run_testing(config, model_path=None, cpu_override=False):
    dataset_config = config["dataset"]
    preprocess_config = config["preprocess"]
    model_config = config["model"]
    trainer_config = config["trainer"]
    system_config = config["system"]

    device, multi_gpu = get_device(cpu_override=cpu_override)

    transform = get_resize_transform(preprocess_config["img_size"])
    dataset = get_dataset(dataset_config, transform=transform)

    class_counts = Counter(dataset.labels)
    class_weights = compute_class_weights(class_counts, device)
    print(f"Class counts: {class_counts}, Class weights: {class_weights}")

    val_size = dataset_config.get("val_size", 0.1)
    test_size = dataset_config.get("test_size", 0.1)
    train_idx, val_idx, test_idx = stratified_split(
        dataset.labels,
        val_size=val_size,
        test_size=test_size,
        seed=system_config["seed"],
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print(
        f"Dataset split -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    num_workers = 0 if not multi_gpu else 2

    batch_size = dataset_config["batch_size"]
    if device.type == "cuda" and multi_gpu:
        batch_size *= torch.cuda.device_count()

    loaders = {
        "train": DataLoader(
            train_dataset, batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_dataset, batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            test_dataset, batch_size, shuffle=False, num_workers=num_workers
        ),
    }

    model = get_model(model_config, data_loader=loaders, device=device)
    model.load_state_dict(
        torch.load(
            os.path.join(
                system_config["save_path"], system_config["save_name"] + ".pth.tar"
            ),
            weights_only=False,
            map_location=torch.device(device),
        )
    )
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

    trainer.test(split="val")
