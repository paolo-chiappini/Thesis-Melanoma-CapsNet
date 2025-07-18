from utils.loaders import get_dataset
from models import get_model
from trainers import get_trainer
from utils.losses import get_loss
from utils.callbacks import get_callbacks, CallbackManager
from config.device_config import get_device

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import Counter
import numpy as np


def get_resize_transform(size):
    from torchvision import transforms as T

    return T.Compose([T.Resize((size, size)), T.ToTensor()])


def compute_class_weights(class_counts, device):
    counts = np.array(list(class_counts.values()), dtype=np.float32)
    weights = 1.0 / counts
    weights /= weights.sum()
    return torch.tensor(weights).to(device)


def make_splitter(seed):
    from sklearn.model_selection import StratifiedShuffleSplit

    return StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)


def run_training(config, model_path=None, cpu_override=False):
    dataset_config = config["dataset"]
    preprocess_config = config["preprocess"]
    model_config = config["model"]
    trainer_config = config["trainer"]
    callback_config = config.get("callbacks", [])
    system_config = config["system"]

    device, multi_gpu = get_device(cpu_override=cpu_override)

    transform = get_resize_transform(preprocess_config["img_size"])
    dataset = get_dataset(dataset_config, transform=transform)

    class_counts = Counter(dataset.labels)
    class_weights = compute_class_weights(class_counts, device)
    print(f"Class counts: {class_counts}, Class weights: {class_weights}")

    splitter = make_splitter(system_config["seed"])
    train_idx, val_idx = next(splitter.split(np.zeros(len(dataset)), dataset.labels))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    num_workers = 0 if not multi_gpu else 2

    batch_size = dataset_config["batch_size"]
    if device.type == "cuda" and multi_gpu:
        batch_size *= torch.cuda.device_count()

    loaders = {
        "train": DataLoader(
            train_dataset, batch_size, shuffle=True, num_workers=num_workers
        ),
        "test": DataLoader(
            val_dataset, batch_size, shuffle=False, num_workers=num_workers
        ),
    }

    model = get_model(model_config, data_loader=loaders, device=device)
    model = model.to(device)
    if multi_gpu:
        model = nn.DataParallel(model)

    loss_criterion = get_loss(trainer_config["loss"], class_weights)

    trainer = get_trainer(trainer_config, model, loaders, loss_criterion, device=device)

    callbacks = get_callbacks(callback_config)
    callback_manager = None
    if len(callbacks) > 0:
        callback_manager = CallbackManager(callbacks=callbacks)

    trainer.run(
        trainer_config["epochs"],
        callback_manager=callback_manager,
    )
    print("=" * 10, "Run finished", "=" * 10)
