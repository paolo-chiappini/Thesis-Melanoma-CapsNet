import os
from collections import OrderedDict

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from .datasets.augmentations import get_transforms


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


def load_model(
    model_structure,
    model_name,
    checkpoints_dir="checkpoints",
    device="cpu",
):
    state_dict = torch.load(
        os.path.join(checkpoints_dir, model_name + ".pth.tar"),
        weights_only=False,
        map_location=torch.device(device),
    )

    model = model_structure
    model = model.to(device)

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


def compute_weighted_accuracy(predicted, target, weights=None, num_labels=2):
    target = target.long()  # ensure it's long for indexing
    if weights is None:
        weights = torch.ones(num_labels, device=predicted.device)

    correct = (predicted == target).float()
    weighted_accuracy = (correct * weights[target]).sum() / weights[target].sum()
    weighted_accuracy = weighted_accuracy.item()

    return weighted_accuracy


def compute_binary_feature_weights(counts_ones, num_samples, device):
    """
    counts_ones: array of length F (count of 1s for each feature)
    num_samples: total number of samples (N)
    """
    counts_zeros = num_samples - counts_ones
    pos_weight = counts_zeros / (counts_ones + 1e-8)  # shape (F,)
    return torch.tensor(pos_weight, dtype=torch.float32).to(device)


def compute_class_weights(class_counts, device):
    counts = np.array(list(class_counts.values()), dtype=np.float32)
    weights = 1.0 / counts
    weights /= weights.sum()
    return torch.tensor(weights).to(device)


def stratified_split(labels, groups=None, val_size=0.1, test_size=0.1, seed=123):
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


def group_stratified_split(labels, groups, test_size=0.1, val_size=0.1, seed=123):
    if len(labels) != len(groups):
        raise ValueError("Labels and groups must have the same length.")

    labels = np.array(labels)
    groups = np.array(groups)
    indices = np.arange(len(labels))

    n_splits_test = int(np.ceil(1.0 / test_size))
    sgkf_test = StratifiedGroupKFold(
        n_splits=n_splits_test, shuffle=True, random_state=seed
    )

    train_val_idx, test_idx = next(sgkf_test.split(indices, labels, groups))

    train_val_labels = labels[train_val_idx]
    train_val_groups = groups[train_val_idx]

    val_proportion = val_size / (1.0 - test_size)
    n_splits_val = int(np.ceil(1.0 / val_proportion))
    sgkf_val = StratifiedGroupKFold(
        n_splits=n_splits_val, shuffle=True, random_state=seed
    )

    train_sub_idx, val_sub_idx = next(
        sgkf_val.split(train_val_idx, train_val_labels, train_val_groups)
    )

    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])

    assert (
        len(train_groups.intersection(val_groups)) == 0
    ), "FATAL: Patient overlap between train and val"
    assert (
        len(train_groups.intersection(test_groups)) == 0
    ), "FATAL: Patient overlap between train and test"
    assert (
        len(val_groups.intersection(test_groups)) == 0
    ), "FATAL: Patient overlap between val and test"

    return train_idx, val_idx, test_idx


def build_dataloaders(config, dataset, batch_size, num_workers=0):
    transform_val = get_transforms(config, is_train=False)

    system_config = config["system"]
    dataset_config = config["dataset"]

    val_size = dataset_config.get("val_size", 0.1)
    test_size = dataset_config.get("test_size", 0.1)

    split_method = group_stratified_split if hasattr(dataset, 'groups') else stratified_split
    print(f'Chosen split method: {split_method}')

    train_idx, val_idx, test_idx = split_method(
        dataset.labels,
        dataset.groups,
        val_size=val_size,
        test_size=test_size,
        seed=system_config["seed"],
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    val_dataset.dataset.tranform = transform_val
    test_dataset.dataset.tranform = transform_val

    print(
        f"Dataset split -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

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

    return loaders
