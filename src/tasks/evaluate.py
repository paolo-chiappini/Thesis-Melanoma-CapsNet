import os
from utils.loaders import get_dataset
from models import get_model
from trainers import get_trainer
from utils.losses import get_loss
from utils.commons import get_resize_transform
from config.device_config import get_device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from utils.evaluation import (
    evaluate_reconstruction,
    compute_capsule_activations,
    mutual_information_capsules,
    summarize_evaluation,
)
from utils.visualization import plot_mi_heatmap
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


def run_evaluation(config, model_path=None, cpu_override=False):
    evaluation_config = config["evaluate"]
    dataset_config = config["dataset"]
    preprocess_config = config["preprocess"]
    model_config = config["model"]
    trainer_config = config["trainer"]
    system_config = config["system"]

    device, multi_gpu = get_device(cpu_override=cpu_override)

    tranform = get_resize_transform(preprocess_config["img_size"])
    dataset = get_dataset(dataset_config, transform=tranform)

    labels = dataset.labels
    attribute_names = dataset.visual_attributes

    indices = np.arange(len(labels))
    labels = np.array(labels)

    test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=evaluation_config["split_size"],
        random_state=system_config["seed"],
    )
    _, test_idx = next(test.split(indices, labels))
    dataset = Subset(dataset, test_idx)

    num_workers = 0 if not multi_gpu else 2

    batch_size = dataset_config["batch_size"]
    if device.type == "cuda" and multi_gpu:
        batch_size *= torch.cuda.device_count()

    loader = {
        "test": DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    }

    model = get_model(model_config, data_loader=loader, device=device)
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

    loss_criterion = get_loss(trainer_config["loss"], None)
    trainer = get_trainer(
        trainer_config,
        model,
        loader,
        loss_criterion,
        device=device,
        checkpoints_dir=system_config["save_path"],
        save_name=system_config["save_name"],
    )
    prepare_batch = trainer.prepare_batch

    recon_results = evaluate_reconstruction(
        model, dataloader=loader["test"], device=device, prepare_batch=prepare_batch
    )
    capsule_activations, attributes = compute_capsule_activations(
        model, dataloader=loader["test"], device=device, prepare_batch=prepare_batch
    )
    mi_results = mutual_information_capsules(
        capsule_activations, attributes, discrete_attributes=True
    )

    # TODO: check if visual attributes exist in dataset
    df_recon, df_mi = summarize_evaluation(recon_results, mi_results, attribute_names)

    df_recon.to_excel("./plots/reconstructions.xlsx", index=False)
    df_mi.to_excel("./plots/mutual_information.xlsx", index=False)

    plot_mi_heatmap(mi_results, attribute_names=attribute_names)
