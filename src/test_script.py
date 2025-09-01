import argparse
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config.device_config import get_device
from utils.commons import build_dataloaders
from utils.datasets.augmentations import get_transforms
from utils.loaders import get_dataset


class EarlyStopper:
    def __init__(self, patience=5, mode="min", delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss if self.mode == "min" else val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, epoch):
        print(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )
        self.val_loss_min = val_loss


def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, 1)

    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    acc = (preds == labels).float().mean().item()
    precision = precision_score(labels_np, preds_np, average="macro", zero_division=0)
    recall = recall_score(labels_np, preds_np, average="macro", zero_division=0)
    f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1}


def run_epoch(phase, loader, model, criterion, optimizer, device):
    """A single pass through the dataset for one epoch."""
    if phase == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels = []
    all_outputs = []

    progress_bar = tqdm(loader, desc=f"{phase.capitalize()} Epoch")

    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        if phase == "train":
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if phase == "train":
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_labels.append(labels.cpu())
        all_outputs.append(outputs.detach().cpu())

    epoch_loss = running_loss / len(loader.dataset)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    epoch_metrics = calculate_metrics(all_outputs, all_labels)

    print(
        f"{phase.capitalize()} Loss: {epoch_loss:.4f} "
        + " ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
    )

    return epoch_loss, epoch_metrics


def compute_class_weights(train_dataloader, num_classes):
    # Count all class occurrences in the training dataset
    class_counts = Counter()

    for _, labels in train_dataloader:
        labels = labels.view(-1)  # flatten in case labels are not already
        class_counts.update(labels.tolist())

    # Ensure all classes are included even if count is zero
    total_samples = sum(class_counts.values())
    class_freqs = [class_counts.get(i, 0) for i in range(num_classes)]

    # Inverse frequency weighting
    weights = [
        total_samples / (num_classes * freq) if freq > 0 else 0.0
        for freq in class_freqs
    ]
    weights_tensor = torch.tensor(weights)

    return weights_tensor


def main(config):
    device, _ = get_device()
    torch.manual_seed(config["system"]["seed"])
    os.makedirs(config["system"]["save_path"], exist_ok=True)
    model_save_path = os.path.join(
        config["system"]["save_path"], f"{config['model']['name']}_best.pth.tar"
    )

    transform = get_transforms(config=config, is_train=True)
    dataset = get_dataset(config=config["dataset"], transform=transform)
    dataloaders = build_dataloaders(
        config, dataset=dataset, batch_size=config["dataset"]["batch_size"]
    )

    print("Loading model...")
    model = models.resnet34(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2),
    )
    model.to(device)

    class_weights = compute_class_weights(dataloaders["train"], num_classes=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print(f"Using class weights: {class_weights.cpu().numpy()}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["trainer"]["learning_rate"],
        weight_decay=1e-4,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
    )

    start_time = time.time()
    early_stopper = EarlyStopper(patience=20)

    for epoch in range(config["trainer"]["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{config['trainer']['epochs']} ---")

        train_loss, train_metrics = run_epoch(
            "train", dataloaders["train"], model, criterion, optimizer, device
        )

        val_loss, val_metrics = run_epoch(
            "val", dataloaders["val"], model, criterion, optimizer, device
        )

        scheduler.step(val_loss)

        early_stopper(val_loss, model, model_save_path, epoch)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break

    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # --- Optional: Final Testing on Test Set ---
    # print("\n--- Testing on final model ---")
    # model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
    # test_loss, test_metrics = run_epoch('test', dataloaders['test'], model, criterion, optimizer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model_path", help="Path to model weights (for visualization)"
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
