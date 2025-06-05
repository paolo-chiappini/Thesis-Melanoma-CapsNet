import os
import sys
import torch
from torchvision import transforms as T
from torch.utils.data import Subset
from trainers import trainer_conv_custom, trainer_with_attributes
import argparse
from utils.loaders import get_dataset
from utils.callbacks import PlotCallback, ReconstructionCallback, CallbackManager
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from utils.datasets import compute_mean_std
from collections import Counter
from tqdm import tqdm
from models.model_conv_attributes_32 import CapsuleNetworkWithAttributes32


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_state = 123
set_seed(random_state)

trainer = trainer_with_attributes
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

multi_gpu = False
# Try CUDA
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    RUN_MODE = "cuda"
    device = torch.device("cuda")
    multi_gpu = torch.cuda.device_count() > 1
    torch.cuda.empty_cache()
# Fallback to directml (for AMD GPU)
elif sys.platform.startswith("win"):
    # Check if DirectML is available
    try:
        import torch_directml
    except ImportError:
        raise ImportError(
            "DirectML is not installed. Please install it to use this mode."
        )

    if torch_directml.is_available() and torch_directml.device_count() > 0:
        RUN_MODE = "directml"
        device = torch_directml.device()
        multi_gpu = torch_directml.device_count() > 1
    else:
        raise RuntimeError("No compatible GPU found (CUDA or DirectML).")
# No supported device found (prevent running on CPU)
else:
    raise RuntimeError("No compatible GPU found (CUDA or DirectML).")

print(
    f"""
============================= CONFIG =============================
- Running mode \t\t: ({RUN_MODE})
- Running on device \t: {device}
- Multi GPU \t\t: {multi_gpu}
==================================================================
"""
)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="data")
parser.add_argument(
    "--augment",
    action="store_true",
    help="Increase size of dataset with augmentations",
)
parser.add_argument("--meta", help="Metadata filename for the dataset", default=None)
parser.add_argument(
    "--dataset",
    default="PH2",
    choices=["PH2", "EXHAM"],
    help="Dataset to use: PH2, EXHAM",
)
parser.add_argument("--cpu", action="store_true", help="Use CPU for training")

args = parser.parse_args()
if args.data_root:
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root path does not exist: {args.data_root}")
    else:
        print(f"Using data root path: {args.data_root}")
DATA_PATH = args.data_root

if args.cpu:
    print("=" * 10, "Running on CPU (OVERRIDING DEVICE)", "=" * 10)
    device = torch.device("cpu")
    multi_gpu = False

# size = 284  # 284 for conv encoder form Perér et al.
size = 282  # for 32x32 inputs in caps

tensor_transform = T.Compose([T.Resize((500, 500)), T.ToTensor()])

epochs = 50
batch_size = 32
learning_rate = 1e-3
routing_steps = 3
lr_decay = 0.96
classes = range(2)  # Benign 0, Malignant 1

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    batch_size *= torch.cuda.device_count()


def main():
    # switch between datasets
    dataset = get_dataset(
        args.dataset,
        DATA_PATH,
        transform=tensor_transform,
        metadata_path=args.meta,
        augment=args.augment,
    )
    if dataset is None:
        print(f"Dataset not found: {args.dataset}")
        exit()

    dataset.check_missing_files()

    class_counts = Counter(dataset.labels)
    print(f"Class counts: {class_counts}")

    train_idx, val_idx = next(splitter.split(np.zeros(len(dataset)), dataset.labels))
    num_workers = 0 if not multi_gpu else 2
    mean, std = compute_mean_std(
        dataset=Subset(dataset, train_idx), num_workers=num_workers
    )

    train_transform = T.Compose(
        [
            T.RandomResizedCrop(size),  # Random crop and resize
            T.RandomHorizontalFlip(),  # Horizontal flip
            T.RandomVerticalFlip(),  # Vertical flip
            T.RandomRotation(degrees=30),  # Random rotation
            # T.ColorJitter(        # Color changes may affect performance
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.2,
            #     hue=0.1
            # ),
            T.ToTensor(),  # Convert to tensor
            T.Normalize(mean=mean, std=std),  # Normalize
        ]
    )

    val_transform = T.Compose(
        [
            T.Resize((size, size)),  # Resize to fixed size
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = Subset(
        get_dataset(
            args.dataset,
            DATA_PATH,
            transform=train_transform,
            metadata_path=dataset.metadata_path,
        ),
        train_idx,
    )
    val_dataset = Subset(
        get_dataset(
            args.dataset,
            DATA_PATH,
            transform=val_transform,
            metadata_path=dataset.metadata_path,
        ),
        val_idx,
    )

    print(f"Len of train: {len(train_dataset)}")
    print(f"Len of validation: {len(val_dataset)}")

    loaders = {}
    loaders["train"] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    loaders["test"] = torch.utils.data.DataLoader(
        val_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    network = CapsuleNetworkWithAttributes32(
        img_shape=loaders["train"].dataset[0][0].numpy().shape,
        channels=3,
        primary_dim=8,
        num_classes=2,
        num_attributes=loaders["train"].dataset[0][2].shape[0],
        output_dim=16,
        routing_steps=routing_steps,
        device=device,
        routing_algorithm="sigmoid",
    )

    caps_net = trainer.CapsNetTrainer(
        loaders,
        batch_size,
        learning_rate,
        routing_steps,
        lr_decay,
        network=network,
        device=device,
        multi_gpu=multi_gpu,
        routing_algorithm="sigmoid",
    )

    callbacks = [
        PlotCallback(),
        ReconstructionCallback(frequency=5, mean=mean, std=std),
    ]
    callback_manager = CallbackManager(callbacks)

    caps_net.run(epochs, classes, callback_manager=callback_manager)
    print("=" * 10, "Run finished", "=" * 10)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
    torch.cuda.empty_cache()
