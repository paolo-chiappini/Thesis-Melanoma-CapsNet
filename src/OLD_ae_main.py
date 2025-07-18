import os
import sys
import torch
import numpy as np
import argparse
import random
from torchvision import transforms as T
from utils.loaders import get_dataset
from collections import Counter
from trainers import trainer_autoencoder
from utils.callbacks import PlotCallback, ReconstructionCallback, CallbackManager
from models.autoencoder import ConvAutoencoder
from src.utils.losses.losses_ae import AECompositeLoss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_state = 123
set_seed(random_state)

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
parser.add_argument("--model", help="Name of the model to load")

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
size = 256
epochs = 75
batch_size = 32
learning_rate = 1e-3
classes = range(2)  # Benign 0, Malignant 1

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    batch_size *= torch.cuda.device_count()


def main():
    tensor_transform = T.Compose([T.Resize((size, size)), T.ToTensor()])

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

    num_workers = 0 if not multi_gpu else 2

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    callbacks = [
        PlotCallback(filename="ae_acc.png"),
        ReconstructionCallback(save_dir="reconstructions/ae", frequency=3),
    ]
    callback_manager = CallbackManager(callbacks)

    autoencoder = ConvAutoencoder()
    trainer_autoencoder.train_autoencoder(
        model=autoencoder,
        dataloader=loader,
        num_epochs=epochs,
        criterion=AECompositeLoss().to(device=device),
        device=device,
        callback_manager=callback_manager,
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
    torch.cuda.empty_cache()
