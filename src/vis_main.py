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
from collections import Counter
from tqdm import tqdm
from models.model_conv_attributes_32 import CapsuleNetworkWithAttributes32
from utils.visualization.capsule_contribution import perturb_all_capsules


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
size = 282  # for 32x32 inputs in caps
epochs = 50
batch_size = 32
learning_rate = 1e-3
routing_steps = 3
lr_decay = 0.96
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

    loaders = {}
    loaders["test"] = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_attributes = loaders["test"].dataset[0][2].shape[0]

    network = CapsuleNetworkWithAttributes32(
        img_shape=loaders["test"].dataset[0][0].numpy().shape,
        channels=3,
        primary_dim=8,
        num_classes=2,
        num_attributes=num_attributes,
        output_dim=16,
        routing_steps=routing_steps,
        device=device,
        routing_algorithm="sigmoid",
    )

    class_0_images = []
    class_1_images = []

    for images, labels, _, _ in loaders["test"]:
        for img, label in zip(images, labels):
            if label.item() == 0 and len(class_0_images) < 3:
                class_0_images.append(img)
            elif label.item() == 1 and len(class_1_images) < 3:
                class_1_images.append(img)

            if len(class_0_images) == 3 and len(class_1_images) == 3:
                break
        if len(class_0_images) == 3 and len(class_1_images) == 3:
            break

    class_0_images = torch.stack(class_0_images)
    class_1_images = torch.stack(class_1_images)

    all_images = torch.cat([class_0_images, class_1_images], dim=0)

    network.load_state_dict(torch.load(args.model, weights_only=False))
    for i, image in enumerate(all_images):
        perturb_all_capsules(
            network,
            image,
            device=device,
            visual_attributes=dataset.visual_attributes,
            out_prefix=f"img{i}_label{0 if i < 3 else 1}",
        )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
    torch.cuda.empty_cache()
