import os
import argparse
import sys
import torch
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_state = 123
set_seed(random_state)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)


def get_device(cpu_override=False):
    multi_gpu = False

    if cpu_override:
        return torch.device("cpu"), False, "cpu"

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        multi_gpu = torch.cuda.device_count() > 1
        torch.cuda.empty_cache()
        return device, multi_gpu, "cuda"

    elif sys.platform.startswith("win"):
        try:
            import torch_directml
        except ImportError:
            raise ImportError(
                "DirectML is not installed. Please install it to use this mode."
            )
        if torch_directml.is_available() and torch_directml.device_count() > 0:
            device = torch_directml.device()
            multi_gpu = torch_directml.device_count() > 1
            return device, multi_gpu, "directml"
        else:
            raise RuntimeError("No compatible GPU found (CUDA or DirectML).")

    raise RuntimeError("No compatible GPU found (CUDA or DirectML).")


def print_device_info(run_mode, device, multi_gpu):
    print(
        f"""
============================= CONFIG =============================
- Running mode \t\t: ({run_mode})
- Running on device \t: {device}
- Multi GPU \t\t: {multi_gpu}
==================================================================
"""
    )


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="data")
parser.add_argument("--augment", action="store_true")
parser.add_argument("--meta", default=None)
parser.add_argument("--dataset", default="PH2", choices=["PH2", "EXHAM"])
parser.add_argument("--cpu", action="store_true", default=False)

args = parser.parse_args()
DATA_PATH = args.data_root

device, multi_gpu, run_mode = get_device(cpu_override=args.cpu)
print_device_info(run_mode, device, multi_gpu)

if not os.path.exists(args.data_root):
    raise FileNotFoundError(f"Data root path does not exist: {args.data_root}")
else:
    print(f"Using data root path: {args.data_root}")

batch_size = 32
if run_mode == "cuda" and multi_gpu:
    batch_size *= torch.cuda.device_count()
