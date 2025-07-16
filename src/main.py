import argparse
import yaml
import torch
import random
import numpy as np
from .tasks import get_task
from .config.device_config import get_device


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["train", "visualize"])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model_path", help="Path to model weights (for visualization)"
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["system"]["seed"])

    task_fn = get_task(args.task)
    task_fn(config, model_path=args.model_path, cpu_override=args.cpu)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
    torch.cuda.empty_cache()
