import argparse
import yaml
import torch
import random
import numpy as np
from tasks import get_task, tasks
import time
from datetime import timedelta


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=[*tasks.keys()],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model_path", help="Path to model weights (for visualization)"
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["system"]["seed"])

    print(">" * 5 + f" Running task: [{args.task}]")

    task_cls = get_task(args.task)
    task = task_cls(config, model_path=args.model_path, cpu_override=args.cpu)

    task.run()

    elapsed = time.time() - start_time
    print(f"\n✅ Finished in {str(timedelta(seconds=round(elapsed)))}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
    torch.cuda.empty_cache()
