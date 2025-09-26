import os
from datetime import datetime

import matplotlib.figure
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .training_logger import generate_experiment_name


class TBLogger:
    def __init__(self, config: dict, log_root: str = "tb_runs"):
        """
        Args:
            config (dict): experiment configuration (hyperparameters, dataset, etc.).
            log_root (str): root folder for TensorBoard logs.
        """
        self.config = config

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = generate_experiment_name(config=config)
        self.log_dir = f"{log_root}/{timestamp}_{run_name}"

        self.weights_dir = os.path.join(self.log_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        with open(f"{self.log_dir}/config.yaml", "w") as f:
            yaml.dump(config, f)

        # log hparams to TensorBoard
        flat_config = self._flatten_dict(config)
        self.writer.add_hparams(flat_config, {})

    def log_metrics(self, metrics: dict, step: int):
        """
        Log scalar metrics (loss, accuracy, etc.)
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
        print("Metrics saved")

    def log_image(self, name: str, image, step: int):
        """
        Log a single image (tensor or matplotlib figure)
        """
        if isinstance(image, torch.Tensor):
            # if batch of images -> make grid
            if image.dim() == 4:
                grid = make_grid(image, normalize=True, scale_each=True)
                self.writer.add_image(name, grid, step)
            else:
                self.writer.add_image(name, image, step)
            print(f"Tensor image saved ({name})")
        elif isinstance(image, matplotlib.figure.Figure):
            self.writer.add_figure(name, image, step)
            print(f"Matplotlib image saved ({name})")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = "/"):
        """
        Flatten nested config dicts for hparam logging
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, bool, str)):
                items.append((new_key, v))
            elif v is None:
                items.append((new_key, "None"))
            else:
                try:
                    items.append((new_key, str(v)))
                except Exception:
                    items.append((new_key, "<unsupported>"))
        return dict(items)

    def save_model(self, model, filename="model_final.pth"):
        model = model.module if isinstance(model, torch.nn.DataParallel) else model
        model_path = os.path.join(self.weights_dir, filename)
        if not model_path.endswith(".pth"):
            model_path = f"{model_path}.pth"

        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict_cpu, model_path)
        print(f"Model saved to {model_path}")

    def close(self):
        self.writer.close()
