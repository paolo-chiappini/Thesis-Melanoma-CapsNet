import gc
import hashlib
import json
import os
import shutil
from datetime import datetime

import matplotlib.figure
import torch
import yaml
from torchvision.utils import save_image


def generate_experiment_name(config: dict, max_len: int = 100):
    dataset_name = config.get("dataset", {}).get("name", "data")
    model_cfg = config.get("model", {})
    trainer_cfg = config.get("trainer", {})

    model_name = model_cfg.get("name", "model")
    pose_dim = model_cfg.get("pose_dim", "pd")
    epochs = trainer_cfg.get("epochs", "e")
    lr = trainer_cfg.get("learning_rate", "lr")

    loss_components = list(trainer_cfg.get("loss", {}).get("components", {}).keys())
    losses = "-".join(loss_components) if loss_components else "loss"

    raw_name = f"{dataset_name}-{model_name}-p{pose_dim}-lr{lr}-e{epochs}-{losses}"

    config_str = json.dumps(config, sort_keys=True)
    hash_digest = hashlib.md5(config_str.encode()).hexdigest()[:6]

    experiment_name = f"{raw_name}-{hash_digest}"

    return experiment_name[:max_len]


class TrainingLogger:
    def __init__(self, config: dict):
        self.config = config
        self.save_path = config["system"]["save_path"]

        self.experiment_name = generate_experiment_name(config)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_dir = os.path.join(
            self.save_path, f"{timestamp}_{self.experiment_name}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.weights_dir = os.path.join(self.experiment_dir, "weights")
        self.metrics_dir = os.path.join(self.experiment_dir, "metrics")
        self.plots_dir = os.path.join(self.experiment_dir, "plots")
        self.reconstructions_dir = os.path.join(self.plots_dir, "reconstructions")
        self.images_dir = os.path.join(self.experiment_dir, "images")

        for d in [
            self.weights_dir,
            self.metrics_dir,
            self.plots_dir,
            self.reconstructions_dir,
            self.images_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        # Save a copy of the config
        with open(os.path.join(self.experiment_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    def save_model(self, model, filename="model_final.pth"):
        gc.collect()
        torch.cuda.empty_cache()

        model = model.module if isinstance(model, torch.nn.DataParallel) else model
        model_path = os.path.join(self.weights_dir, filename)
        if not model_path.endswith(".pth"):
            model_path = f"{model_path}.pth"

        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict_cpu, model_path)
        print(f"Model saved to {model_path}")

    def save_metrics(self, metrics: dict, filename="metrics.json"):
        metrics_path = os.path.join(self.metrics_dir, filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")

    def save_plot(self, figure, name: str):
        path = os.path.join(self.plots_dir, f"{name}.png")
        figure.savefig(path)
        print(f"Plot saved to {path}")

    def save_image(self, image, name: str):
        path = os.path.join(self.images_dir, f"{name}.png")

        if isinstance(image, torch.Tensor):
            save_image(image, path)
            print(f"Tensor image saved to {path}")
        elif isinstance(image, matplotlib.figure.Figure):
            image.savefig(path)
            print(f"Matplotlib image saved to {path}")
        else:
            raise TypeError(f"Unsupported type for save_image: {type(image)}")

    def save_artifact(self, src_path: str, subfolder: str = "misc"):
        target_dir = os.path.join(self.experiment_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(src_path, target_dir)
        print(f"Artifact copied to {target_dir}")
