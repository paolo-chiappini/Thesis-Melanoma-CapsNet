import torch
from collections import Counter
from utils.commons import (
    get_transforms,
    build_dataloaders,
    load_model,
    compute_class_weights,
    compute_binary_feature_weights,
)
from utils.loaders import get_dataset
from models import get_model
from config.device_config import get_device


class BaseRunner:
    def __init__(self, config, model_path=None, cpu_override=False):
        self.config = config
        self.model_path = model_path
        self.cpu_override = cpu_override

        self.device, self.multi_gpu, self.batch_size, self.num_workers = (
            self.setup_device()
        )

        self.dataset = None
        self.loaders = None
        self.model = None
        self.loss_criterion = None
        self.weights = {}

    def setup_device(self):
        device, multi_gpu = get_device(cpu_override=self.cpu_override)
        batch_size = self.config["dataset"]["batch_size"]
        if device.type == "cuda" and multi_gpu:
            batch_size *= torch.cuda.device_count()
        num_workers = 0 if not multi_gpu else 2
        return device, multi_gpu, batch_size, num_workers

    def prepare_dataset(self, is_train=True):
        transform = get_transforms(config=self.config, is_train=is_train)
        self.dataset = get_dataset(config=self.config["dataset"], transform=transform)
        self.loaders = build_dataloaders(
            config=self.config,
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def compute_weights(self):
        class_counts = Counter(self.dataset.labels)
        self.weights["class_weights"] = compute_class_weights(class_counts, self.device)

        if self.config["system"].get("has_visual_attributes", False):
            attribute_counts_ones = torch.sum(self.dataset.visual_features, dim=0)
            self.weights["attribute_weights"] = compute_binary_feature_weights(
                attribute_counts_ones, len(self.dataset), self.device
            )

    def build_model(self, load_weights=False):
        self.model = get_model(
            config=self.config["model"], data_loader=self.loaders, device=self.device
        )
        if load_weights:
            self.model = load_model(
                model_structure=self.model,
                model_name=self.config["system"]["save_name"],
                checkpoints_dir=self.config["system"]["save_path"],
                device=self.device,
            )
        elif self.multi_gpu:  # assumes that for training load_weights = False
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    # Template method
    def run(self):
        """
        Template method: defines base running behaviour.
        Subclasses should override "prepare" and "execute".
        """
        self.prepare()
        self.execute()

    def prepare(self):
        """
        Hook for specific dataset/model preparation.
        """
        raise NotImplementedError

    def execute(self):
        """
        Hook for specific running behaviour of the task.
        """
        raise NotImplementedError
