from abc import ABC, abstractmethod
import torch
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np


class BaseTrainer(ABC):
    def __init__(
        self,
        model,
        loaders,
        criterion,
        optimizer,
        scheduler=None,
        device="cuda",
        checkpoints_dir="checkpoints",
        save_name=None,
    ):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoints_dir = checkpoints_dir
        self.save_name = save_name
        self.class_weights = None
        self.attribute_weights = None
        self.early_stop = False

        os.makedirs(checkpoints_dir, exist_ok=True)

    def set_weights(self, weights_dict):
        """
        Set weight attributes from a dictionary.

        Parameters:
        - weights_dict: dict with keys like "class_weights", "attribute_weights".
        """
        if not isinstance(weights_dict, dict):
            raise TypeError("weights_dict must be a dictionary")

        valid_keys = {"class_weights", "attribute_weights"}

        for key, value in weights_dict.items():
            if key not in valid_keys:
                raise KeyError(
                    f"Invalid key '{key}' in weights_dict. Valid keys: {valid_keys}"
                )

            if key == "class_weights":
                self.class_weights = self._to_tensor(value)
            elif key == "attribute_weights":
                self.attribute_weights = self._to_tensor(value)

    def _to_tensor(self, value):
        """
        Convert weights to a torch tensor (float32).
        """
        if isinstance(value, torch.Tensor):
            return value.float()
        elif isinstance(value, (list, tuple, np.ndarray)):
            return torch.tensor(value, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported weight type: {type(value)}")

    @abstractmethod
    def prepare_batch(self, batch):
        """
        Unpacks a batch and moves it to device, return data needed for forward pass and loss computation.

        Args:
            batch (Tuple): Tuple of items used in the training loop (e.g. (image, label, mask))
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, outputs, batch_data):
        """
        Computes model-specific loss

        Args:
            outputs (Tuple): outputs of the model
            batch_data (Tuple): data of the current batch
        """
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, outputs, batch_data):
        """
        Computes model-specific evaluation metrics

        Args:
            outputs (Tuple): outputs of the model
            batch_data (Tuple): data of the current batch
        """
        raise NotImplementedError

    def train_one_epoch(self, epoch, callback_manager=None, split="train"):
        assert split in self.loaders, f"'{split}' loader not found in self.loaders."

        self.model.train()
        running_loss = 0.0
        loader = self.loaders[split]
        all_metrics = []

        progress = tqdm(loader, desc=f"Train Epoch {epoch}")
        for i, batch in enumerate(progress):
            batch_data = self.prepare_batch(batch)
            self.optimizer.zero_grad()

            outputs = self.model(batch_data["inputs"])
            loss = self.compute_loss(outputs, batch_data)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / (i + 1))

            if callback_manager:
                callback_manager.on_batch_end(
                    batch=i,
                    logs={
                        "loss": running_loss / (i + 1),
                        "epoch": epoch,
                        "phase": split,
                    },
                )

            metrics = self.compute_metrics(outputs, batch_data)
            all_metrics.append(metrics)

        avg_metrics = self._aggregate_metrics(all_metrics)
        if callback_manager:
            logs = {
                "loss": running_loss / len(loader),
                "epoch": epoch,
                "phase": split,
            }
            logs.update(avg_metrics)

            callback_manager.on_epoch_end(
                epoch=epoch,
                logs=logs,
            )

        return running_loss / len(loader)

    def evaluate(self, epoch, callback_manager=None, split="val"):
        assert split in self.loaders, f"'{split}' loader not found in self.loaders."

        self.model.eval()
        running_loss = 0.0
        all_metrics = []
        loader = self.loaders[split]
        images = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch_data = self.prepare_batch(batch)
                images = batch_data["inputs"]
                outputs = self.model(images)
                loss = self.compute_loss(outputs, batch_data)
                running_loss += loss.item()

                if callback_manager:
                    callback_manager.on_batch_end(
                        batch=i,
                        logs={
                            "loss": running_loss / (i + 1),
                            "epoch": epoch,
                            "phase": split,
                        },
                    )

                metrics = self.compute_metrics(outputs, batch_data)
                all_metrics.append(metrics)

        avg_metrics = self._aggregate_metrics(all_metrics)
        if callback_manager:
            logs = {
                "loss": running_loss / len(loader),
                "epoch": epoch,
                "phase": split,
                "model": self.model,
            }
            logs.update(avg_metrics)

            callback_manager.on_epoch_end(
                epoch=epoch,
                logs=logs,
            )

            # check if an early stop signal has been raised
            if not self.early_stop:
                self.early_stop = logs.get("stop", False)

            reconstructions = outputs[1]
            callback_manager.on_reconstruction(
                images[:8], reconstructions[:8], epoch, split
            )
        return running_loss / len(loader)

    def save_checkpoint(self, name=None):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        name = name or f"model_{now}"
        name += ".pth.tar"
        output_dir = os.path.join(self.checkpoints_dir, name)
        torch.save(self.model.state_dict(), output_dir)
        print(f"Model {name} saved at location {output_dir}")

    def run(self, epochs, callback_manager=None):
        print(callback_manager)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch, callback_manager)
            val_loss = self.evaluate(epoch, callback_manager)

            print(
                f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}"
            )

            if self.early_stop:
                print(f"🛑 Early stop at epoch {epoch}")
                break

            if self.scheduler:
                self.scheduler.step()

        self.save_checkpoint(self.save_name)

    def test(self, split="test"):
        """
        Runs evaluation on the holdout/test set.
        """
        assert split in self.loaders, f"'{split}' loader not found in self.loaders."

        self.model.eval()
        running_loss = 0.0
        all_metrics = []

        loader = self.loaders[split]
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Testing on {split}"):
                batch_data = self.prepare_batch(batch)
                outputs = self.model(batch_data["inputs"])

                # compute loss
                loss = self.compute_loss(outputs, batch_data)
                running_loss += loss.item()

                # compute metrics
                metrics = self.compute_metrics(outputs, batch_data)
                all_metrics.append(metrics)

        # aggregate metrics (assuming metrics are dicts)
        avg_metrics = self._aggregate_metrics(all_metrics)
        avg_loss = running_loss / len(loader)

        print(
            f"Test Results on {split} split -> Loss: {avg_loss:.4f}, Metrics: {avg_metrics}"
        )
        return {"loss": avg_loss, **avg_metrics}

    def _aggregate_metrics(self, metrics_list):
        """
        Aggregates a list of metrics dictionaries into averages.
        """
        if not metrics_list:
            return {}

        aggregated = {}
        for key in metrics_list[0]:
            aggregated[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        return aggregated
