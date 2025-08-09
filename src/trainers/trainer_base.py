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
        metrics=None,  # dictionary of torchmetrics
    ):
        self.model = model.to(device)
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoints_dir = checkpoints_dir
        self.save_name = save_name
        self.metrics = metrics or {}
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
            setattr(self, key, self._to_tensor(value))

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
    def unpack_model_outputs(self, outputs):
        """
        Unpacks the outputs of the model.

        Args:
            outputs (Tuple): Tuple of items returned by the model forward method.
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
    def compute_custom_metrics(self, outputs, batch_data):
        """
        Computes model-specific evaluation metrics

        Args:
            outputs (Tuple): outputs of the model
            batch_data (Tuple): data of the current batch
        """
        pass

    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def _update_metrics(self, outputs, batch_data):
        outputs_dict = self.unpack_model_outputs(outputs)

        preds = torch.sigmoid(outputs_dict["preds"]) > 0.5
        preds = preds.int()
        targets = batch_data["visual_attributes"].int()
        for name, metric in self.metrics.items():
            if "attr_" in name:
                idx = int(name.split("_")[-1])
                metric.update(preds[:, idx], targets[:, idx])
            else:
                metric.update(preds, targets)

    def _compute_metrics(self, outputs=None, batch_data=None):
        """
        Compute metrics from TorchMetrics and optional subclass `compute_metrics`.
        """
        metrics_dict = {name: m.compute().item() for name, m in self.metrics.items()}

        # If subclass has its own compute_metrics method
        if hasattr(self, "compute_metrics") and callable(
            getattr(self, "compute_metrics")
        ):
            if outputs is not None and batch_data is not None:
                custom_metrics = super().compute_metrics(outputs, batch_data)
                metrics_dict.update(custom_metrics)

        return metrics_dict

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

    def train_one_epoch(self, epoch, callback_manager=None, split="train"):
        assert split in self.loaders, f"'{split}' loader not found in self.loaders."

        self.model.train()
        self._reset_metrics()
        all_custom_metrics = []
        running_loss = 0.0
        loader = self.loaders[split]

        progress = tqdm(loader, desc=f"Train Epoch {epoch}")
        for i, batch in enumerate(progress):
            batch_data = self.prepare_batch(batch)
            self.optimizer.zero_grad()

            outputs = self.model(batch_data["inputs"])
            losses = self.compute_loss(outputs, batch_data)
            losses = losses if isinstance(losses, dict) else {"loss": losses}
            total_loss = sum(losses.values())
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()
            progress.set_postfix(loss=running_loss / (i + 1))

            # TODO: wrap in debug statement
            loss_log = {k: v.item() for k, v in losses.items()}
            print(f"[Epoch {epoch} | Batch {i}] Loss components: {loss_log}")

            self._update_metrics(outputs, batch_data)

            if hasattr(self, "compute_custom_metrics"):
                all_custom_metrics.append(
                    self.compute_custom_metrics(outputs, batch_data)
                )

            if callback_manager:
                callback_manager.on_batch_end(
                    batch=i,
                    logs={
                        "loss": running_loss / (i + 1),
                        "epoch": epoch,
                        "phase": split,
                    },
                )

        tm_metrics = self._compute_metrics()
        avg_custom_metrics = self._aggregate_metrics(all_custom_metrics)
        avg_metrics = {**tm_metrics, **avg_custom_metrics}

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
        self._reset_metrics()
        all_custom_metrics = []
        running_loss = 0.0
        loader = self.loaders[split]

        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch_data = self.prepare_batch(batch)
                images = batch_data["inputs"]
                outputs = self.model(images)
                outputs_dict = self.unpack_model_outputs(outputs)
                losses = self.compute_loss(outputs, batch_data)
                losses = losses if isinstance(losses, dict) else {"loss": losses}
                running_loss += sum(losses.values()).item()

                if callback_manager:
                    callback_manager.on_batch_end(
                        batch=i,
                        logs={
                            "loss": running_loss / (i + 1),
                            "epoch": epoch,
                            "phase": split,
                        },
                    )

                self._update_metrics(outputs, batch_data)

                if hasattr(self, "compute_custom_metrics"):
                    all_custom_metrics.append(
                        self.compute_custom_metrics(outputs, batch_data)
                    )

                # TODO: remove, may be used for visualization still (makes the segmentation comparison clearer)
                masks = batch_data["masks"]

        tm_metrics = self._compute_metrics()
        avg_custom_metrics = self._aggregate_metrics(all_custom_metrics)
        avg_metrics = {**tm_metrics, **avg_custom_metrics}

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

            self.early_stop = self.early_stop or logs.get("stop", False)

            if "reconstructions" in outputs_dict:
                callback_manager.on_reconstruction(
                    images[:8] * masks[:8],
                    outputs_dict["reconstructions"][:8],
                    epoch,
                    split,
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
        self._reset_metrics()
        all_custom_metrics = []
        running_loss = 0.0

        loader = self.loaders[split]
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Testing on {split}"):
                batch_data = self.prepare_batch(batch)
                outputs = self.model(batch_data["inputs"])

                losses = self.compute_loss(outputs, batch_data)
                losses = losses if isinstance(losses, dict) else {"loss": losses}
                running_loss += sum(losses.values()).item()

                self._update_metrics(outputs, batch_data)

                if hasattr(self, "compute_custom_metrics"):
                    all_custom_metrics.append(
                        self.compute_custom_metrics(outputs, batch_data)
                    )

        results = {"loss": running_loss / len(loader)}

        tm_metrics = self._compute_metrics()
        avg_custom_metrics = self._aggregate_metrics(all_custom_metrics)
        avg_metrics = {**tm_metrics, **avg_custom_metrics}

        results.update(avg_metrics)

        print(f"Test Results on {split} split -> {results}")
        return results
