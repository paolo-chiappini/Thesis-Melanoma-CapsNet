from abc import ABC, abstractmethod
import torch
import os
from tqdm import tqdm
from datetime import datetime


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
    ):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoints_dir = checkpoints_dir

        os.makedirs(checkpoints_dir, exist_ok=True)

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

        progress = tqdm(loader, desc=f"Train Epoch {epoch}")
        for i, batch in enumerate(progress):
            batch_data = self.prepare_batch(batch)
            self.optimizer.zero_grad()

            outputs = self.model(*batch_data["inputs"])
            loss = self.compute_loss(outputs, batch_data)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / (i + 1))

            if callback_manager:
                callback_manager.on_batch_end(
                    i,
                    {"loss": running_loss / (i + 1), "epoch": epoch, "phase": "train"},
                )

        return running_loss / len(loader)

    def evaluate(self, epoch, callback_manager=None, split="val"):
        assert split in self.loaders, f"'{split}' loader not found in self.loaders."

        self.model.eval()
        running_loss = 0.0
        loader = self.loaders[split]

        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch_data = self.prepare_batch(batch)
                outputs = self.model(*batch_data["inputs"])
                loss = self.compute_loss(outputs, batch_data)
                running_loss += loss.item()

        return running_loss / len(loader)

    def save_checkpoint(self, name=None):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        name = name or f"model_{now}.pth.tar"
        torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, name))

    def run(self, epochs, callback_manager=None):
        print(callback_manager)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch, callback_manager)
            val_loss = self.evaluate(epoch, callback_manager)

            print(
                f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}"
            )
            if self.scheduler:
                self.scheduler.step()

        self.save_checkpoint()

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
                outputs = self.model(*batch_data["inputs"])

                # compute loss
                loss = self.compute_loss(outputs, batch_data)
                running_loss += loss.item()

                # compute metrics
                metrics = self.compute_metrics(outputs, batch_data)
                all_metrics.append(metrics)

        # aggregate metrics (assuming metrics are dicts)
        avg_metrics = self._aggregate_metrics(all_metrics)
        avg_loss = running_loss / len(loader)

        print(f"Test Results -> Loss: {avg_loss:.4f}, Metrics: {avg_metrics}")
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
