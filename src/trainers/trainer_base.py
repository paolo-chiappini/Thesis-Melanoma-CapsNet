# credits: https://github.com/danielhavir/capsule-network

import torch
import torch.nn as nn
import torch.optim as optim
import os
from numpy import prod
from datetime import datetime
from time import time
from tqdm import tqdm

CHECKPOINTS_PATH = "checkpoints/"
if not os.path.exists(CHECKPOINTS_PATH):
    os.mkdir(CHECKPOINTS_PATH)


class CapsNetTrainerBase:
    def __init__(
        self,
        loaders,
        learning_rate,
        lr_decay=0.9,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        multi_gpu=(torch.cuda.device_count() > 1),
        network_architecture=None,
        loss_function=None,
    ):
        assert network_architecture is not None, "Network architecture must be provided"
        assert loss_function is not None, "Loss function must be provided"

        self.device = device
        self.multi_gpu = multi_gpu
        self.loaders = loaders
        self.network = network_architecture.to(self.device)

        if self.multi_gpu:
            self.network = nn.DataParallel(self.network)

        self.loss_function = loss_function
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_decay
        )

    def __repr__(self):
        return repr(self.network)

    def run(self, epochs, classes, callback_manager=None):
        print(8 * "=", "Run started".upper(), 8 * "=")
        self.classes = classes
        eye = torch.eye(len(classes)).to(self.device)

        for epoch in range(1, epochs + 1):
            epoch_start_time = time()
            for phase in ["train", "test"]:
                print(f"{phase.capitalize()}ing...")
                self.network.train() if phase == "train" else self.network.eval()

                t0 = time()
                phase_results = self._run_phase(epoch, phase, eye, callback_manager)
                elapsed_time = time() - t0
                phase_results["time"] = round(elapsed_time, 3)
                self._on_phase_end(phase, epoch, phase_results, callback_manager)

            self.scheduler.step()
            print(f"Total epoch time: {round(time() - epoch_start_time, 2)} seconds\n")

        self._save_model()

    def _run_phase(self, epoch, phase, eye, callback_manager):
        running_loss = 0.0
        correct = 0
        total = 0

        loader = self.loaders[phase]
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"{phase.capitalize()} Epoch {epoch}",
        )

        for i, batch in pbar:
            loss, batch_correct, batch_total, batch_reconstructions = self._run_batch(
                batch, eye, phase
            )
            running_loss += loss
            correct += batch_correct
            total += batch_total
            accuracy = float(correct) / float(total)

            pbar.set_postfix(
                {"Loss": f"{running_loss/(i+1):.4f}", "Acc": f"{accuracy:.4f}"}
            )

            if phase == "train" and callback_manager:
                callback_manager.on_batch_end(
                    batch=i,
                    logs={
                        "loss": running_loss / (i + 1),
                        "accuracy": accuracy,
                        "epoch": epoch,
                        "phase": phase,
                    },
                )

        return {
            "loss": running_loss / (i + 1),
            "accuracy": accuracy,
            "reconstructions": batch_reconstructions,
        }

    def _run_batch(self, batch, eye, phase):
        images, labels, visual_attributes = batch
        images, labels = images.to(self.device), labels.to(self.device)
        labels_one_hot = eye[labels]

        self.optimizer.zero_grad()

        outputs, reconstructions = self.network(images)
        loss = self.loss_function(
            outputs,
            labels,
            images,
            reconstructions,
        )

        if phase == "train":
            loss.backward()
            self.optimizer.step()

        _, predicted = torch.max(malignancy_scores, 1)
        _, labels = torch.max(labels_one_hot, 1)
        correct = (predicted == labels).sum()
        total = labels.size(0)

        return loss.item(), correct, total, reconstructions

    def _on_phase_end(self, phase, epoch, metrics, callback_manager):
        print(
            f"{phase.upper()} Epoch {epoch}, "
            f"Loss {metrics['loss']:.4f}, Accuracy {metrics['accuracy']:.4f}, "
            f"Time {metrics['time']}s"
        )
        if callback_manager:
            callback_manager.on_epoch_end(
                epoch=epoch, logs={**metrics, "epoch": epoch, "phase": phase}
            )

        if phase == "test":
            images, _, _ = next(iter(self.loaders["test"]))
            callback_manager.on_reconstruction(
                images[:8], metrics["reconstructions"][:8], epoch, phase
            )

    def _save_model(self):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        accuracy = self._saved_accuracy
        error_rate = round((1 - accuracy) * 100, 2)
        path = os.path.join(CHECKPOINTS_PATH, f"{error_rate}_{now}.pth.tar")
        torch.save(self.network.state_dict(), path)

    def _evaluate_per_class_accuracy(self):
        class_correct = [0.0 for _ in self.classes]
        class_total = [0.0 for _ in self.classes]

        for images, labels, visual_attributes in self.loaders["test"]:
            images, labels, visual_attributes = (
                images.to(self.device),
                labels.to(self.device),
                visual_attributes.to(self.device),
            )
            outputs, _, malignancy_scores = self.network(images)
            _, predicted = torch.max(malignancy_scores, 1)
            c = (predicted == labels).squeeze()

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i, cls in enumerate(self.classes):
            acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            print(f"Accuracy of {cls:>5} : {acc:.2f} %")
