# credits: https://github.com/danielhavir/capsule-network

import torch
import torch.nn as nn
import torch.optim as optim
import os
from numpy import prod
from datetime import datetime
from time import time
from models.model_conv import CapsuleNetwork
from utils.losses.losses import CapsuleLoss

CHECKPOINTS_PATH = "checkpoints/"
if not os.path.exists(CHECKPOINTS_PATH):
    os.mkdir(CHECKPOINTS_PATH)


class CapsNetTrainer:
    def __init__(
        self,
        loaders,
        batch_size,
        learning_rate,
        routing_steps=3,
        lr_decay=0.9,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        multi_gpu=(torch.cuda.device_count() > 1),
        routing_algorithm="softmax",
    ):
        self.device = device
        self.multi_gpu = multi_gpu

        self.loaders = loaders
        img_shape = self.loaders["train"].dataset[0][0].numpy().shape

        self.network = CapsuleNetwork(
            img_shape,
            channels=3,
            primary_dim=8,
            num_classes=2,
            output_dim=16,
            routing_steps=routing_steps,
            device=self.device,
            routing_algorithm=routing_algorithm,
        ).to(self.device)

        from torchinfo import summary

        if os.getenv("DEBUG") == "1":
            print("Torch Model Summary")
            summary(
                self.network,
                input_size=(batch_size, 3, img_shape[1], img_shape[2]),
                device=self.device.type,
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                    "mult_adds",
                ],
            )

        if self.multi_gpu:
            self.network = nn.DataParallel(self.network)

        self.loss_function = CapsuleLoss(loss_lambda=0.4, recontruction_loss_scale=5e-4)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_decay
        )
        print("Torch Model Built")
        print(
            f"Total params count: {sum([prod(p.size()) for p in self.network.parameters()])}"
        )
        print(
            f"Trainable params count: {sum([prod(p.size()) for p in self.network.parameters() if p.requires_grad])}"
        )
        print(f"Device: {self.device}")
        print(f"Multi GPU: {self.multi_gpu}")

    def __repr__(self):
        return repr(self.network)

    def run(self, epochs, classes, callback_manager=None):
        print(8 * "#", "Run started".upper(), 8 * "#")
        eye = torch.eye(len(classes)).to(self.device)

        for epoch in range(1, epochs + 1):
            for phase in ["train", "test"]:
                print(f"{phase}ing...".capitalize())
                if phase == "train":
                    self.network.train()
                else:
                    self.network.eval()

                t0 = time()
                running_loss = 0.0
                correct = 0
                total = 0
                for i, (images, labels) in enumerate(self.loaders[phase]):
                    t1 = time()
                    images, labels = images.to(self.device), labels.to(self.device)
                    labels = eye[labels]  # one-hot encoding

                    self.optimizer.zero_grad()

                    outputs, reconstructions = self.network(images)
                    loss = self.loss_function(outputs, labels, images, reconstructions)

                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    _, labels = torch.max(labels, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    accuracy = float(correct) / float(total)

                    if phase == "train":
                        print(
                            f"Epoch {epoch}, Batch {i+1}, Loss {running_loss/(i+1)}",
                            f"Accuracy {accuracy} Time {round(time()-t1, 3)}s",
                        )

                        if callback_manager is not None:
                            callback_manager.on_batch_end(
                                batch=i,
                                logs={
                                    "loss": running_loss / (i + 1),
                                    "accuracy": accuracy,
                                    "epoch": epoch,
                                    "phase": phase,
                                },
                            )

                print(
                    f"{phase.upper()} Epoch {epoch}, Loss {running_loss/(i+1)}",
                    f"Accuracy {accuracy} Time {round(time()-t0, 3)}s",
                )

                if callback_manager is not None:
                    callback_manager.on_epoch_end(
                        epoch=epoch,
                        logs={
                            "loss": running_loss / (i + 1),
                            "accuracy": accuracy,
                            "epoch": epoch,
                            "phase": phase,
                        },
                    )

                # Periodically visualize reconstructions (e.g. every 5 epochs on test phase)
                if phase == "test":
                    callback_manager.on_reconstruction(
                        images[:8], reconstructions[:8], epoch, phase
                    )

            self.scheduler.step()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        error_rate = round((1 - accuracy) * 100, 2)
        torch.save(
            self.network.state_dict(),
            os.path.join(CHECKPOINTS_PATH, f"{error_rate}_{now}.pth.tar"),
        )

        class_correct = list(0.0 for _ in classes)
        class_total = list(0.0 for _ in classes)
        for images, labels in self.loaders["test"]:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs, reconstructions = self.network(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i in range(len(classes)):
            print(
                "Accuracy of %5s : %2d %%"
                % (classes[i], 100 * class_correct[i] / class_total[i])
            )
