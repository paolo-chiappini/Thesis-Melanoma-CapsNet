# credits: https://github.com/danielhavir/capsule-network

import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from time import time
from utils.losses.losses import CombinedLoss
from tqdm import tqdm

CHECKPOINTS_PATH = "checkpoints/"
if not os.path.exists(CHECKPOINTS_PATH):
    os.mkdir(CHECKPOINTS_PATH)


class CapsNetTrainer:
    def __init__(
        self,
        loaders,
        batch_size,
        learning_rate,
        lr_decay=0.9,
        network=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        multi_gpu=(torch.cuda.device_count() > 1),
        criterion=CombinedLoss,
    ):
        self.device = device
        self.multi_gpu = multi_gpu

        self.loaders = loaders
        img_shape = self.loaders["train"].dataset[0][0].numpy().shape

        assert network is not None, "Network architecture must be defined"
        self.network = network.to(self.device)

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

        self.loss_function = criterion
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_decay
        )

    def __repr__(self):
        return repr(self.network)

    def run(self, epochs, classes, visual_attributes_all, callback_manager=None):
        print(8 * "=", "Run started".upper(), 8 * "=")
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
                loader = tqdm(
                    enumerate(self.loaders[phase]),
                    total=len(self.loaders[phase]),
                    desc=f"{phase.capitalize()} Epoch {epoch}",
                )
                for i, batch in loader:
                    images = batch[0]
                    labels = batch[1]
                    visual_attributes = batch[2]
                    masks = batch[3]
                    # va_masks = batch[4] # TODO: add back in trainig

                    t1 = time()
                    images, labels, masks = (
                        images.to(self.device),
                        labels.to(self.device),
                        masks.to(self.device),
                    )
                    labels = eye[labels]  # one-hot encoding

                    self.optimizer.zero_grad()

                    outputs, reconstructions, malignancy_scores, _ = self.network(
                        images
                    )
                    loss = self.loss_function(
                        outputs,
                        visual_attributes,
                        malignancy_scores,
                        labels,
                        images,
                        reconstructions,
                        masks,
                    )

                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item()

                    # _, predicted = torch.max(outputs, 1)
                    # _, labels = torch.max(labels, 1)
                    _, predicted = torch.max(malignancy_scores, 1)
                    _, labels = torch.max(labels, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    accuracy = float(correct) / float(total)

                    if phase == "train":
                        loader.set_postfix(
                            loss=running_loss / (i + 1), accuracy=accuracy
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
        attributes_correct = list(0.0 for _ in visual_attributes)
        attributes_total = list(0.0 for _ in visual_attributes)
        for batch in self.loaders["test"]:
            images = batch[0]
            labels = batch[1]
            gt_visual_attributes = batch[2]
            masks = batch[3]
            # TODO: Add va masks

            images, labels, visual_attributes, masks = (
                images.to(self.device),
                labels.to(self.device),
                visual_attributes.to(self.device),
                masks.to(self.device),
            )

            outputs, reconstructions, malignancy_scores, _ = self.network(images)
            _, predicted = torch.max(malignancy_scores, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            # Attribute accuracy
            binarized_vas = (outputs > 0.5).float()
            correct_vas = (binarized_vas == gt_visual_attributes).float()

            for attr in range(gt_visual_attributes.size(1)):
                attributes_correct[attr] += correct_vas[:, attr].sum().item()
                attributes_total[attr] += gt_visual_attributes.size(0)

        for i in range(len(classes)):
            print(
                "Accuracy of %5s : %2d %%"
                % (classes[i], 100 * class_correct[i] / class_total[i])
            )

        for i in range(len(visual_attributes_all)):
            acc = (
                100 * attributes_correct[i] / attributes_total[i]
                if attributes_total[i] > 0
                else 0
            )
            print(
                "Accuracy of attribute %5s : %2.2f%%" % (visual_attributes_all[i], acc)
            )
