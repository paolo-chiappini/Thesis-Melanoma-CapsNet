import torch
from models.model_conv_attributes import CapsuleNetworkWithAttributes
from utils.losses.losses import CombinedLoss
from .trainer_base import CapsNetTrainerBase


class CapsNetTrainer(CapsNetTrainerBase):
    def __init__(
        self,
        loaders,
        learning_rate,
        routing_steps=3,
        lr_decay=0.9,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        multi_gpu=(torch.cuda.device_count() > 1),
        routing_algorithm="softmax",
    ):
        img_shape = loaders["train"].dataset[0][0].numpy().shape
        num_attributes = loaders["train"].dataset[0][2].shape[0]

        self.network = CapsuleNetworkWithAttributes(
            img_shape,
            channels=3,
            primary_dim=8,
            num_classes=2,
            num_attributes=num_attributes,
            output_dim=16,
            routing_steps=routing_steps,
            device=device,
            routing_algorithm=routing_algorithm,
        )

        super().__init__(
            loaders,
            learning_rate,
            lr_decay,
            device,
            multi_gpu,
            self.network,
            CombinedLoss(),
        )

    def _run_batch(self, batch, eye, phase):
        images, labels, visual_attributes = batch
        images, labels = images.to(self.device), labels.to(self.device)
        labels_one_hot = eye[labels]

        self.optimizer.zero_grad()

        outputs, reconstructions, malignancy_scores = self.network(images)
        loss = self.loss_function(
            outputs,
            visual_attributes,
            malignancy_scores,
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
