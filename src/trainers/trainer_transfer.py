from .trainer_base import BaseTrainer
import torch
from utils.commons import compute_weighted_accuracy


class TransferTrainer(BaseTrainer):
    def prepare_batch(self, batch):
        images, labels = batch
        return {
            "inputs": images.to(self.device),
            "labels": labels.to(self.device),
            "visual_attributes": labels.to(
                self.device
            ),  # Trick to use the same base trainer
        }

    def compute_loss(self, outputs, batch_data):
        return self.criterion(outputs, batch_data["labels"])

    def unpack_model_outputs(self, outputs):
        return {"preds": outputs}

    def compute_custom_metrics(self, outputs, batch_data):
        outputs_dict = self.unpack_model_outputs(outputs)
        logits = outputs_dict["preds"]

        _, predicted = torch.max(logits, 1)
        labels = batch_data["labels"]

        weighted_accuracy = compute_weighted_accuracy(
            predicted=predicted,
            target=labels,
            weights=self.class_weights,
            num_labels=logits.size(1),
        )

        return {"accuracy": weighted_accuracy}
