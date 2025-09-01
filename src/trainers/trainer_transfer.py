from .trainer_base import BaseTrainer
import torch
import torch.nn as nn
from utils.commons import compute_weighted_accuracy


class TransferTrainer(BaseTrainer):
    def __init__(self, model, loaders, criterion, optimizer, scheduler=None, device="cuda", checkpoints_dir="checkpoints", save_name=None, metrics=None):
        super().__init__(model, loaders, criterion, torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4), scheduler, device, checkpoints_dir, save_name, metrics)

    def set_weights(self, weights_dict):
        super().set_weights(weights_dict)
        if self.class_weights is not None:
             self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
             print(f"Criterion updated with weights: {self.class_weights}")
        else:
             self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def prepare_batch(self, batch):
        images, labels = batch['image'], batch['label']
        return {
            "inputs": images.to(self.device),
            "labels": labels.to(self.device),
            "visual_attributes": labels.to(
                self.device
            ),  # Trick to use the same base trainer
        }

    def compute_loss(self, outputs, batch_data):
        logits = outputs
        labels = batch_data["labels"]
        
        if labels.ndim > 1 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        elif labels.ndim > 1:
            raise ValueError(f"Labels tensor has incorrect shape {labels.shape}. Expected (batch_size,).")
        
        return self.criterion(logits, labels.long())

    def unpack_model_outputs(self, outputs):
        return {"preds": outputs}

    def compute_custom_metrics(self, outputs, batch_data):
        logits = self.unpack_model_outputs(outputs)["preds"]
        labels = batch_data["labels"]

        if labels.ndim > 1 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        _, predicted = torch.max(logits, 1)

        acc = (predicted == labels).float().mean().item()
        
        return {"accuracy": acc}