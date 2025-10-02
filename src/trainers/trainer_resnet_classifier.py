import torch

from .trainer_base import BaseTrainer


class ResnetClassifierTrainer(BaseTrainer):
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
        metrics=None,
    ):
        super().__init__(
            model,
            loaders,
            criterion,
            torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4),
            scheduler,
            device,
            checkpoints_dir,
            save_name,
            metrics,
        )

    # def set_weights(self, weights_dict):
    #     super().set_weights(weights_dict)
    #     if self.class_weights is not None:
    #         self.criterion = nn.CrossEntropyLoss(
    #             weight=self.class_weights, label_smoothing=0.1
    #         )
    #         print(f"Criterion updated with weights: {self.class_weights}")
    #     else:
    #         self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def prepare_batch(self, batch: dict):
        batch.update(
            {
                "visual_attributes_targets": batch[
                    "malignancy_targets"
                ],  # Trick to use the same base trainer
            }
        )
        return super().prepare_batch(batch=batch)

    def compute_loss(self, outputs, batch_data):
        return self.criterion(model_outputs=outputs, targets=batch_data)

    def unpack_model_outputs(self, outputs):
        return {"logits": outputs["malignancy_scores"]}

    def compute_custom_metrics(self, outputs, batch_data):
        logits = self.unpack_model_outputs(outputs)["logits"]
        labels = batch_data["malignancy_targets"]

        if labels.ndim > 1 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        _, predicted = torch.max(logits, 1)

        acc = (predicted == labels).float().mean().item()

        return {"accuracy": acc}
