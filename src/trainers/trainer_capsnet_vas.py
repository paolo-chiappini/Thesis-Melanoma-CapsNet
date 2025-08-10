from .trainer_base import BaseTrainer
import torch
from utils.commons import compute_weighted_accuracy

_CORRECT_THRESHOLD = 0.5


class CapsNetTrainerVAs(BaseTrainer):
    def prepare_batch(self, batch):
        images, labels, visual_attributes, masks = batch
        return {
            "inputs": images.to(self.device),
            "labels": labels.to(self.device),
            "visual_attributes": visual_attributes.to(self.device),
            "masks": masks.to(self.device),
        }

    def compute_loss(self, outputs, batch_data):
        attribute_logits, reconstructions, malignancy_scores, attribute_poses = outputs
        return self.criterion(
            attribute_logits,
            attribute_poses,
            batch_data["visual_attributes"],
            malignancy_scores,
            torch.eye(len(malignancy_scores[0])).to(self.device)[
                batch_data["labels"]
            ],  # one hot encoded labels
            batch_data["inputs"],
            reconstructions,
            batch_data["masks"],
        )

    def compute_custom_metrics(self, outputs, batch_data):
        outputs_dict = self.unpack_model_outputs(outputs)

        _, predicted = torch.max(outputs_dict["malignancy"], 1)
        labels = batch_data["labels"].to(self.device)

        weighted_accuracy = compute_weighted_accuracy(
            predicted=predicted,
            target=labels,
            weights=self.class_weights,
            num_labels=outputs_dict["malignancy"].size(1),
        )

        predicted_vas = outputs_dict["preds"]
        predicted_vas = (predicted_vas >= _CORRECT_THRESHOLD).float()  # binarize
        attributes = batch_data["visual_attributes"].to(self.device)

        weighted_accuracy_vas = compute_weighted_accuracy(
            predicted=predicted_vas,
            target=attributes,
            weights=self.attribute_weights,
            num_labels=attributes.shape[1],
        )

        return {
            "accuracy": weighted_accuracy,
            "accuracy_vas": weighted_accuracy_vas,
        }

    def unpack_model_outputs(self, outputs):
        va_scores, reconstructions, malignancy_scores, capsules = outputs
        return {
            "preds": va_scores,
            "reconstructions": reconstructions,
            "malignancy": malignancy_scores,
            "capsule_poses": capsules,
        }
