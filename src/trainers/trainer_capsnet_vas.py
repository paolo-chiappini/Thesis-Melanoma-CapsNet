from .trainer_base import BaseTrainer
import torch

_OUT_VAS_SCORES_IDX = 0
_OUT_MALIGNANCY_SCORES_IDX = 2
_CORRECT_THRESHOLD = 0.75


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
        predicted_vas, reconstructions, malignancy_scores, class_pose_vectors = outputs
        return self.criterion(
            predicted_vas,
            batch_data["visual_attributes"],
            malignancy_scores,
            torch.eye(len(malignancy_scores[0])).to(self.device)[
                batch_data["labels"]
            ],  # one hot encoded labels
            batch_data["inputs"][0],
            reconstructions,
            batch_data["masks"],
        )

    def compute_metrics(self, outputs, batch_data):
        _, predicted = torch.max(outputs[_OUT_MALIGNANCY_SCORES_IDX], 1)
        predicted_vas = outputs[_OUT_VAS_SCORES_IDX]
        accuracy = (predicted == batch_data["labels"]).float().mean().item()

        predicted_vas = (predicted_vas >= _CORRECT_THRESHOLD).float()  # binarize
        accuracy_vas = (
            (predicted_vas == batch_data["visual_attributes"]).float().mean().item()
        )

        return {"accuracy": accuracy, "accuracy_vas": accuracy_vas}
