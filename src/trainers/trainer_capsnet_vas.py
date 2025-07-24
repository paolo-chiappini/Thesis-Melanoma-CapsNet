from .trainer_base import BaseTrainer
import torch

__OUT_MALIGNANCY_SCORES_IDX = 2


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
        outputs, reconstructions, malignancy_scores, class_pose_vectors = outputs
        return self.criterion(
            outputs,
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
        _, predicted = torch.max(outputs[__OUT_MALIGNANCY_SCORES_IDX], 1)
        accuracy = (predicted == batch_data["labels"]).float().mean().item()
        return {"accuracy": accuracy}
