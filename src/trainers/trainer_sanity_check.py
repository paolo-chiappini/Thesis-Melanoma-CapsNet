from utils.commons import compute_weighted_accuracy
from .trainer_base import BaseTrainer


class SanityCheckTrainer(BaseTrainer):
    def prepare_batch(self, batch):
        images, labels, visual_attributes, masks = batch
        return {
            "inputs": images.to(self.device),
            "labels": labels.to(self.device),
            "visual_attributes": visual_attributes.to(self.device),
            "masks": masks.to(self.device),
        }

    def compute_loss(self, outputs, batch_data):
        predicted_vas = outputs
        return self.criterion(
            predicted_vas,
            batch_data["visual_attributes"],
        )

    def compute_custom_metrics(self, outputs, batch_data):
        predicted_vas = outputs
        predicted_vas = (predicted_vas >= 0.5).float()  # binarize
        attributes = batch_data["visual_attributes"].to(self.device)

        weighted_accuracy_vas = compute_weighted_accuracy(
            predicted=predicted_vas,
            target=attributes,
            num_labels=attributes.shape[1],
        )

        return {
            "accuracy": weighted_accuracy_vas,
        }

    def unpack_model_outputs(self, outputs):
        return {"preds": outputs}
