import torch

from utils.commons import compute_weighted_accuracy
from utils.visualization.plot_reconstruction_examples import (
    plot_reconstruction_examples,
)

from .trainer_base import BaseTrainer

_CORRECT_THRESHOLD = 0.5


class CapsNetTrainerMPL(BaseTrainer):
    def compute_loss(self, outputs, batch_data):
        images = batch_data["images"]
        lesion_masks = batch_data["lesion_masks"]
        va_masks = batch_data["va_masks"]
        global_reconstructions = outputs["reconstructions"]
        local_reconstructions = outputs["attribute_reconstructions"]

        total_loss = self.criterion(model_outputs=outputs, targets=batch_data)

        if self.current_phase == "val" and self.current_batch == len(
            self.loaders["val"]
        ):
            visual_attrs = self.loaders["val"].dataset.dataset.visual_attributes
            plot_reconstruction_examples(
                images=images,
                global_recons=global_reconstructions,
                capsule_recons=local_reconstructions,
                lesion_masks=lesion_masks,
                va_masks=va_masks,
                epoch=self.current_epoch,
                phase=self.current_phase,
                va_mask_labels=visual_attrs,
                logger=self.logger,
                max_capsules=len(visual_attrs),
            )

        return total_loss

    def compute_custom_metrics(self, outputs, batch_data):
        outputs_dict = self.unpack_model_outputs(outputs)

        _, predicted = torch.max(outputs_dict["malignancy_scores"], 1)
        labels = batch_data["malignancy_targets"].to(self.device)

        weighted_accuracy = compute_weighted_accuracy(
            predicted=predicted,
            target=labels,
            weights=self.class_weights,
            num_labels=outputs_dict["malignancy_scores"].size(1),
        )

        predicted_vas = outputs_dict["logits"]
        predicted_vas = (predicted_vas >= _CORRECT_THRESHOLD).float()  # binarize
        attributes = batch_data["visual_attributes_targets"].to(self.device)

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
        outputs.update({"logits": outputs["attribute_logits"]})
        return outputs
