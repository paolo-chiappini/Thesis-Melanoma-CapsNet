import torch

from losses.msr_loss import MaskedMSELoss
from utils.commons import compute_weighted_accuracy
from utils.visualization.plot_reconstruction_examples import (
    plot_reconstruction_examples,
)

from .trainer_base import BaseTrainer

_CORRECT_THRESHOLD = 0.5


class CapsNetTrainerMSR(BaseTrainer):

    def compute_loss(self, outputs, batch_data):
        images = batch_data["images"]
        lesion_masks = batch_data["lesion_masks"]
        va_masks = batch_data["va_masks"]
        attribute_poses = outputs["attribute_poses"]

        total_loss = self.criterion(model_outputs=outputs, targets=batch_data)

        alpha = 5.0
        beta = 10.0

        global_recon_criterion = MaskedMSELoss(background_penalization=1.0)
        local_recon_criterion = MaskedMSELoss(background_penalization=1.0)

        global_reconstruction = self.model.decode(attribute_poses)
        loss_global_recon = global_recon_criterion(
            global_reconstruction, images, lesion_masks
        )

        loss_msr = 0.0
        N, K, pose_dim = attribute_poses.shape
        H, W = images.shape[2], images.shape[3]

        local_reconstructions = []
        for k in range(K):
            pose_mask = torch.zeros_like(attribute_poses, device=attribute_poses.device)
            pose_mask[:, k, :] = 1

            local_reconstruction_k = self.model.decode(attribute_poses, pose_mask)
            if self.current_phase == "val" and self.current_batch == len(
                self.loaders["val"]
            ):
                local_reconstructions.append(local_reconstruction_k)

            masks_k = va_masks[:, k, :, :].unsqueeze(1)

            loss_msr_k = local_recon_criterion(local_reconstruction_k, images, masks_k)
            loss_msr += loss_msr_k
        loss_msr_avg = loss_msr / K

        total_recon_loss = alpha * loss_global_recon + beta * loss_msr_avg

        if self.current_phase == "val" and self.current_batch == len(
            self.loaders["val"]
        ):
            plot_reconstruction_examples(
                images=images,
                global_recons=global_reconstruction,
                capsule_recons=torch.stack(local_reconstructions, dim=1),
                lesion_masks=lesion_masks,
                va_masks=va_masks,
                epoch=self.current_epoch,
                phase=self.current_phase,
                va_mask_labels=self.loaders["val"].dataset.dataset.visual_attributes,
                logger=self.logger,
            )

        total_loss.update({"msr_loss": total_recon_loss})
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
