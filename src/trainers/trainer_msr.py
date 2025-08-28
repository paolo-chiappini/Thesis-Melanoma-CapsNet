from .trainer_base import BaseTrainer
import torch
import torch.nn as nn
from utils.commons import compute_weighted_accuracy
from utils.losses.losses_msr import MaskedMSELoss

_CORRECT_THRESHOLD = 0.5


class CapsNetTrainerMSR(BaseTrainer):
    def prepare_batch(self, batch):
        return {
            "inputs": batch["image"].to(self.device),
            "labels": batch["label"].to(self.device),
            "visual_attributes": batch["visual_features"].to(self.device),
            "masks": batch["segmentation"].to(self.device),
            "va_masks": batch["va_masks"].to(self.device),
        }

    def compute_loss(self, outputs, batch_data):
        # TODO: use unpack method here?
        images = batch_data["inputs"]
        masks = batch_data["masks"]
        va_masks = batch_data["va_masks"]

        attribute_logits, reconstructions, malignancy_scores, attribute_poses = outputs
        total_loss = self.criterion(
            attribute_logits,
            attribute_poses,
            batch_data["visual_attributes"],
            malignancy_scores,
            torch.eye(len(malignancy_scores[0])).to(self.device)[
                batch_data["labels"]
            ],  # one hot encoded labels
            images,
            reconstructions,
            masks,
        )

        # TODO: temporary MSR Loss implementation
        alpha = 0.5
        beta = 0.5

        global_recon_criterion = nn.MSELoss()
        local_recon_criterion = MaskedMSELoss()

        global_reconstruction = self.model.decode(attribute_poses)
        loss_global_recon = global_recon_criterion(global_reconstruction, images)

        loss_msr = 0.0
        N, K, pose_dim = attribute_poses.shape
        H, W = images.shape[2], images.shape[3]

        for k in range(K):
            single_capsule_poses = torch.zeros_like(
                attribute_poses, device=attribute_poses.device
            )
            single_capsule_poses[:, k, :] = attribute_poses[
                :, k, :
            ]  # isolate the k-th capsule

            local_reconstruction_k = self.model.decode(single_capsule_poses)

            masks_k = va_masks[:, k, :, :].unsqueeze(1)

            loss_msr_k = local_recon_criterion(local_reconstruction_k, images, masks_k)
            loss_msr += loss_msr_k
        loss_msr_avg = loss_msr / K

        total_recon_loss = alpha * loss_global_recon + beta * loss_msr_avg

        total_loss.update({"msr_loss": total_recon_loss})
        return total_loss

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
