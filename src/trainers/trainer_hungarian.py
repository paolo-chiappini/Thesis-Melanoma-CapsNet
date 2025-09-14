import torch

from layers import HungarianMatcher
from losses.hungarian_loss import HungarianLoss
from utils.commons import compute_weighted_accuracy
from utils.visualization.plot_reconstruction_examples import (
    plot_reconstruction_examples,
)

from .trainer_base import BaseTrainer

_CORRECT_THRESHOLD = 0.5


class CapsNetTrainerHungarian(BaseTrainer):
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
        **kwargs
    ):
        super().__init__(
            model,
            loaders,
            criterion,
            optimizer,
            scheduler,
            device,
            checkpoints_dir,
            save_name,
            metrics,
            **kwargs
        )
        self.msr_alpha = kwargs.get("msr_alpha", 5.0)
        self.msr_beta = kwargs.get("msr_beta", 5.0)
        self.global_bg_pen = kwargs.get("global_bg_pen", 1.0)
        self.local_bg_pen = kwargs.get("local_bg_pen", 0.1)
        self.global_normalize = kwargs.get("global_normalize", "total")
        self.local_normalize = kwargs.get("local_normalize", "total")
        self.lambda_cls = kwargs.get("lambda_cls", 1.0)
        self.lambda_recon = kwargs.get("lambda_recon", 1.0)

        empty_cls_weight = kwargs.get("empty_cls_weight", 0.1)

        # TODO: Change this
        self.matcher = HungarianMatcher(self.lambda_cls, self.lambda_recon)
        self.loss_fn = HungarianLoss(
            self.lambda_cls, self.lambda_recon, empty_cls_weight
        )

    def compute_loss(self, outputs, batch_data):
        matching_indices = self.matcher(
            outputs=outputs, targets=batch_data, decoder=self.model.decode
        )

        loss_dict = self.loss_fn(
            outputs=outputs,
            targets=batch_data,
            indices=matching_indices,
            decoder=self.model.decode,
        )

        if self.current_phase == "val" and self.current_batch == len(
            self.loaders["val"]
        ):
            self.visualize_reconstructions(outputs, batch_data)

        additional_loss = self.criterion(model_outputs=outputs, targets=batch_data)
        loss_dict.update(additional_loss)

        return loss_dict

    def visualize_reconstructions(self, outputs, batch_data):
        images = batch_data["images"]
        lesion_masks = batch_data["lesion_masks"]
        va_masks = batch_data["va_masks"]
        attribute_poses = outputs["attribute_poses"]

        N, K, pose_dim = attribute_poses.shape

        global_reconstruction = self.model.decode(attribute_poses)

        local_reconstructions = []
        for k in range(K):
            pose_mask = torch.zeros_like(attribute_poses)
            pose_mask[:, k, :] = 1
            local_reconstruction_k = self.model.decode(attribute_poses, pose_mask)
            local_reconstructions.append(local_reconstruction_k)

        local_recons_stacked = torch.stack(local_reconstructions, dim=1)

        visual_attrs = self.loaders["val"].dataset.dataset.visual_attributes

        plot_reconstruction_examples(
            images=images,
            global_recons=global_reconstruction,
            capsule_recons=local_recons_stacked,
            lesion_masks=lesion_masks,
            va_masks=va_masks,
            epoch=self.current_epoch,
            phase=self.current_phase,
            va_mask_labels=visual_attrs,
            logger=self.logger,
            max_capsules=len(visual_attrs),
        )

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
