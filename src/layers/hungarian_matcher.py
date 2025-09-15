from typing import Callable

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from losses.msr_loss import MaskedMSELoss


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        lambda_cls: float = 1,
        lambda_recon: float = 1,
        recon_criterion: nn.Module = None,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_recon = lambda_recon

        if lambda_cls <= 0 and lambda_recon <= 0:
            raise ValueError(
                "At least one of lambda_cls or lambda_recon must be positive."
            )

        if recon_criterion is None:
            print(
                "No recon_criterion provided to HungarianMatcher. Defaulting to MMSE."
            )
            self.recon_criterion = MaskedMSELoss(background_penalization=0.1)
        else:
            self.recon_criterion = recon_criterion

    @torch.no_grad()
    def forward(self, outputs: dict, targets: dict, decoder: Callable):
        """
        Computes optimal matching using the Hungarian algorithm

        Args:
            outputs (dict): dictionary containing these entries:
                "attribute_logits": predictions for the presence (or absence) of an attribute.
                "attribute_poses": capsule poses for the attributes.
            targets (dict): dictionary of targets containing these entries:
                "visual_attribute_targets": binary ground truth labels for attributes.
                "images": input images.
                "va_masks": attribute-specific segmentation masks.
            decoder (function): model's decoder to perform reconstructions.

        Returns:
            A list containing tuples (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        batch_size, num_capsules = outputs["attribute_logits"].shape
        _, _, pose_dim = outputs["attribute_poses"].shape
        device = outputs["attribute_logits"].device

        # (N_batch, K_attr)
        attribute_targets = targets["visual_attributes_targets"].float()
        num_attributes = attribute_targets.shape[1]

        # (N_batch, K_preds)
        logits = outputs["attribute_logits"]

        # (N_batch, K_preds, 1) -> (N_batch, K_preds, K_attr)
        logits_expanded = logits.unsqueeze(2).expand(
            batch_size, num_capsules, num_attributes
        )  # unsqueeze for broadcasting

        # (N_batch, 1, K_attr) -> (N_batch, K_preds, K_attr)
        target_expanded = attribute_targets.unsqueeze(1).expand(
            batch_size, num_capsules, num_attributes
        )

        cost_cls = F.binary_cross_entropy_with_logits(
            logits_expanded, target_expanded, reduction="none"
        )

        poses = outputs["attribute_poses"]  # (N_batch, K_attr, pose_dim)
        images = targets["images"]  # (N_batch, K_attr, C, H, W)
        masks = targets["va_masks"]  # (N_batch, K_attr, H, W)

        eye = torch.eye(num_attributes, device=device).view(
            1, num_attributes, num_attributes, 1
        )
        masked_poses_batch = poses.unsqueeze(1) * eye

        # (N_batch * K_attr, C, H, W)
        masked_poses_flat = masked_poses_batch.view(
            batch_size * num_attributes, num_attributes, pose_dim
        )

        recons_flat = decoder(masked_poses_flat)
        recons = recons_flat.view(batch_size, num_attributes, *recons_flat.shape[1:])

        cost_recon = torch.zeros(
            batch_size, num_attributes, num_attributes, device=device
        )

        for j in range(num_attributes):
            target_mask_j = masks[:, j, :, :].unsqueeze(1)
            for i in range(num_attributes):
                recons_i = recons[:, i, :, :, :]
                loss_ij = self.recon_criterion(recons_i, images, target_mask_j)
                cost_recon[:, i, j] = loss_ij

        is_present = (
            (attribute_targets > 0.5).float().unsqueeze(1)
        )  # (N_batch, 1, K_attr)
        cost_recon = cost_recon * is_present

        C = self.lambda_cls * cost_cls + self.lambda_recon * cost_recon
        C_cpu = C.detach().cpu().numpy()

        indices = [linear_sum_assignment(cost_matrix) for cost_matrix in C_cpu]

        return [
            (
                torch.as_tensor(i, dtype=torch.long, device=device),
                torch.as_tensor(j, dtype=torch.long, device=device),
            )
            for i, j in indices
        ]
