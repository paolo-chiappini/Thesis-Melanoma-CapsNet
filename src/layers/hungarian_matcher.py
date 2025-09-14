from typing import Callable

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    def __init__(self, lambda_cls: float = 1, lambda_recon: float = 1):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_recon = lambda_recon

        if lambda_cls <= 0 and lambda_recon <= 0:
            raise ValueError(
                "At least one of lambda_cls or lambda_recon must be positive."
            )

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

        # (N_batch, K_preds, C, H, W)
        reconstructions = []
        for k in range(num_capsules):
            pose_mask = torch.zeros_like(poses, device=device)
            pose_mask[:, k, :] = 1
            reconstructions.append(decoder(poses, pose_mask))

        # (N_batch, K_preds, C,  H,  W)
        recons_expanded = torch.stack(reconstructions, dim=1)
        # (N_batch, 1,       C,  H,  W)
        images_expanded = images.unsqueeze(1)
        # (N_batch, 1,       K_attr,    1,  C,  H,  W)
        masks_expanded = masks.unsqueeze(2).unsqueeze(1)

        # (N_batch, K_preds, C, H, W) -> (N_batch, K_preds, 1, C, H, W)
        error_sq = ((recons_expanded - images_expanded) ** 2).unsqueeze(2)
        # (N_batch, K_preds, K_attr, C, H, W)
        masked_error = error_sq * masks_expanded

        sum_masked_error = masked_error.sum(dim=(-3, -2, -1))

        mask_area = masks.sum(dim=(-2, -1)).clamp(min=1e-6)  # avoid zero division

        cost_recon = sum_masked_error / mask_area.squeeze(-1).unsqueeze(1)

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
