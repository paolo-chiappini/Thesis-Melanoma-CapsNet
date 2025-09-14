from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn

from losses.losses import AttributeLoss
from losses.msr_loss import MaskedMSELoss


class HungarianLoss(nn.Module):
    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_recon: float = 1.0,
        empty_cls_weight: float = 0.1,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_recon = lambda_recon
        self.empty_cls_weight = empty_cls_weight

        self.cls_criterion = AttributeLoss()
        self.recon_criterion = MaskedMSELoss(background_penalization=0.1)

    def _get_source_permutation_idx(
        self, indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        source_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(
        self, indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat(
            [torch.full_like(target, i) for i, (_, target) in enumerate(indices)]
        )
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def loss_classification(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        source_idx = self._get_source_permutation_idx(indices)
        source_logits = outputs["attribute_logits"][source_idx]

        # get optimally-associated target
        target_idx = self._get_target_permutation_idx(indices)
        target_labels_ordered = targets["visual_attributes_targets"][target_idx]

        pos_weight = torch.ones_like(target_labels_ordered)
        weights = torch.where(
            target_labels_ordered > 0.5, pos_weight, self.empty_cls_weight
        )

        num_capsules = outputs["attribute_logits"].shape[1]
        # loss = F.binary_cross_entropy_with_logits(source_logits, target_labels_ordered, weight=weights, reduction="sum")

        self.cls_criterion.attribute_weights = weights
        loss = self.cls_criterion(source_logits, target_labels_ordered)

        return loss / num_capsules

    def loss_reconstruction(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        decoder: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        source_idx = self._get_source_permutation_idx(indices)
        # get optimally-associated target
        target_idx = self._get_target_permutation_idx(indices)
        target_labels_ordered = targets["visual_attributes_targets"][target_idx]

        is_present_mask = target_labels_ordered > 0.5

        device = outputs["attribute_poses"].device

        if not is_present_mask.any():
            return torch.tensor(0.0, device=device)

        # select associated masks
        ordered_masks = targets["va_masks"][target_idx]
        target_masks_present = ordered_masks[is_present_mask]

        loss_msr = 0.0
        num_present_matches = is_present_mask.sum()

        batch_indices_present = source_idx[0][is_present_mask]
        source_capsule_indices_present = source_idx[1][is_present_mask]

        for k in range(num_present_matches):
            # get the data for the i-th present match
            batch_idx = batch_indices_present[k]
            source_capsule_idx = source_capsule_indices_present[k]

            all_poses_for_sample = outputs["attribute_poses"][
                batch_idx
            ]  # (K_attr, pose_dim)

            masked_poses = torch.zeros_like(all_poses_for_sample)
            masked_poses[source_capsule_idx] = all_poses_for_sample[source_capsule_idx]

            # (1, K, pose_dim)
            reconstruction = decoder(masked_poses.unsqueeze(0))  # Shape: (1, C, H, W)

            image_target = targets["images"][batch_idx].unsqueeze(0)
            mask_target = target_masks_present[k].unsqueeze(0)

            loss_msr += self.recon_criterion(reconstruction, image_target, mask_target)

        num_capsules = outputs["attribute_logits"].shape[1]
        loss = loss_msr / num_capsules

        return loss

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        decoder: Callable[[torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_cls = self.loss_classification(outputs, targets, indices)
        loss_recon = self.loss_reconstruction(outputs, targets, indices, decoder)

        total_loss = self.lambda_cls * loss_cls + self.lambda_recon * loss_recon

        return {
            "loss_total": total_loss,
            "loss_cls": loss_cls,
            "loss_recon": loss_recon,
        }
