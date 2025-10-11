# credits: https://github.com/danielhavir/capsule-network

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.functional.helpers import get_primary_caps_coords


class MarginLoss(nn.Module):
    def __init__(self, size_average=False, loss_lambda=0.5, **kwargs):
        """
        Lk = Tk max(0, m+ - ||vk||)2 + λ (1 - Tk) max(0, ||vk|| - m-)2      (4)
        """
        super(MarginLoss, self).__init__()
        self.size_average = size_average
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.loss_lambda = loss_lambda

    def forward(self, inputs, targets):
        L_k = (
            targets * F.relu(self.m_plus - inputs) ** 2
            + self.loss_lambda * (1 - targets) * F.relu(inputs - self.m_minus) ** 2
        )
        return L_k.mean() if self.size_average else L_k.sum()


def focal_loss_with_logits(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes Focal Loss for binary classification targets with logits input.

    Args:
        inputs (torch.Tensor): Raw logits (scores) from the model.
        targets (torch.Tensor): Ground truth labels (0 or 1).
        gamma (float): Focusing parameter. Higher gamma reduces the loss contribution
                       from easy-to-classify examples.
        alpha (float): Weighting factor for the positive class (often 0.25).
                       Use pos_weight for per-instance/class weighting instead if available.
        pos_weight (torch.Tensor, optional): Weight for positive examples, as in BCE.
                                            If provided, it overrides alpha's role in positive weighting.

    Returns:
        torch.Tensor: The computed focal loss.
    """
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    pt = torch.exp(-BCE_loss)

    focus_term = (1 - pt) ** gamma

    if pos_weight is not None:
        weighting_term = targets * pos_weight.to(inputs.device) + (1 - targets) * 1.0
    elif alpha is not None:
        alpha = torch.tensor(alpha).to(inputs.device)
        weighting_term = targets * alpha + (1 - targets) * (1 - alpha)
    else:
        weighting_term = 1.0

    loss = weighting_term * focus_term * BCE_loss

    return loss.mean()


class AttributeLoss(nn.Module):
    def __init__(
        self,
        attribute_weights: Optional[
            List[float]
        ] = None,  # Pos_weight for each attribute
        **kwargs
    ):
        super(AttributeLoss, self).__init__()

        self.loss_criterion = F.binary_cross_entropy_with_logits

        if attribute_weights is not None:
            self.register_buffer(
                "attribute_weights",
                torch.tensor(attribute_weights, dtype=torch.float32),
            )
        else:
            self.attribute_weights = None

    def forward(
        self, attribute_scores: torch.Tensor, attribute_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the attribute loss using Multi-label Soft Margin (BCE with Logits).

        Parameters:
        - attribute_scores (torch.Tensor): Predicted logits (B, N_attributes)
        - attribute_targets (torch.Tensor): Target labels (B, N_attributes)
        """
        attribute_targets = attribute_targets.float()

        loss = self.loss_criterion(
            attribute_scores,
            attribute_targets,
            pos_weight=self.attribute_weights,
        )

        return loss


class MalignancyLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[List[float]] = None,  # Used as pos_weight in Focal Loss
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = 0.25,
        **kwargs
    ):
        super(MalignancyLoss, self).__init__()
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        if class_weights is not None:
            # class_weights should be [pos_weight] for the positive (malignant) class
            self.register_buffer(
                "pos_weight", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.pos_weight = None

    def forward(
        self, malignancy_scores: torch.Tensor, malignancy_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the malignancy loss using Focal Loss.

        Parameters:
        - malignancy_scores (torch.Tensor): Predicted malignancy scores (batch_size, 1)
        - malignancy_targets (torch.Tensor): Target malignancy scores (batch_size, 1)
        """
        malignancy_targets = malignancy_targets.float()

        loss = focal_loss_with_logits(
            inputs=malignancy_scores,
            targets=malignancy_targets,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            pos_weight=self.pos_weight,
        )

        return loss


class ContrastivePoseLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, **kwargs):
        """
        An InfoNCE-based contrastive loss to enforce separation between capsule poses
        for present (positive) and absent (negative) attributes.

        Args:
            temperature (float): The temperature parameter for the softmax.
            loss_lambda (float): The weight of this loss component.
        """
        super(ContrastivePoseLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(
        self, attribute_poses: torch.Tensor, va_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            attribute_poses (torch.Tensor): Shape (N, K, pose_dim). The capsule poses.
            va_labels (torch.Tensor): Shape (N, K). The binary ground-truth labels.
        """
        N, K, pose_dim = attribute_poses.shape
        device = attribute_poses.device

        # reshape for vectorization
        # poses: (N, K, d) -> (N * K, d)
        # labels: (N, K) -> (N * K,)
        poses = attribute_poses.view(N * K, pose_dim)
        labels = va_labels.view(N * K)

        # 1. pairwise cosine-sim
        poses = F.normalize(poses, p=2, dim=-1)

        # similarity_matrix shape: (N * K, N * K)
        similarity_matrix = torch.matmul(poses, poses.T) / self.temperature

        # 2. create masks for positive pairs and self-comparisons
        labels = labels.unsqueeze(1)  # reshape for broadcasting: (N*K, 1)

        positive_mask = (labels == labels.T).float()
        identity_mask = torch.eye(N * K, device=device)
        positive_mask = positive_mask - identity_mask

        has_positive_pairs = positive_mask.sum(dim=1) > 0
        if not has_positive_pairs.any():
            return torch.tensor(0.0, device=device)  # no valid pairs in the batch

        # 3. compute the SupCon loss on valid pairs
        sim_matrix_anchors = similarity_matrix[has_positive_pairs]
        pos_mask_anchors = positive_mask[has_positive_pairs]

        self_mask = identity_mask[has_positive_pairs]

        logits = sim_matrix_anchors - (
            self_mask * 1e9
        )  # subtract a large number from self-similarity to avoid log(0)

        # stabilize logits by subtracting the max value per row for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        log_sum_exp_denominator = torch.log(
            torch.exp(logits).sum(dim=1, keepdim=True) + 1e-8
        )

        # 4. compute final loss
        # loss formula: L = -1/|P(i)| * Σ_p [sim_ip - log_sum_exp_denominator]
        log_prob = logits - log_sum_exp_denominator

        sum_log_prob_pos = (pos_mask_anchors * log_prob).sum(dim=1)

        num_positive_pairs = pos_mask_anchors.sum(dim=1)

        loss_per_anchor = -sum_log_prob_pos / (num_positive_pairs + 1e-8)

        loss = loss_per_anchor.mean()

        return loss


class DisentanglementLoss(nn.Module):
    def __init__(self, **kwargs):
        super(DisentanglementLoss, self).__init__()

    def forward(self, attribute_poses: torch.Tensor) -> torch.Tensor:
        B, K, pose_dim = attribute_poses.shape

        # mean-center for correlation computation
        mean_poses = torch.mean(attribute_poses, dim=[0, 2], keepdim=True)
        centered_poses = attribute_poses - mean_poses

        reshaped_poses = centered_poses.permute(0, 2, 1).reshape(B * pose_dim, K)

        C = torch.matmul(reshaped_poses.T, reshaped_poses) / (B * pose_dim - 1)

        diag_mask = torch.eye(K, device=C.device).bool()

        penalty = torch.sum(C[~diag_mask] ** 2)

        return penalty


class MaskAlignmentLoss(nn.Module):

    def __init__(self, **kwargs):
        super(MaskAlignmentLoss, self).__init__()

    def forward(
        self, predicted_masks: torch.Tensor, gt_masks: torch.Tensor
    ) -> torch.Tensor:
        pred_flat = predicted_masks.view(-1)
        gt_flat = gt_masks.view(-1)

        intersection = (pred_flat * gt_flat).sum()

        areas_sum = pred_flat.sum() + gt_flat.sum()

        smooth = 1e-6
        dice_coefficient = (2.0 * intersection + smooth) / (areas_sum + smooth)

        dice_loss = 1 - dice_coefficient

        return dice_loss


class RoutingAgreementLoss(nn.Module):
    def __init__(self, img_size: int = 256, **kwargs):
        super(RoutingAgreementLoss, self).__init__()
        self.img_size = img_size

        self.register_buffer("primary_coords", get_primary_caps_coords(img_size))

    def forward(
        self, coupling_coefficients: torch.Tensor, gt_masks: torch.Tensor
    ) -> torch.Tensor:
        B, num_primary, num_attribute = coupling_coefficients.shape
        H, W = gt_masks.shape[2:]
        device = coupling_coefficients.device

        if H != self.img_size or W != self.img_size:
            raise ValueError("Mask size must match the input image size.")

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        primary_coords_int = (
            self.primary_coords.round().long().clamp(0, H - 1).to(device=device)
        )

        mask_value_at_primary_caps_center = gt_masks[
            :, :, primary_coords_int[:, 1], primary_coords_int[:, 0]
        ].permute(
            0, 2, 1
        )  # (B, num_primary, num_attribut)

        outside_mask_penalty_map = 1.0 - mask_value_at_primary_caps_center.float()

        loss_agreement = (coupling_coefficients * outside_mask_penalty_map).mean()

        return loss_agreement
