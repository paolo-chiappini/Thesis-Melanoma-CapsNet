# credits: https://github.com/danielhavir/capsule-network

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AttributeLoss(nn.Module):
    def __init__(
        self,
        loss_lambda: float = 1.0,
        attribute_weights: Optional[List[float]] = None,
        loss_criterion=F.binary_cross_entropy_with_logits,
        **kwargs
    ):
        super(AttributeLoss, self).__init__()
        self.loss_lambda = loss_lambda
        self.loss_criterion = loss_criterion
        self.attribute_weights = None

        if attribute_weights is not None:
            self.attribute_weights = torch.tensor(
                attribute_weights, dtype=torch.float32
            )

    def forward(
        self, attribute_scores: torch.Tensor, attribute_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the attribute loss.

        Parameters:
        - attribute_scores (torch.Tensor): Predicted attribute scores (batch_size, num_attributes)
        - attribute_targets (torch.Tensor): Target attribute scores (batch_size, num_attributes)

        Returns:
        - loss (torch.Tensor): Computed loss value
        """
        device = attribute_scores.device
        attribute_targets = attribute_targets.to(device)

        loss = self.loss_criterion(
            attribute_scores,
            attribute_targets,
            pos_weight=self.attribute_weights,
        )

        return self.loss_lambda * loss


class MalignancyLoss(nn.Module):
    def __init__(
        self,
        loss_lambda: float = 1.0,
        class_weights: Optional[List[float]] = None,
        loss_criterion=F.binary_cross_entropy_with_logits,
        **kwargs
    ):
        super(MalignancyLoss, self).__init__()
        self.loss_lambda = loss_lambda
        self.loss_criterion = loss_criterion

        if class_weights is not None:
            self.register_buffer("pos_weight", torch.tensor(class_weights))
        else:
            self.pos_weight = None

    def forward(
        self, malignancy_scores: torch.Tensor, malignancy_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the malignancy loss.

        Parameters:
        - malignancy_scores (torch.Tensor): Predicted malignancy scores (batch_size, num_malignancies)
        - malignancy_targets (torch.Tensor): Target malignancy scores (batch_size, num_malignancies)

        Returns:
        - loss (torch.Tensor): Computed loss value
        """
        return self.loss_lambda * self.loss_criterion(
            malignancy_scores, malignancy_targets, pos_weight=self.pos_weight
        )


class MultiLabelCapsuleMarginLoss(nn.Module):
    def __init__(
        self,
        m_plus: float = 0.9,
        m_minus: float = 0.1,
        lambda_: float = 0.5,
        loss_lambda: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
        self.loss_lambda = loss_lambda

    def forward(
        self, attribute_poses: torch.Tensor, attribute_targets: torch.Tensor
    ) -> torch.Tensor:
        lengths = torch.norm(attribute_poses, dim=-1)

        left = F.relu(self.m_plus - lengths) ** 2
        right = F.relu(lengths - self.m_minus) ** 2
        loss = attribute_targets * left + self.lambda_ * (1 - attribute_targets) * right

        return self.loss_lambda * loss.mean()


class MultiLabelLogitMarginLoss(nn.Module):
    def __init__(
        self, margin: float = 1.0, loss_lambda: float = 1.0, base_loss=None, **kwargs
    ):
        super().__init__()
        self.margin = margin
        self.base_loss = base_loss or nn.BCEWithLogitsLoss()
        self.loss_lambda = loss_lambda

    def forward(
        self, attribute_logits: torch.Tensor, attribute_targets: torch.Tensor
    ) -> torch.Tensor:
        base = self.base_loss(attribute_logits, attribute_targets)

        pos_mask = attribute_targets == 1
        neg_mask = attribute_targets == 0

        pos_penalty = (
            F.relu(self.margin - attribute_logits[pos_mask]).mean()
            if pos_mask.any()
            else 0
        )
        neg_penalty = (
            F.relu(self.margin + attribute_logits[neg_mask]).mean()
            if neg_mask.any()
            else 0
        )

        margin_loss = pos_penalty + neg_penalty
        return (base + margin_loss) * self.loss_lambda


class ContrastivePoseLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, loss_lambda: float = 1.0, **kwargs):
        """
        An InfoNCE-based contrastive loss to enforce separation between capsule poses
        for present (positive) and absent (negative) attributes.

        Args:
            temperature (float): The temperature parameter for the softmax.
            loss_lambda (float): The weight of this loss component.
        """
        super(ContrastivePoseLoss, self).__init__()
        self.temperature = temperature
        self.lambda_contrastive = loss_lambda
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

        return self.lambda_contrastive * loss
