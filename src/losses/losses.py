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
