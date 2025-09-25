# credits: https://github.com/danielhavir/capsule-network

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
        loss_lambda=1.0,
        loss_criterion=F.binary_cross_entropy_with_logits,
        attribute_weights=None,
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

    def forward(self, attribute_scores, attribute_targets):
        """
        Compute the attribute loss.

        Parameters:
        - attribute_scores: Predicted attribute scores (batch_size, num_attributes)
        - attribute_targets: Target attribute scores (batch_size, num_attributes)

        Returns:
        - loss: Computed loss value
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
        loss_lambda=1.0,
        loss_criterion=F.binary_cross_entropy_with_logits,
        **kwargs
    ):
        super(MalignancyLoss, self).__init__()
        self.loss_lambda = loss_lambda
        self.loss_criterion = loss_criterion

    def forward(self, malignancy_scores, malignancy_targets):
        """
        Compute the malignancy loss.

        Parameters:
        - malignancy_scores: Predicted malignancy scores (batch_size, num_malignancies)
        - malignancy_targets: Target malignancy scores (batch_size, num_malignancies)

        Returns:
        - loss: Computed loss value
        """
        return self.loss_lambda * self.loss_criterion(
            malignancy_scores, malignancy_targets
        )
