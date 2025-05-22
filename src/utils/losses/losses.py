# credits: https://github.com/danielhavir/capsule-network

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    def __init__(self, size_average=False, loss_lambda=0.5):
        """
        Lk = Tk max(0, m+ - ||vk||)2 + λ (1 - Tk) max(0, ||vk|| - m-)2      (4)
        """
        super(MarginLoss, self).__init__()
        self.size_average = size_average
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.loss_lambda = loss_lambda

    def forward(self, inputs, labels):
        L_k = (
            labels * F.relu(self.m_plus - inputs) ** 2
            + self.loss_lambda * (1 - labels) * F.relu(inputs - self.m_minus) ** 2
        )
        return L_k.mean() if self.size_average else L_k.sum()


class CapsuleLoss(nn.Module):
    def __init__(
        self, loss_lambda=0.5, recontruction_loss_scale=5e-4, size_average=False
    ):
        """
            Combined loss: L_margin + L_reconstruction (SSE was used as reconstruction)

            Params:
            - recontruction_loss_scale: 	param for scaling down the the reconstruction loss.
        - size_average:		    if True, reconstruction loss becomes MSE instead of SSE.
        """
        super(CapsuleLoss, self).__init__()
        self.size_average = size_average
        self.margin_loss = MarginLoss(
            size_average=size_average, loss_lambda=loss_lambda
        )
        self.reconstruction_loss = nn.MSELoss(size_average=size_average)
        self.recontruction_loss_scale = recontruction_loss_scale

    def forward(self, inputs, labels, images, reconstructions):
        margin_loss = self.margin_loss(inputs, labels)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        caps_loss = margin_loss + self.recontruction_loss_scale * reconstruction_loss

        return caps_loss


class AttributeLoss(nn.Module):
    def __init__(self, loss_lambda=1.0, loss_criterion=F.binary_cross_entropy):
        super(AttributeLoss, self).__init__()
        self.loss_lambda = loss_lambda
        self.loss_criterion = loss_criterion

    def forward(self, attribute_scores, attribute_targets):
        """
        Compute the attribute loss.

        Parameters:
        - attribute_scores: Predicted attribute scores (batch_size, num_attributes)
        - attribute_targets: Target attribute scores (batch_size, num_attributes)

        Returns:
        - loss: Computed loss value
        """
        return self.loss_lambda * self.loss_criterion(
            attribute_scores, attribute_targets
        )


class MalignancyLoss(nn.Module):
    def __init__(self, loss_lambda=1.0, loss_criterion=F.mse_loss):
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


class CombinedLoss(nn.Module):
    def __init__(
        self,
        margin_loss_lambda=0.4,
        reconstruction_loss_scale=5e-4,
        attribute_loss_lambda=1.0,
        malignancy_loss_lambda=1.0,
    ):
        super(CombinedLoss, self).__init__()
        self.capsule_loss = CapsuleLoss(
            loss_lambda=margin_loss_lambda,
            recontruction_loss_scale=reconstruction_loss_scale,
        )
        self.attribute_loss = AttributeLoss(loss_lambda=attribute_loss_lambda)
        self.malignancy_loss = MalignancyLoss(loss_lambda=malignancy_loss_lambda)

    def forward(
        self,
        capsule_outputs,
        attribute_targets,
        malignancy_scores,
        targets,
        images,
        reconstructions,
    ):
        capsule_loss = self.capsule_loss(
            malignancy_scores, targets, images, reconstructions
        )
        attribute_loss = self.attribute_loss(capsule_outputs, attribute_targets)
        # malignancy_loss = self.malignancy_loss # Not implemented

        total_loss = capsule_loss + attribute_loss
        return total_loss
