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


class WeightedMarginLoss(nn.Module):
    def __init__(
        self, class_weights=None, gamma=2.0, size_average=False, loss_lambda=0.5
    ):
        super(WeightedMarginLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.size_average = size_average
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.loss_lambda = loss_lambda

    def forward(self, inputs, labels):
        L_k = (
            labels * F.relu(self.m_plus - inputs) ** 2
            + self.loss_lambda * (1 - labels) * F.relu(inputs - self.m_minus) ** 2
        )

        # Focal modulation to emphasize imbalanced classes
        if self.gamma > 0:
            pt = inputs * labels + (1 - inputs) * (1 - labels)
            mod_factor = (1 - pt).pow(self.gamma)
            L_k *= mod_factor

        if self.class_weights is not None:
            class_weights = self.class_weights
            L_k *= class_weights.unsqueeze(0)

        return L_k.mean() if self.size_average else L_k.sum()


class CapsuleLoss(nn.Module):
    def __init__(
        self,
        loss_lambda=0.5,
        reconstruction_loss_scale=5e-4,
        size_average=False,
        focal_gamma=2.0,
        class_weights=None,
    ):
        """
        Combined loss: L_margin + L_reconstruction (SSE was used as reconstruction)

        Params:
        - recontruction_loss_scale: param for scaling down the the reconstruction loss.
        - size_average: if True, reconstruction loss becomes MSE instead of SSE.
        """
        super(CapsuleLoss, self).__init__()
        self.size_average = size_average
        self.margin_loss = WeightedMarginLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            size_average=size_average,
            loss_lambda=loss_lambda,
        )
        self.reconstruction_loss = nn.MSELoss(size_average=size_average)
        self.reconstruction_loss_scale = reconstruction_loss_scale

    def forward(self, inputs, labels, images, reconstructions, masks):
        margin_loss = self.margin_loss(inputs, labels)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images * masks)
        caps_loss = margin_loss + self.reconstruction_loss_scale * reconstruction_loss

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


class SegmentationLoss(nn.Module):
    def __init__(self, loss_lamdba=1.0, smooth=1e-6):
        super(SegmentationLoss, self).__init__()
        self.loss_lambda = loss_lamdba
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, segmentations, segmentations_targets):
        """
        Compute the segmentation loss.

        Args:
        - segmentations: predicted segmentations mask (batch_size, image_shape).
        - segmentations_targets  predicted segmentations mask (batch_size, image_shape).

        Returns:egmentation loss.
        - loss: Computed loss value
        """
        segmentations = torch.sigmoid(segmentations)
        bce_loss = self.bce(segmentations, segmentations_targets)
        intersection = (segmentations * segmentations_targets).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            segmentations.sum(dim=(1, 2, 3))
            + segmentations_targets.sum(dim=(1, 2, 3))
            + self.smooth
        )
        dice_loss = 1 - dice.mean()
        return self.loss_lambda * (bce_loss + dice_loss)


class MaskedReconstructionLoss(nn.Module):
    def __init__(
        self,
        loss_lamdba=1.0,
        background_lambda=0.0,
        reconstruction_loss_scale=1e-4,
        size_average=False,
    ):
        super(MaskedReconstructionLoss, self).__init__()
        self.loss_lambda = loss_lamdba
        self.background_lambda = background_lambda
        self.reconstruction_loss = nn.MSELoss(size_average=size_average)
        self.recontruction_loss_scale = reconstruction_loss_scale

    def forward(self, reconstructions, masks, inputs):
        """
        Compute the reconstruction loss masked to the lesion segmentation mask.
        The background is still accounted for, but it is penalized by the λ_background term.

        Args:
        - reconstructions: reconstructed outputs from capsules.
        - masks: ground-truth segmentation masks for the lesions.
        - inputs: input images to the network.

        Returns:
        - loss: Computed loss value
        """
        # masked_loss = self.reconstruction_loss(reconstructions * masks, inputs * masks)
        masked_loss = self.reconstruction_loss(
            reconstructions, inputs * masks
        )  # Only inputs are masked to favor automatic segmentation.

        # background_loss = self.reconstruction_loss(
        #     reconstructions * (1 - masks), inputs * (1 - masks)
        # )
        # return self.loss_lambda(masked_loss + background_loss * self.background_lambda)
        return self.loss_lambda * masked_loss


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
        class_weights=None,
        margin_loss_lambda=0.4,
        margin_loss_gamma=2.0,
        reconstruction_loss_scale=5e-4,
        attribute_loss_lambda=1.0,
        malignancy_loss_lambda=1.0,
        segmentation_loss_lambda=1.0,
    ):
        super(CombinedLoss, self).__init__()
        self.capsule_loss = CapsuleLoss(
            loss_lambda=margin_loss_lambda,
            reconstruction_loss_scale=reconstruction_loss_scale,
            focal_gamma=margin_loss_gamma,
            class_weights=class_weights,
        )
        self.attribute_loss = AttributeLoss(loss_lambda=attribute_loss_lambda)
        self.malignancy_loss = MalignancyLoss(loss_lambda=malignancy_loss_lambda)
        self.segmentaion_loss = SegmentationLoss(loss_lamdba=segmentation_loss_lambda)

    def forward(
        self,
        capsule_outputs,
        attribute_targets,
        malignancy_scores,
        targets,
        images,
        reconstructions,
        masks,
    ):
        capsule_loss = self.capsule_loss(
            malignancy_scores, targets, images, reconstructions, masks
        )
        attribute_loss = self.attribute_loss(capsule_outputs, attribute_targets)
        # malignancy_loss = self.malignancy_loss # Not implemented

        total_loss = capsule_loss + attribute_loss
        return total_loss
