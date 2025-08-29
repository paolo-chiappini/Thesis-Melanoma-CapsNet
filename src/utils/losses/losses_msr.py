import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        """
        Calculates the Mean Squared Error on the input and target tensors,
        but only for the pixels where the mask is > 0.

        Args:
        - input (torch.Tensor): The predicted tensor (e.g., reconstruction).
        - target (torch.Tensor): The ground truth tensor (e.g., original image).
        - mask (torch.Tensor): A binary mask tensor of the same shape.

        Returns:
        - loss (torch.Tensor): The masked MSE loss.
        """
        # Ensure mask is broadcastable to the same shape as input/target
        mask = mask.expand_as(target)

        squared_error = F.mse_loss(input, target, reduction="none")

        # Apply the mask. This zeroes out the error where the mask is 0.
        masked_squared_error = squared_error * mask

        num_pixels_in_mask = mask.sum()
        loss = masked_squared_error.sum() / (num_pixels_in_mask + 1e-8)

        return loss
