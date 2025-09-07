import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    def __init__(self, background_penalization=0.0, normalize="total"):
        """
        Args:
            background_penalization (float): Weight for background MSE.
            normalize (str): 'total' (normalize over all pixels),
                             'foreground' (normalize only by lesion pixels),
                             or 'per_region' (normalize each region separately).
        """
        super(MaskedMSELoss, self).__init__()
        self.background_penalization = background_penalization
        assert normalize in ["total", "foreground", "per_region"]
        self.normalize = normalize

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
        mask = mask.expand_as(target).float()

        squared_error = F.mse_loss(input, target, reduction="none")

        lesion_error = squared_error * mask
        background_error = squared_error * (1 - mask)

        if self.normalize == "total":
            # Normalize over all pixels
            total_pixels = float(input.numel())
            loss = (
                lesion_error.sum()
                + self.background_penalization * background_error.sum()
            ) / total_pixels

        elif self.normalize == "foreground":
            # Normalize over lesion pixels only
            lesion_pixels = mask.sum() + 1e-8
            loss = (
                lesion_error.sum()
                + self.background_penalization * background_error.sum()
            ) / lesion_pixels

        elif self.normalize == "per_region":
            # Normalize each region separately
            lesion_pixels = mask.sum() + 1e-8
            background_pixels = (1 - mask).sum() + 1e-8

            lesion_loss = lesion_error.sum() / lesion_pixels
            background_loss = background_error.sum() / background_pixels

            loss = lesion_loss + self.background_penalization * background_loss

        return loss
