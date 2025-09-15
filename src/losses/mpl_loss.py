from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

VGG_LAYERS = {
    "conv1_2": 4,
    "conv2_2": 9,
    "conv3_3": 16,
    "conv4_3": 23,
}


class VGGFeatureExtractor(nn.Module):
    """
    Helper to extract features from VGG layers
    """

    def __init__(self, layers: List[int]):
        super().__init__()
        self.vgg_features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.layers = sorted(layers)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for i, layer in enumerate(self.vgg_features):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features


class MaskedPerceptualLossVGG16(nn.Module):
    def __init__(
        self,
        background_penalization: float = 0.0,
    ):
        """
        Args:
            background_penalization (float): Weight for background contribution. 0.0 ignores the background.
        """
        super(MaskedPerceptualLossVGG16, self).__init__()
        self.background_penalization = background_penalization
        self.feature_extractor = VGGFeatureExtractor(layers=VGG_LAYERS.values())
        self.preprocess = VGG16_Weights.DEFAULT.transforms()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Recostructed image batch (N, C, H, W).
            target (torch.Tensor): Groud-truth image batch (N, C, H, W).
            mask (torch.Tensor): Binary mask (N, 1, H, W).

        Returns:
            torch.Tensor: Calculated scalar MPL.
        """
        self.feature_extractor = self.feature_extractor.to(input.device)
        self.feature_extractor.eval()

        input_preprocessed = self.preprocess(input)
        target_preprocessed = self.preprocess(target)

        with torch.no_grad():
            input_features = self.feature_extractor(input_preprocessed)
            target_features = self.feature_extractor(target_preprocessed)

        total_loss = 0.0

        for i in range(len(input_features)):
            in_feature = input_features[i]
            tgt_feature = target_features[i]

            resized_mask = F.interpolate(
                mask, size=in_feature.shape[-2:], mode="bilinear", align_corners=False
            )

            masked_input_feature = in_feature * resized_mask
            masked_target_feature = tgt_feature * resized_mask

            foreground_loss = self.mse_loss(masked_input_feature, masked_target_feature)

            mask_sum = resized_mask.sum() * in_feature.shape[1] + 1e-8
            normalized_foreground_loss = foreground_loss / mask_sum

            total_loss += normalized_foreground_loss

            if self.background_penalization > 0:
                inverted_mask = 1 - resized_mask

                background_input_feature = in_feature * inverted_mask
                background_target_feature = tgt_feature * inverted_mask

                background_loss = self.mse_loss(
                    background_input_feature, background_target_feature
                )

                inverted_mask_sum = inverted_mask.sum() * in_feature.shape[1] + 1e-8
                normalized_background_loss = background_loss / inverted_mask_sum

                total_loss += self.background_penalization * normalized_background_loss

        return total_loss / len(input_features)


class MSRPerceptualLoss(nn.Module):
    def __init__(
        self, alpha: float, beta: float, lambda_bg_global: float, lambda_bg_local: float
    ):
        """
        Args:
            alpha (float): Global reconstruction weight.
            beta (float): Local reconstruction weight.
            lambda_bg_global (float): Background penalization weight for global reconstructions.
            lambda_bg_local (float): Local penalization weight for global reconstructions.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mpl_global = MaskedPerceptualLossVGG16(
            background_penalization=lambda_bg_global
        )
        self.mpl_local = MaskedPerceptualLossVGG16(
            background_penalization=lambda_bg_local
        )

    def forward(
        self,
        image: torch.Tensor,
        global_reconstruction: torch.Tensor,
        attribute_reconstructions: List[torch.Tensor],
        lesion_mask: torch.Tensor,
        attribute_masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): Ground-truth image batc.
            global_reconstruction (torch.Tensor): The global reconstruction from all capsules.
            attribute_reconstructions (List[torch.Tesnor]): List of local reconstructions, one per capsule.
            lesion_mask (torch.Tensor): The global lesion mask.
            attribute_masks (List[torch.Tensor]): List of local masks for each visual attribute.

        Returns:
            torch.Tensor: The final combined loss.
        """
        global_loss = self.mpl_global(global_reconstruction, image, lesion_mask)

        local_losses = []
        num_capsules = attribute_reconstructions.shape[1]

        for k in range(num_capsules):
            reconstruction_k = attribute_reconstructions[:, k, :, :, :]

            mask_k = attribute_masks[:, k, :, :].unsqueeze(1)

            loss_k = self.mpl_local(reconstruction_k, image, mask_k)
            local_losses.append(loss_k)

        total_local_loss = torch.stack(local_losses).mean()

        final_loss = self.alpha * global_loss + self.beta * total_local_loss

        return final_loss
